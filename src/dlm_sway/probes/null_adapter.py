"""Null-adapter baseline probe — per-kind calibration matrix (S02).

Every numeric primitive reports its raw metric *and* a z-score against
a null-adapter distribution. This probe is the runtime engine that
establishes that distribution — for **every** numeric probe kind the
user has downstream in the suite, not just one.

How it works:

1. The runner populates ``ctx.downstream_kinds`` with every probe kind
   that appears after this one in the suite.
2. For each target kind, we ask its probe class for a
   :meth:`~dlm_sway.probes.base.Probe.calibrate_spec` — a small spec
   suitable for null calibration. A probe that returns ``None`` opts
   out (typically because its inputs can't be synthesized, e.g.
   ``adapter_revert`` without an embedder, or ``adapter_ablation``
   which needs ``as_scaled_adapter`` that the proxy doesn't expose).
3. For each calibrating kind × seed, we run the probe through a
   :class:`~dlm_sway.probes._null_proxy.NullCalibrationBackendProxy`
   which makes ``as_finetuned()`` yield ``as_null_adapter(seed)`` —
   so the probe's own math is computing "what does my metric look
   like when the fine-tune is structural noise?".
4. We harvest each run's ``raw`` value, aggregate to ``(mean, std, n)``
   per kind, and publish under ``evidence["null_stats"]``.
5. The runner threads ``null_stats`` into ``RunContext`` for every
   subsequent probe, which then prefers the z-score path over the
   fixed-threshold path (see :mod:`dlm_sway.probes._zscore`).

Backends that don't implement
:class:`~dlm_sway.core.scoring.NullCalibratedBackend` cause this probe
to ``Verdict.SKIP``; every downstream probe falls back to fixed
thresholds and surfaces ``(no calibration)`` in the report.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field

from dlm_sway.core.result import ProbeResult, Verdict, safe_finalize
from dlm_sway.core.scoring import NullCalibratedBackend
from dlm_sway.probes._null_cache import compute_key, load, save
from dlm_sway.probes._null_proxy import NullCalibrationBackendProxy
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext, registry


class NullAdapterSpec(ProbeSpec):
    """Spec for ``kind: null_adapter``.

    Place this probe **first** in the suite so its output populates
    :attr:`RunContext.null_stats` before subsequent probes consult it.
    """

    kind: Literal["null_adapter"] = "null_adapter"
    runs: int = Field(default=3, ge=1, le=10)
    """Number of independent null adapters to evaluate. Three is the
    smallest that yields a usable std; more is better but quickly
    dominates suite runtime."""
    init_scale: float = 0.02
    """Stddev of the zero-mean Gaussian used to fill lora_A/lora_B."""
    seed_base: int = 1000
    """First seed; successive runs use ``seed_base + run_idx``."""
    calibrate_kinds: list[str] = Field(default_factory=list)
    """Which probe kinds to calibrate. Empty = auto-populate from
    ``ctx.downstream_kinds`` (the kinds that appear after this probe
    in the suite). Set explicitly to force calibration of specific
    kinds regardless of suite order."""
    cache: bool = True
    """Read / write the on-disk calibration cache under
    ``~/.dlm-sway/null-stats``. Keyed by backend identity + calibration
    params. Disable to force a fresh calibration (e.g. when you suspect
    the cached stats are stale)."""
    rank_multipliers: list[float] = Field(default_factory=lambda: [1.0])
    """Rank multipliers at which to calibrate. Each value scales the
    null-adapter noise std by ``sqrt(multiplier)`` — mathematically
    equivalent to rank-scaling the LoRA output variance. Default
    ``[1.0]`` preserves pre-S10 single-rank behavior byte-for-byte.

    Three-point profiles like ``[0.5, 1.0, 2.0]`` let users read
    "how rank-saturated is my adapter?" off the report:

    - A healthy adapter's z-score is stable across multipliers.
    - An adapter that's barely above noise at its own rank but
      solidly above noise at ``0.5x`` is rank-saturated — a smaller
      rank would have yielded a sharper signal.

    Per-rank stats land in ``evidence["null_stats_by_rank"]`` keyed
    by ``f"rank_{mult:.2f}"``; the 1.0x group (when present) also
    lands under ``evidence["null_stats"]`` for back-compat with
    probes that consume a single calibration level.
    """


class NullAdapterProbe(Probe):
    """Populate ``ctx.null_stats`` with per-kind null distributions.

    The probe itself reports ``Verdict.PASS`` on success — its job is
    calibration, not judgment. If the backend can't support null-view
    substitution, reports ``Verdict.SKIP`` with a clear message; every
    downstream numeric probe then falls back to fixed thresholds.
    """

    kind = "null_adapter"
    spec_cls = NullAdapterSpec
    category = "baseline"

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        assert isinstance(spec, NullAdapterSpec)
        if not isinstance(ctx.backend, NullCalibratedBackend):
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.SKIP,
                score=None,
                message=(
                    "backend does not implement NullCalibratedBackend — "
                    "numeric probes will fall back to fixed thresholds"
                ),
            )

        registered = registry()

        # Decide which kinds to calibrate. Explicit spec field wins;
        # otherwise auto-populate from downstream_kinds.
        target_kinds: list[str] = list(spec.calibrate_kinds)
        if not target_kinds:
            target_kinds = [k for k in ctx.downstream_kinds if k and k != spec.kind]
        # De-dupe while preserving order; drop self and unregistered.
        seen: set[str] = set()
        filtered: list[str] = []
        for k in target_kinds:
            if k == spec.kind or k in seen or k not in registered:
                continue
            seen.add(k)
            filtered.append(k)
        target_kinds = filtered

        # Validate rank multipliers up front; empty list is nonsensical.
        rank_multipliers = list(spec.rank_multipliers) or [1.0]
        for mult in rank_multipliers:
            if mult <= 0.0 or not math.isfinite(mult):
                return ProbeResult(
                    name=spec.name,
                    kind=spec.kind,
                    verdict=Verdict.ERROR,
                    score=None,
                    message=f"rank_multipliers must be positive and finite; got {mult!r}",
                )

        # Cache lookup: backends can opt in by providing a
        # ``cache_identity()`` method returning a stable string. The
        # key incorporates both that identity and the calibration
        # parameters that actually influence the output — including
        # the sorted rank-multiplier tuple so multi-rank caches don't
        # collide with single-rank.
        cache_key: str | None = None
        if spec.cache:
            backend_identity = _backend_identity(ctx.backend)
            cache_key = compute_key(
                backend_identity=backend_identity,
                params={
                    "runs": spec.runs,
                    "init_scale": spec.init_scale,
                    "seed_base": spec.seed_base,
                    "top_k": ctx.top_k,
                    "kinds": sorted(target_kinds),
                    "rank_multipliers": sorted(rank_multipliers),
                },
            )
            cached = load(cache_key)
            if cached is not None and "null_stats_by_rank" in cached:
                return _pass_from_cache(spec, cached)
            # Pre-S10 cache entries only have ``null_stats`` (implicit
            # single-rank). Promote them into the new shape so repeated
            # runs benefit from the existing cache.
            if cached is not None and "null_stats" in cached:
                promoted = dict(cached)
                promoted["null_stats_by_rank"] = {
                    _rank_key(1.0): cached["null_stats"],
                }
                return _pass_from_cache(spec, promoted)

        null_stats_by_rank: dict[str, dict[str, dict[str, float]]] = {}
        per_rank_skipped: dict[str, list[dict[str, str]]] = {}
        per_rank_samples: dict[str, dict[str, list[float]]] = {}

        for mult in rank_multipliers:
            rkey = _rank_key(mult)
            per_kind_stats, samples, skipped = _calibrate_at_rank(
                ctx=ctx,
                spec=spec,
                target_kinds=target_kinds,
                registered=registered,
                rank_scale=mult,
            )
            null_stats_by_rank[rkey] = per_kind_stats
            per_rank_samples[rkey] = samples
            per_rank_skipped[rkey] = skipped

        # Back-compat surface: ``null_stats`` is the 1.0x group when
        # present, else the first multiplier's stats (so older probes
        # that only read the single-rank dict still get *something*).
        primary_rkey = _rank_key(1.0)
        if primary_rkey in null_stats_by_rank:
            primary_stats = null_stats_by_rank[primary_rkey]
            primary_skipped = per_rank_skipped[primary_rkey]
            primary_samples = per_rank_samples[primary_rkey]
        else:
            first_rkey = _rank_key(rank_multipliers[0])
            primary_stats = null_stats_by_rank[first_rkey]
            primary_skipped = per_rank_skipped[first_rkey]
            primary_samples = per_rank_samples[first_rkey]

        evidence: dict[str, Any] = {
            "null_stats": primary_stats,
            "null_stats_by_rank": null_stats_by_rank,
            "per_kind_raw_samples": primary_samples,
            "skipped_kinds": primary_skipped,
            "calibrated_kinds": list(primary_stats.keys()),
            "runs": spec.runs,
            "init_scale": spec.init_scale,
            "seed_base": spec.seed_base,
            "rank_multipliers": rank_multipliers,
            "weight": spec.weight,
            "from_cache": False,
        }

        if cache_key is not None:
            save(
                cache_key,
                {
                    "null_stats": primary_stats,
                    "null_stats_by_rank": null_stats_by_rank,
                    "runs": spec.runs,
                    "init_scale": spec.init_scale,
                    "seed_base": spec.seed_base,
                    "rank_multipliers": rank_multipliers,
                    "calibrated_kinds": list(primary_stats.keys()),
                },
            )

        if len(rank_multipliers) == 1:
            message = (
                f"null calibration: {len(primary_stats)} kinds calibrated over {spec.runs} seeds"
            )
        else:
            mults_str = ", ".join(f"{m:g}x" for m in rank_multipliers)
            message = (
                f"null calibration: {len(primary_stats)} kinds × "
                f"{len(rank_multipliers)} ranks [{mults_str}] over {spec.runs} seeds"
            )
        if primary_skipped:
            message += f" ({len(primary_skipped)} opted out)"

        return safe_finalize(
            name=spec.name,
            kind=spec.kind,
            verdict=Verdict.PASS,
            score=1.0,
            evidence=evidence,
            message=message,
        )


def _rank_key(mult: float) -> str:
    """Canonical string key for a rank multiplier. Stable across runs."""
    return f"rank_{mult:.2f}"


def _calibrate_at_rank(
    *,
    ctx: RunContext,
    spec: NullAdapterSpec,
    target_kinds: list[str],
    registered: dict[str, type[Probe]],
    rank_scale: float,
) -> tuple[dict[str, dict[str, float]], dict[str, list[float]], list[dict[str, str]]]:
    """Run the full kind × seed calibration matrix at one rank multiplier.

    Returns ``(per_kind_stats, per_kind_samples, skipped)``.
    """
    per_kind_stats: dict[str, dict[str, float]] = {}
    per_kind_samples: dict[str, list[float]] = {}
    skipped: list[dict[str, str]] = []

    for kind in target_kinds:
        probe_cls = registered[kind]
        try:
            cal_spec = probe_cls.calibrate_spec(ctx)
        except Exception as exc:  # noqa: BLE001 — defensive
            skipped.append({"kind": kind, "reason": f"calibrate_spec raised: {exc}"})
            continue
        if cal_spec is None:
            skipped.append(
                {"kind": kind, "reason": "probe opted out (calibrate_spec returned None)"}
            )
            continue

        probe = probe_cls()
        raws: list[float] = []
        errors: list[str] = []
        for run_idx in range(spec.runs):
            seed = spec.seed_base + run_idx
            proxy = NullCalibrationBackendProxy(
                ctx.backend,  # type: ignore[arg-type]
                seed=seed,
                init_scale=spec.init_scale,
                rank_scale=rank_scale,
            )
            cal_ctx = RunContext(
                backend=proxy,
                seed=seed,
                top_k=ctx.top_k,
                sections=ctx.sections,
                doc_text=ctx.doc_text,
                null_stats={},  # calibration uses fixed thresholds — no recursion
                downstream_kinds=(),
            )
            try:
                cal_result = probe.run(cal_spec, cal_ctx)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"seed={seed}: {type(exc).__name__}: {exc}")
                continue
            raw = cal_result.raw
            if raw is not None and math.isfinite(raw):
                raws.append(float(raw))
            elif cal_result.verdict == Verdict.ERROR:
                errors.append(f"seed={seed}: probe ERROR — {cal_result.message}")

        if raws:
            mean = statistics.fmean(raws)
            std = statistics.pstdev(raws) if len(raws) > 1 else 0.0
            per_kind_stats[kind] = {
                "mean": mean,
                # C9: clamp the std floor so the downstream z-score
                # path doesn't blow up when every seed produces
                # identical raws.
                "std": max(std, 1e-6),
                "n": float(len(raws)),
            }
            per_kind_samples[kind] = raws
        else:
            reason = "no finite raws across all seeds"
            if errors:
                reason += f" ({errors[0]})"
            skipped.append({"kind": kind, "reason": reason})

    return per_kind_stats, per_kind_samples, skipped


def _pass_from_cache(spec: NullAdapterSpec, cached: dict[str, Any]) -> ProbeResult:
    """Rebuild a PASS result from a cache-loaded evidence dict."""
    stats_by_rank: dict[str, dict[str, dict[str, float]]] = dict(
        cached.get("null_stats_by_rank") or {}
    )
    # Prefer the explicit 1.0x group; fall back to the legacy ``null_stats``.
    primary_stats = stats_by_rank.get(_rank_key(1.0), cached.get("null_stats", {}))
    evidence: dict[str, Any] = dict(cached)
    evidence["null_stats"] = primary_stats
    evidence["null_stats_by_rank"] = stats_by_rank
    evidence.setdefault("skipped_kinds", [])
    evidence.setdefault("calibrated_kinds", list(primary_stats.keys()))
    evidence["weight"] = spec.weight
    evidence["from_cache"] = True
    n_kinds = len(primary_stats)
    n_ranks = len(stats_by_rank)
    message = (
        f"null calibration: {n_kinds} kinds (loaded from cache)"
        if n_ranks <= 1
        else f"null calibration: {n_kinds} kinds × {n_ranks} ranks (loaded from cache)"
    )
    return safe_finalize(
        name=spec.name,
        kind=spec.kind,
        verdict=Verdict.PASS,
        score=1.0,
        evidence=evidence,
        message=message,
    )


def _backend_identity(backend: Any) -> str | None:
    """Ask the backend for a stable cache identity string, if it has one.

    Duck-typed: backends that can't uniquely identify themselves (the
    dummy backend in tests, for example) simply don't provide this
    method, and caching is skipped for them.
    """
    fn = getattr(backend, "cache_identity", None)
    if not callable(fn):
        return None
    try:
        value = fn()
    except Exception:  # noqa: BLE001 — cache is best-effort
        return None
    return str(value) if value else None


def get_null_stats(ctx: RunContext, probe_kind: str) -> Mapping[str, float] | None:
    """Look up null-adapter stats for ``probe_kind`` in the run context.

    Returns ``{"mean": …, "std": …, "n": …}`` when calibration ran for
    this kind, else ``None``. Probes treat ``None`` as "fall back to
    the fixed threshold from your spec" and surface ``(no calibration)``
    in the report.
    """
    return ctx.null_stats.get(probe_kind)


def get_null_stats_by_rank(
    ctx: RunContext, probe_kind: str
) -> Mapping[str, Mapping[str, float]] | None:
    """Look up per-rank null-adapter stats for ``probe_kind``.

    Returns ``{rank_key: {"mean": …, "std": …, "n": …}}`` across every
    rank multiplier the ``null_adapter`` probe calibrated. ``None`` when
    no multi-rank calibration ran (pre-S10 behavior, or S02's single-
    rank default).
    """
    by_rank = ctx.null_stats_by_rank
    if not by_rank:
        return None
    out: dict[str, Mapping[str, float]] = {}
    for rkey, kind_map in by_rank.items():
        if probe_kind in kind_map:
            out[rkey] = kind_map[probe_kind]
    return out or None
