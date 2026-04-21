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
from typing import Any, Literal

from pydantic import Field

from dlm_sway.core.result import ProbeResult, Verdict, safe_finalize
from dlm_sway.core.scoring import NullCalibratedBackend
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

        per_kind_stats: dict[str, dict[str, float]] = {}
        per_kind_samples: dict[str, list[float]] = {}
        skipped_kinds: list[dict[str, str]] = []

        for kind in target_kinds:
            probe_cls = registered[kind]
            try:
                cal_spec = probe_cls.calibrate_spec(ctx)
            except Exception as exc:  # noqa: BLE001 — defensive
                skipped_kinds.append(
                    {"kind": kind, "reason": f"calibrate_spec raised: {exc}"}
                )
                continue
            if cal_spec is None:
                skipped_kinds.append(
                    {
                        "kind": kind,
                        "reason": "probe opted out (calibrate_spec returned None)",
                    }
                )
                continue

            probe = probe_cls()
            raws: list[float] = []
            errors: list[str] = []
            for run_idx in range(spec.runs):
                seed = spec.seed_base + run_idx
                proxy = NullCalibrationBackendProxy(
                    ctx.backend, seed=seed, init_scale=spec.init_scale
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
                    errors.append(
                        f"seed={seed}: probe ERROR — {cal_result.message}"
                    )

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
                skipped_kinds.append({"kind": kind, "reason": reason})

        evidence: dict[str, Any] = {
            "null_stats": per_kind_stats,
            "per_kind_raw_samples": per_kind_samples,
            "skipped_kinds": skipped_kinds,
            "calibrated_kinds": list(per_kind_stats.keys()),
            "runs": spec.runs,
            "init_scale": spec.init_scale,
            "seed_base": spec.seed_base,
            "weight": spec.weight,
        }

        message = (
            f"null calibration: {len(per_kind_stats)} kinds calibrated "
            f"over {spec.runs} seeds"
        )
        if skipped_kinds:
            message += f" ({len(skipped_kinds)} opted out)"

        return safe_finalize(
            name=spec.name,
            kind=spec.kind,
            verdict=Verdict.PASS,
            score=1.0,
            evidence=evidence,
            message=message,
        )


def get_null_stats(ctx: RunContext, probe_kind: str) -> dict[str, float] | None:
    """Look up null-adapter stats for ``probe_kind`` in the run context.

    Returns ``{"mean": …, "std": …, "n": …}`` when calibration ran for
    this kind, else ``None``. Probes treat ``None`` as "fall back to
    the fixed threshold from your spec" and surface ``(no calibration)``
    in the report.
    """
    return ctx.null_stats.get(probe_kind)
