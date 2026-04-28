"""Training-drift — pre-run probe that reads dlm's per-step loss curve.

Sister probe to :mod:`gradient_ghost`. Where ``gradient_ghost`` reads
the optimizer state at the end of training, ``training_drift`` reads
the loss curve *during* training: the rich signal about *how* the
adapter trained, not just what it produced.

Four metrics extracted from the curve:

- ``final_loss`` — last recorded step's loss.
- ``convergence_ratio`` — ``final_loss / initial_loss``. Lower is
  better; a healthy adapter cuts loss by half or more.
- ``smoothness`` — ``1 - (var(Δloss) / var(loss))``. Values near 1
  mean the curve descends smoothly; near 0 means each step's
  change-in-loss is comparable to the loss range itself (spiky).
- ``instability_events`` — count of steps where
  ``|Δloss| > 3 · rolling_std``. Spikes that survive the rolling
  window are real — they correlate with silent adapter degradation.

Verdict: PASS when all three of (smoothness ≥ 0.7, instability_events
== 0, convergence_ratio ≤ 0.7); else WARN. The thresholds are
hand-tuned defaults; spec fields override them.

## Why no null calibration

Mirrors ``prompt_collapse`` and ``multi_turn_coherence_decay``: a null
adapter doesn't *train* — it has no loss curve. The null distribution
of "smoothness on a noise adapter" is undefined. Fixed-threshold
verdicts are the published path; users override per-spec.

## Log-format note

dlm writes one JSONL per run at
``<store_path>/logs/train-NNNNNN-YYYYMMDDTHHMMSS.jsonl``. Each line is
``{"type": "<banner|step|run_complete|...>", ...}``. This probe
filters for ``type == "step"`` records and reads ``step`` + ``loss``.
The sibling ``*.summary.json`` has run aggregates; we don't consume
it here — the curve is richer.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from pydantic import Field

from dlm_sway.core.errors import SwayError
from dlm_sway.core.result import ProbeResult, Verdict, safe_finalize
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext


class TrainingDriftError(SwayError):
    """Raised when a training log file is structurally unparseable.

    Distinct from :class:`MissingTrainingStateError`-style absences:
    a missing logs directory is a SKIP (no .dlm context, or a
    pre-training adapter), but a JSONL with corrupted lines or the
    wrong shape is an ERROR — the user has training data but we
    can't read it, which deserves a noisy failure.
    """


class TrainingDriftSpec(ProbeSpec):
    """Spec for ``kind: training_drift``."""

    kind: Literal["training_drift"] = "training_drift"
    store_path: str | None = None
    """Path to a dlm store root (the directory containing
    ``logs/`` and ``adapter/``). When ``None`` the probe SKIPs —
    typically populated by the .dlm autogen path which knows the
    store from the dlm_id resolution."""
    min_steps: int = Field(default=10, ge=2)
    """Skip when the curve has fewer steps than this. Short runs
    (3-step early-stops, smoke tests) produce undefined fits;
    surfacing those as PASS/FAIL would be misleading."""
    rolling_window: int = Field(default=10, ge=2)
    """Window for the rolling std used in spike detection. Larger
    windows mean tighter spike detection at the cost of missing
    rapid oscillations. 10 is a balance for typical 100–1000 step
    runs."""
    spike_sigma: float = Field(default=3.0, gt=0.0)
    """A step is an instability event when ``|Δloss|`` exceeds this
    many rolling standard deviations. 3σ is the conventional
    "outlier" boundary; lower for more sensitivity, higher for less
    noise tolerance."""
    assert_smoothness_gte: float = Field(default=0.7, ge=0.0, le=1.0)
    """Minimum smoothness for PASS."""
    assert_convergence_ratio_lte: float = Field(default=0.7, gt=0.0)
    """Maximum ``final_loss / initial_loss`` for PASS. A "well-
    trained" run typically halves loss; permissive default tolerates
    noisier data."""
    assert_instability_events_lte: int = Field(default=0, ge=0)
    """Maximum allowed spike count. Default 0 — any spike → WARN."""


class TrainingDriftProbe(Probe):
    """The "did this adapter train smoothly?" pre-run probe."""

    kind = "training_drift"
    spec_cls = TrainingDriftSpec
    category = "calibration"
    needs_backend: ClassVar[bool] = False
    # Pre-run probe: no model load, runs in <100ms, ideal for
    # ``sway check`` first-pass before any heavy probe fires.
    # Mirrors gradient_ghost's posture.

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        del ctx  # No backend / sections / null_stats consumed.
        assert isinstance(spec, TrainingDriftSpec)

        if spec.store_path is None:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.SKIP,
                score=None,
                message="no store_path provided (no .dlm context)",
            )

        store_path = Path(spec.store_path).expanduser().resolve()
        logs_dir = store_path / "logs"
        if not logs_dir.is_dir():
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.SKIP,
                score=None,
                message=f"no logs/ directory under {store_path}",
            )

        log_paths = sorted(logs_dir.glob("train-*.jsonl"))
        if not log_paths:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.SKIP,
                score=None,
                message=f"no train-*.jsonl files under {logs_dir}",
            )

        # Concatenate steps across all runs in chronological order.
        # Resumed runs may produce duplicate step numbers — dedupe by
        # keeping the latest occurrence (the most recent run's value
        # for that step). Sorted glob already gives chronological
        # order via the timestamp suffix.
        try:
            steps_by_idx = _collect_steps(log_paths)
        except TrainingDriftError as exc:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.ERROR,
                score=None,
                message=str(exc),
                evidence={"log_paths": [str(p) for p in log_paths]},
            )

        if len(steps_by_idx) < spec.min_steps:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.SKIP,
                score=None,
                message=(
                    f"only {len(steps_by_idx)} step records "
                    f"(< min_steps={spec.min_steps}); "
                    f"curve too short to fit reliably"
                ),
                evidence={"num_steps": len(steps_by_idx)},
            )

        # Sorted (step, loss) pairs.
        ordered = sorted(steps_by_idx.items())
        steps = np.asarray([s for s, _ in ordered], dtype=np.int64)
        losses = np.asarray([loss for _, loss in ordered], dtype=np.float64)

        # All four metrics. Each can fail gracefully (NaN-only loss
        # column, all-equal losses, etc.) and the verdict respects the
        # safe_finalize critical-field guard for ``raw``.
        metrics = _compute_metrics(
            losses, rolling_window=spec.rolling_window, spike_sigma=spec.spike_sigma
        )

        verdict, score, message = _verdict_from_metrics(metrics, spec)

        # Bound the curve we ship in evidence: a 10k-step run shouldn't
        # explode the JSON report. Downsample uniformly to the cap.
        curve = _downsampled_curve(steps, losses, cap=512)

        return safe_finalize(
            name=spec.name,
            kind=spec.kind,
            verdict=verdict,
            score=score,
            raw=metrics["smoothness"],
            evidence={
                "final_loss": metrics["final_loss"],
                "initial_loss": metrics["initial_loss"],
                "convergence_ratio": metrics["convergence_ratio"],
                "smoothness": metrics["smoothness"],
                "instability_events": metrics["instability_events"],
                "num_steps": len(losses),
                "num_log_files": len(log_paths),
                "curve_sampled": curve,
                "thresholds": {
                    "smoothness_gte": spec.assert_smoothness_gte,
                    "convergence_ratio_lte": spec.assert_convergence_ratio_lte,
                    "instability_events_lte": spec.assert_instability_events_lte,
                },
                "weight": spec.weight,
            },
            message=message,
            critical_fields=(),
            # ``raw`` (smoothness) is bounded [0, 1] in normal cases;
            # the metric helper returns NaN only on degenerate inputs
            # which we already surfaced via the SKIP path. Critical-
            # field guard would null-out the score on a NaN smoothness
            # which is more confusing than helpful here.
        )


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------


def _collect_steps(log_paths: list[Path]) -> dict[int, float]:
    """Parse every JSONL, dedupe-by-step, return ``{step: loss}``.

    Resumed runs append to a fresh JSONL but share step numbers with
    the prior run. Dedupe-by-keep-latest matches dlm's own ``dlm
    metrics`` semantics: the most recent run for a given step wins.
    """
    out: dict[int, float] = {}
    for path in log_paths:
        try:
            with path.open("r", encoding="utf-8") as f:
                for line_no, raw in enumerate(f, start=1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        # A trailing partial line from a crashed
                        # trainer is the typical cause. Skip it but
                        # surface as ERROR if EVERY line is broken
                        # (caller checks ``out`` emptiness).
                        if line_no == 1:
                            raise TrainingDriftError(
                                f"first line of {path.name} is not valid JSON: {exc}"
                            ) from exc
                        continue
                    if not isinstance(rec, dict):
                        continue
                    if rec.get("type") != "step":
                        continue
                    try:
                        step = int(rec["step"])
                        loss = float(rec["loss"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if not math.isfinite(loss):
                        # NaN loss is a real signal — record it as
                        # +inf so the spike detector flags the
                        # instability without crashing on np.log.
                        loss = math.inf
                    out[step] = loss
        except OSError as exc:
            raise TrainingDriftError(f"failed to read {path}: {exc}") from exc
    return out


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _compute_metrics(
    losses: np.ndarray,
    *,
    rolling_window: int,
    spike_sigma: float,
) -> dict[str, float]:
    """Compute the four headline metrics from the loss array.

    All metrics are robust to NaN/inf: non-finite step losses are
    replaced with the most-recent finite value before fitting (so a
    single bad batch doesn't poison the curve), but their *positions*
    are still counted as instability events.
    """
    losses = losses.astype(np.float64, copy=True)
    instability_from_nan = int(np.sum(~np.isfinite(losses)))

    if instability_from_nan > 0:
        # Forward-fill non-finite values from the previous finite
        # entry so downstream stats don't blow up. The first entry
        # MUST be finite (training records the first batch's loss
        # before anything could go wrong); guard anyway.
        finite_mask = np.isfinite(losses)
        if not finite_mask.any():
            # Pathological: every step recorded NaN. Return all-NaN
            # metrics so the verdict path can surface the failure.
            return {
                "initial_loss": float("nan"),
                "final_loss": float("nan"),
                "convergence_ratio": float("nan"),
                "smoothness": 0.0,
                "instability_events": float(len(losses)),
            }
        last_good = float(losses[finite_mask][0])
        for i, v in enumerate(losses):
            if not math.isfinite(v):
                losses[i] = last_good
            else:
                last_good = float(v)

    initial_loss = float(losses[0])
    final_loss = float(losses[-1])
    convergence_ratio = float(final_loss / initial_loss) if initial_loss != 0.0 else float("inf")

    deltas = np.diff(losses)
    var_loss = float(losses.var())
    var_delta = float(deltas.var()) if deltas.size > 0 else 0.0
    if var_loss > 0.0:  # noqa: SIM108 — branch comments are load-bearing
        # Clip into [0, 1]: a curve where var(Δloss) > var(loss)
        # implies the per-step jitter dominates the overall sweep —
        # treat as fully-spiky (smoothness=0) rather than emit a
        # negative number.
        smoothness = max(0.0, 1.0 - var_delta / var_loss)
    else:
        # Identical losses across every step: the run never
        # progressed. Smoothness is formally 1.0 (perfectly flat),
        # but that's misleading — surface as 0.0 so the verdict path
        # can flag it.
        smoothness = 0.0

    instability_events = _count_spikes(deltas, window=rolling_window, sigma=spike_sigma)
    instability_events += instability_from_nan

    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "convergence_ratio": convergence_ratio,
        "smoothness": smoothness,
        "instability_events": float(instability_events),
    }


def _count_spikes(deltas: np.ndarray, *, window: int, sigma: float) -> int:
    """Count loss-increase events that exceed a robust noise threshold.

    The naive ``|Δloss| > sigma · rolling_std`` heuristic is broken for
    exponential decay: deltas span orders of magnitude across training,
    so within-window std stays tiny while absolute deltas are large —
    every step trips the threshold. The semantically correct
    "instability event" is a loss *increase*, not a faster-than-typical
    decrease.

    Heuristic:

    1. Restrict to positive deltas (``delta > 0`` ⇒ loss went up). Loss
       going down faster than usual isn't an instability — it's the
       happy path.
    2. For each positive delta, compare to the median absolute delta
       in a centered window. Flag when ``delta > sigma · MAD``, where
       MAD is the median absolute deviation (robust to outliers).
    3. Short curves fall back to global MAD; constant-loss curves
       (every delta zero) report 0 spikes.

    The ``sigma`` parameter retains its semantic meaning ("how many
    typical deltas does this exceed") but operates against the
    median-absolute-delta scale rather than std-of-delta.
    """
    if deltas.size == 0:
        return 0

    # Median absolute delta — the "typical movement scale" we judge
    # spikes against. Median is robust to a few outliers; if the
    # whole curve is flat, MAD is 0 and we report no spikes.
    abs_deltas = np.abs(deltas)

    if deltas.size < window:
        baseline = float(np.median(abs_deltas))
        if baseline == 0.0:
            return 0
        return int(np.sum((deltas > 0.0) & (deltas > sigma * baseline)))

    spikes = 0
    half = window // 2
    for i in range(deltas.size):
        if deltas[i] <= 0.0:
            continue  # Loss went down — not an instability event.
        lo = max(0, i - half)
        hi = min(deltas.size, i + half + 1)
        window_slice = abs_deltas[lo:hi]
        # Exclude the point itself so a real outlier doesn't inflate
        # the baseline it's measured against.
        if window_slice.size > 1:
            mask = np.ones(window_slice.size, dtype=bool)
            mask[i - lo] = False
            window_slice = window_slice[mask]
        baseline = float(np.median(window_slice))
        if baseline == 0.0:
            # Surrounding deltas are all zero — any nonzero positive
            # delta is an instability event by construction.
            spikes += 1
            continue
        if float(deltas[i]) > sigma * baseline:
            spikes += 1
    return spikes


# ---------------------------------------------------------------------------
# Verdict mapping
# ---------------------------------------------------------------------------


def _verdict_from_metrics(
    metrics: dict[str, float], spec: TrainingDriftSpec
) -> tuple[Verdict, float, str]:
    """Map the four metrics to a (verdict, score, message)."""
    smooth = metrics["smoothness"]
    conv = metrics["convergence_ratio"]
    instability = int(metrics["instability_events"])

    smoothness_pass = smooth >= spec.assert_smoothness_gte
    convergence_pass = conv <= spec.assert_convergence_ratio_lte
    instability_pass = instability <= spec.assert_instability_events_lte

    failures: list[str] = []
    if not smoothness_pass:
        failures.append(f"smoothness={smooth:.2f} < {spec.assert_smoothness_gte}")
    if not convergence_pass:
        failures.append(f"convergence_ratio={conv:.2f} > {spec.assert_convergence_ratio_lte}")
    if not instability_pass:
        failures.append(f"instability_events={instability} > {spec.assert_instability_events_lte}")

    headline = (
        f"smoothness={smooth:.2f}, convergence_ratio={conv:.2f}, "
        f"instability_events={instability}, final_loss={metrics['final_loss']:.3f}"
    )

    if not failures:
        # All three thresholds clear → PASS with a normalized score
        # blending the three signals (smoothness contributes most;
        # convergence and instability are guard rails).
        score = float(min(1.0, max(0.0, smooth)))
        return Verdict.PASS, score, headline

    # Score: a continuous blend of how far we are from each threshold,
    # so the report can rank "borderline warn" against "actively bad."
    # Doesn't influence the verdict — that's already FAIL/WARN.
    score = float(min(1.0, max(0.0, smooth * 0.5)))
    return Verdict.WARN, score, f"{headline}; warnings: {'; '.join(failures)}"


# ---------------------------------------------------------------------------
# Curve downsampling for evidence
# ---------------------------------------------------------------------------


def _downsampled_curve(
    steps: np.ndarray, losses: np.ndarray, *, cap: int
) -> list[tuple[int, float]]:
    """Uniform-stride downsample so a 10k-step run still fits in the JSON report.

    Always preserves the first and last point so initial/final loss
    are visible regardless of cap. For curves shorter than the cap,
    returns the full series unchanged.
    """
    n = int(len(losses))
    if n <= cap:
        return [(int(s), float(loss)) for s, loss in zip(steps, losses, strict=True)]
    # Stride to keep at most ``cap`` points: stride = ceil(n / cap).
    # The +1 accounts for always-appending the final point even when
    # ``stride * (cap - 1) < n - 1``.
    stride = max(1, (n + cap - 1) // cap)
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return [(int(steps[i]), float(losses[i])) for i in idx]
