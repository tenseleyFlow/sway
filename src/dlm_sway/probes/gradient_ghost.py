"""P3 / S25 — GradientGhostProbe (pre-run, cross-repo, no model load).

Reads the optimizer-state snapshot dlm writes alongside every adapter
version (``training_state.pt``) and answers one practical question
**before any forward pass fires**: was this adapter trained long enough
to converge?

Loads in ~50 ms (a 1.5B-param adapter's optimizer state is ~50 MB
pickle). The full adherence suite costs 30+ s; running this first as
a pre-flight check lets ``sway check`` short-circuit on obviously-
broken adapters without paying the model-load tax.

## Signal ladder (most → least decisive)

1. **``global_step < min_steps_threshold``** — primary signal.
   Catches the 90%-case "user did ``--max-steps 5`` for a smoke test
   and forgot to retrain." No analysis needed; verdict: FAIL.
2. **NaN / zero ``exp_avg_sq``** — strong secondary signal.
   Adam's second-moment estimate didn't accumulate any useful
   variance, meaning gradients didn't propagate meaningfully. Often
   co-occurs with case 1 but worth flagging independently for
   trainings that *did* run many steps with broken loss.
3. **Per-layer ratio: layer mean vs global mean of
   ``exp_avg_sq``** — heuristic. Layers whose second-moment is
   dramatically above the global mean still see large gradient
   variance — they haven't converged. Verdict: WARN when a
   significant fraction of layers cross the threshold.

## Why no null calibration

Other probes z-score against a null-adapter baseline ("how much
signal does random noise produce?"). Gradient state has no
meaningful null — there's no equivalent to a "random optimizer
snapshot" that the LoRA's noise floor would settle into. Verdict
thresholds are explicit heuristics; document as "tune from user
feedback" rather than fake calibration math.

## Inputs

The spec carries ``adapter_path`` (the dir containing
``training_state.pt`` + ``adapter_model.safetensors``). Typically
populated by the dlm autogen bridge from
``DlmHandle.adapter_path``.

## Verdict thresholds (all configurable in spec)

- ``min_steps_threshold = 50`` — below this is severely undertrained.
- ``undertrained_layer_ratio = 2.0`` — a layer's mean ``exp_avg_sq``
  must be > 2× the **minimum** layer's mean to count as "still has
  high gradient variance." We compare against the min (not the
  global mean) because mean rises with outliers — under a global-
  mean baseline the per-layer ratio asymptotically caps at
  ``N/(N-K)`` and can never exceed ``ratio`` for K layers without K
  also rising. Min-baseline gives a stable "K layers are anomalously
  high vs the calmest layer" signal regardless of how many layers
  spike.
- ``layer_failure_frac = 0.3`` — WARN if more than 30% of layers
  cross the per-layer threshold.
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field

from dlm_sway.core.errors import MissingTrainingStateError, SwayError
from dlm_sway.core.result import ProbeResult, Verdict, safe_finalize
from dlm_sway.probes._param_id_mapping import ParamMappingError, map_param_ids_to_layers
from dlm_sway.probes._training_state import TrainingStateError, load_training_state
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext


class GradientGhostSpec(ProbeSpec):
    """Spec for ``kind: gradient_ghost``."""

    kind: Literal["gradient_ghost"] = "gradient_ghost"
    adapter_path: str
    """Path to a dlm adapter version directory (must contain
    ``training_state.pt``). Resolved relative to the spec file's
    cwd via the same convention sway already uses for
    ``models.ft.adapter``."""
    min_steps_threshold: int = Field(default=50, ge=1)
    """``global_step`` below this → FAIL (severely undertrained)."""
    undertrained_layer_ratio: float = Field(default=2.0, gt=1.0)
    """A layer counts as 'high gradient variance' when its mean
    ``exp_avg_sq`` exceeds ``ratio * min_layer_mean``. The min-
    baseline (rather than global mean) is robust to outliers — see
    the module docstring for the asymptotic-cap reasoning."""
    layer_failure_frac: float = Field(default=0.3, ge=0.0, le=1.0)
    """WARN when more than this fraction of layers cross the
    ``undertrained_layer_ratio`` threshold."""


class GradientGhostProbe(Probe):
    """Pre-run training-health probe."""

    kind = "gradient_ghost"
    spec_cls = GradientGhostSpec
    category = "calibration"
    needs_backend: ClassVar[bool] = False

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        del ctx  # No backend / sections / null_stats needed.
        assert isinstance(spec, GradientGhostSpec)
        adapter_dir = Path(spec.adapter_path).expanduser()

        # === 1) Load + validate training state ===
        try:
            snap = load_training_state(adapter_dir)
        except MissingTrainingStateError as exc:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.SKIP,
                score=None,
                message=str(exc),
            )
        except (TrainingStateError, SwayError) as exc:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.ERROR,
                score=None,
                message=str(exc),
            )

        # === 2) Layer grouping (best-effort; degrades to no-grouping) ===
        try:
            grouping = map_param_ids_to_layers(adapter_dir, num_params=len(snap.per_param))
            layer_count = grouping.num_layers
            params_per_layer = grouping.params_per_layer
        except ParamMappingError:
            grouping = None
            layer_count = 0
            params_per_layer = 0

        # === 3) Primary signal: global_step floor ===
        if snap.global_step < spec.min_steps_threshold:
            return safe_finalize(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.FAIL,
                score=0.0,
                raw=float(snap.global_step),
                z_score=None,
                evidence={
                    "global_step": snap.global_step,
                    "min_steps_threshold": spec.min_steps_threshold,
                    "epoch": snap.epoch,
                    "num_params": len(snap.per_param),
                    "num_layers": layer_count,
                    "best_val_loss": snap.best_val_loss
                    if math.isfinite(snap.best_val_loss)
                    else None,
                    "primary_signal": "global_step_below_threshold",
                },
                message=(
                    f"severely undertrained: global_step={snap.global_step} "
                    f"< threshold {spec.min_steps_threshold}. Probe scores on "
                    f"this adapter will be unreliable; consider retraining."
                ),
            )

        # === 4) NaN / zero exp_avg_sq detection ===
        finite_means = [
            ps.exp_avg_sq_mean for ps in snap.per_param if math.isfinite(ps.exp_avg_sq_mean)
        ]
        nan_or_inf_count = len(snap.per_param) - len(finite_means)
        if not finite_means:
            return safe_finalize(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.FAIL,
                score=0.0,
                raw=0.0,
                z_score=None,
                evidence={
                    "global_step": snap.global_step,
                    "num_params": len(snap.per_param),
                    "num_nonfinite_exp_avg_sq": nan_or_inf_count,
                    "primary_signal": "all_optimizer_state_nan",
                },
                message=(
                    "every per-param exp_avg_sq is NaN or non-finite — "
                    "training didn't propagate gradients meaningfully."
                ),
            )

        # === 5) Per-layer secondary signal ===
        global_mean = statistics.fmean(finite_means)
        per_layer_means: dict[int, float] = {}
        per_layer_undertrained: list[int] = []
        baseline_min: float = 0.0

        if grouping is not None and global_mean > 0.0:
            # Group finite per-param means by layer index.
            buckets: dict[int, list[float]] = {idx: [] for idx in grouping.layer_indices}
            for ps in snap.per_param:
                if not math.isfinite(ps.exp_avg_sq_mean):
                    continue
                layer = grouping.layer_of.get(ps.param_id)
                if layer is not None:
                    buckets[layer].append(ps.exp_avg_sq_mean)
            for layer_idx, vals in buckets.items():
                if not vals:
                    continue
                per_layer_means[layer_idx] = statistics.fmean(vals)
            # Min-baseline ratio (see module docstring on why we use
            # min instead of global mean — global-mean ratio is
            # asymptotically capped and can't catch the case where K
            # layers all spike together).
            if per_layer_means:
                baseline_min = min(per_layer_means.values())
                if baseline_min > 0.0:
                    for layer_idx, mean in per_layer_means.items():
                        if mean > spec.undertrained_layer_ratio * baseline_min:
                            per_layer_undertrained.append(layer_idx)

        frac_undertrained = len(per_layer_undertrained) / layer_count if layer_count > 0 else 0.0

        # Top-3 worst layers (highest ratio vs baseline_min) — useful
        # evidence even when no layer crosses the threshold.
        ranked_layers = sorted(per_layer_means.items(), key=lambda kv: -kv[1])[:3]
        worst_layers = [
            {"layer": idx, "ratio": (mean / baseline_min) if baseline_min > 0 else None}
            for idx, mean in ranked_layers
        ]

        # === 6) Final verdict ===
        if frac_undertrained > spec.layer_failure_frac:
            verdict = Verdict.FAIL
            message = (
                f"{frac_undertrained:.0%} of layers ({len(per_layer_undertrained)}/"
                f"{layer_count}) still show high gradient variance — training-loss "
                f"curve likely hasn't flattened. global_step={snap.global_step}."
            )
            score = 0.3  # Bottom of "partial" band.
        elif per_layer_undertrained:
            verdict = Verdict.WARN
            message = (
                f"{frac_undertrained:.0%} of layers ({len(per_layer_undertrained)}/"
                f"{layer_count}) above {spec.undertrained_layer_ratio:.1f}× the global "
                f"exp_avg_sq mean — adapter may be partially undertrained but other "
                f"probes can still produce signal. global_step={snap.global_step}."
            )
            score = 0.7
        else:
            verdict = Verdict.PASS
            message = (
                f"global_step={snap.global_step}, no layer above "
                f"{spec.undertrained_layer_ratio:.1f}× global exp_avg_sq mean — "
                f"training looks converged."
            )
            score = 0.9

        return safe_finalize(
            name=spec.name,
            kind=spec.kind,
            verdict=verdict,
            score=score,
            raw=frac_undertrained,
            z_score=None,
            evidence={
                "global_step": snap.global_step,
                "epoch": snap.epoch,
                "num_params": len(snap.per_param),
                "num_layers": layer_count,
                "params_per_layer": params_per_layer,
                "global_mean_exp_avg_sq": global_mean,
                "frac_layers_undertrained": frac_undertrained,
                "num_layers_undertrained": len(per_layer_undertrained),
                "worst_layers": worst_layers,
                "num_nonfinite_exp_avg_sq": nan_or_inf_count,
                "best_val_loss": snap.best_val_loss if math.isfinite(snap.best_val_loss) else None,
                "use_qlora": snap.use_qlora,
            },
            message=message,
        )
