"""N2 AdapterAblation — the sway signature primitive.

Scales the LoRA additive term by λ ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.25}
and measures the mean divergence from the base distribution at each
step. Fits a monotonic response curve; reports three shape metrics:

- **linearity**: R² of a linear fit on ``(λ, mean_div)``. High means
  the adapter's effect scales predictably; low means it's "all or
  nothing" (degenerate).
- **saturation_lambda**: the smallest λ at which divergence reaches
  90% of the λ=1 value. Too low (<0.3) means the adapter fires at
  partial strength — fragile. Too high (>1.0) means the adapter is
  under-trained.
- **overshoot**: divergence at λ=1.25 divided by λ=1.0. >1.05 is the
  healthy "pushing past 1 still moves the model" signal. An overshoot
  below 1.0 suggests collapse.

This is the single novel primitive that no generic eval harness
provides — sway's position next to the adapter math makes it possible.

Requires the backend to implement
:class:`~dlm_sway.core.scoring.ScalableDifferentialBackend`. Probes
SKIP gracefully on backends that don't.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from dlm_sway.core.result import ProbeResult, Verdict
from dlm_sway.core.scoring import ScalableDifferentialBackend
from dlm_sway.probes._divergence import Divergence, divergence
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext


class AdapterAblationSpec(ProbeSpec):
    kind: Literal["adapter_ablation"] = "adapter_ablation"
    prompts: list[str] = Field(default_factory=list)
    lambdas: list[float] = Field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
        min_length=3,
    )
    divergence: Divergence = "js"
    top_k: int | None = None
    assert_linearity_gte: float = 0.85
    assert_saturation_between: tuple[float, float] = (0.3, 1.05)
    assert_overshoot_gte: float = 1.02


class AdapterAblationProbe(Probe):
    kind = "adapter_ablation"
    spec_cls = AdapterAblationSpec
    category = "ablation"

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        assert isinstance(spec, AdapterAblationSpec)
        if not spec.prompts:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.ERROR,
                score=None,
                message="no prompts provided",
            )
        if not isinstance(ctx.backend, ScalableDifferentialBackend):
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.SKIP,
                score=None,
                message=(
                    "backend does not implement ScalableDifferentialBackend — "
                    "adapter ablation requires LoRA-scale access"
                ),
            )

        top_k = spec.top_k if spec.top_k is not None else ctx.top_k

        # Reference distribution at λ=0 (adapter scaled to zero → base).
        lam_zero = min(spec.lambdas)
        per_lambda: list[float] = []
        for lam in spec.lambdas:
            divs_for_lam: list[float] = []
            for prompt in spec.prompts:
                with ctx.backend.as_scaled_adapter(lam_zero) as ref:
                    ref_dist = ref.next_token_dist(prompt, top_k=top_k)
                with ctx.backend.as_scaled_adapter(lam) as scaled:
                    scaled_dist = scaled.next_token_dist(prompt, top_k=top_k)
                divs_for_lam.append(divergence(ref_dist, scaled_dist, kind=spec.divergence))
            per_lambda.append(float(np.mean(divs_for_lam)))

        lambdas_arr = np.asarray(spec.lambdas, dtype=np.float64)
        divs_arr = np.asarray(per_lambda, dtype=np.float64)

        linearity = _r_squared(lambdas_arr, divs_arr)
        saturation_lambda = _saturation_lambda(lambdas_arr, divs_arr)
        overshoot = _overshoot(lambdas_arr, divs_arr)

        # Pass when all three shape metrics land in their healthy bands.
        sat_lo, sat_hi = spec.assert_saturation_between
        ok_lin = linearity >= spec.assert_linearity_gte
        ok_sat = saturation_lambda is not None and sat_lo <= saturation_lambda <= sat_hi
        ok_over = overshoot >= spec.assert_overshoot_gte
        verdict = Verdict.PASS if (ok_lin and ok_sat and ok_over) else Verdict.FAIL

        lin_score = max(0.0, min(1.0, linearity / max(spec.assert_linearity_gte, 1e-6)))
        over_score = max(0.0, min(1.0, (overshoot - 1.0) / 0.2))
        sat_score = 1.0 if ok_sat else 0.3
        score = 0.4 * lin_score + 0.3 * sat_score + 0.3 * over_score

        return ProbeResult(
            name=spec.name,
            kind=spec.kind,
            verdict=verdict,
            score=score,
            raw=linearity,
            evidence={
                "lambdas": spec.lambdas,
                "mean_divergence_per_lambda": per_lambda,
                "linearity": linearity,
                "saturation_lambda": saturation_lambda,
                "overshoot": overshoot,
                "passed_linearity": ok_lin,
                "passed_saturation": ok_sat,
                "passed_overshoot": ok_over,
                "weight": spec.weight,
            },
            message=(
                f"R²={linearity:.2f}, sat_λ={saturation_lambda:.2f} "
                f"({'in' if ok_sat else 'out of'} band), overshoot={overshoot:.2f}"
                if saturation_lambda is not None
                else f"R²={linearity:.2f}, saturation undetected, overshoot={overshoot:.2f}"
            ),
        )


def _r_squared(x: np.ndarray, y: np.ndarray) -> float:
    """Coefficient of determination for a linear fit of ``y`` on ``x``."""
    if x.size < 2:
        return 0.0
    xm = float(x.mean())
    ym = float(y.mean())
    denom = float(((x - xm) ** 2).sum())
    if denom == 0.0:
        return 0.0
    slope = float(((x - xm) * (y - ym)).sum()) / denom
    intercept = ym - slope * xm
    y_pred = slope * x + intercept
    ss_res = float(((y - y_pred) ** 2).sum())
    ss_tot = float(((y - ym) ** 2).sum())
    if ss_tot == 0.0:
        return 1.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def _saturation_lambda(lambdas: np.ndarray, divs: np.ndarray) -> float | None:
    """Smallest λ ≤ 1.0 at which divergence reaches 90% of div(λ=1)."""
    # Locate the index of λ=1.0 (or the closest entry ≤ 1.0).
    candidates = np.where(np.isclose(lambdas, 1.0, atol=1e-6))[0]
    if candidates.size == 0:
        # Fall back to the largest λ ≤ 1.0.
        mask = lambdas <= 1.0
        if not mask.any():
            return None
        idx1 = int(np.argmax(lambdas * mask))
    else:
        idx1 = int(candidates[0])
    target = 0.9 * float(divs[idx1])
    if target <= 0:
        return None
    for lam, d in zip(lambdas[: idx1 + 1], divs[: idx1 + 1], strict=False):
        if d >= target:
            return float(lam)
    return None


def _overshoot(lambdas: np.ndarray, divs: np.ndarray) -> float:
    """``div(λ_max) / div(λ=1)``. Returns 1.0 if λ_max ≤ 1.0."""
    idx_max = int(np.argmax(lambdas))
    candidates = np.where(np.isclose(lambdas, 1.0, atol=1e-6))[0]
    if candidates.size == 0:
        return 1.0
    idx1 = int(candidates[0])
    if idx_max == idx1:
        return 1.0
    d1 = float(divs[idx1])
    dmax = float(divs[idx_max])
    if d1 <= 0:
        return 1.0
    return dmax / d1
