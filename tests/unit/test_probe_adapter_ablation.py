"""Tests for :mod:`dlm_sway.probes.adapter_ablation`.

Uses the dummy backend's lam-interpolation implementation to exercise
the full probe path without loading a real model.
"""

from __future__ import annotations

import numpy as np
import pytest

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.result import Verdict
from dlm_sway.core.scoring import ScalableDifferentialBackend, TokenDist
from dlm_sway.probes.adapter_ablation import (
    _overshoot,
    _r_squared,
    _saturation_lambda,
)
from dlm_sway.probes.base import RunContext, build_probe


class TestShapeMetrics:
    def test_r_squared_perfect_linear(self) -> None:
        x = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
        y = 2 * x + 0.1
        assert _r_squared(x, y) > 0.99

    def test_r_squared_zero_slope_defined(self) -> None:
        x = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
        y = np.zeros_like(x)
        # Flat y → ss_tot = 0 → defined as 1.0 (perfect fit).
        assert _r_squared(x, y) == 1.0

    def test_saturation_lambda_expected(self) -> None:
        lambdas = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
        divs = np.asarray([0.0, 0.5, 0.8, 0.95, 1.0], dtype=np.float64)
        sat, reason = _saturation_lambda(lambdas, divs)
        assert sat == 0.75  # 0.95 / 1.0 = 0.95 ≥ 0.9
        assert reason == "found"

    def test_overshoot_recovered(self) -> None:
        lambdas = np.asarray([0.0, 0.5, 1.0, 1.25], dtype=np.float64)
        divs = np.asarray([0.0, 0.5, 1.0, 1.15], dtype=np.float64)
        assert _overshoot(lambdas, divs) == 1.15

    def test_saturation_flat_curve(self) -> None:
        """Adapter that produces no signal — every divergence is zero."""
        lambdas = np.asarray([0.0, 0.5, 1.0, 1.25], dtype=np.float64)
        divs = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        sat, reason = _saturation_lambda(lambdas, divs)
        assert sat is None
        assert reason == "flat_curve"

    def test_saturation_overshoot_peak_above_one(self) -> None:
        """B3 fix: curve peaks at λ=1.25 (0.95 reaches 90% of max=1.0).
        Saturation now picks the smallest λ where divs ≥ 0.9 × 1.0 = 0.9 —
        namely λ=0.5 with div=0.95. Pre-fix this returned ``None`` because
        the search was bounded at λ ≤ 1.0 with `div(λ=1)` as reference."""
        lambdas = np.asarray([0.0, 0.5, 1.0, 1.25], dtype=np.float64)
        divs = np.asarray([0.0, 0.95, 0.7, 1.0], dtype=np.float64)
        sat, reason = _saturation_lambda(lambdas, divs)
        assert sat == 0.5
        # Curve was monotonic up to and including the saturation point;
        # the dip happened *after* it, which the per-saturation-point
        # monotonicity check accepts.
        assert reason == "found"

    def test_saturation_non_monotonic_before_saturation(self) -> None:
        """A curve that zigzags up to the saturation point gets a WARN."""
        lambdas = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
        # 0 → 0.6 → 0.4 (dip) → 0.95 (90% of max=1.0)
        divs = np.asarray([0.0, 0.6, 0.4, 0.95, 1.0], dtype=np.float64)
        sat, reason = _saturation_lambda(lambdas, divs)
        assert sat == 0.75
        assert reason == "non_monotonic"

    def test_saturation_below_floor_with_negative_max(self) -> None:
        """Pathological negative-only curve (shouldn't happen with JS but
        guards against numerical drift / future probe variants)."""
        lambdas = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
        divs = np.asarray([-0.5, -0.3, -0.1], dtype=np.float64)
        sat, reason = _saturation_lambda(lambdas, divs)
        assert sat is None
        assert reason == "flat_curve"  # max ≤ 0 → flat by definition

    def test_saturation_nan_max_treated_as_flat(self) -> None:
        lambdas = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
        divs = np.asarray([0.1, np.nan, 0.5], dtype=np.float64)
        sat, reason = _saturation_lambda(lambdas, divs)
        assert sat is None
        assert reason == "flat_curve"

    def test_saturation_monotonically_decreasing(self) -> None:
        """A curve that goes *down* with λ — adapter is anti-correlated
        with its own effect (the degenerate case where lam=1 is closer
        to base than lam=0)."""
        lambdas = np.asarray([0.0, 0.5, 1.0, 1.25], dtype=np.float64)
        divs = np.asarray([0.9, 0.6, 0.3, 0.1], dtype=np.float64)
        sat, reason = _saturation_lambda(lambdas, divs)
        # max is at λ=0 → smallest λ where divs ≥ 0.9*0.9=0.81 is λ=0 itself.
        assert sat == 0.0
        # Curve is strictly decreasing; the monotonic-non-decreasing check
        # on divs[:0+1] is trivially satisfied (length 1), so we classify
        # as "found" — the probe's overshoot / linearity checks pick up
        # the real pathology here.
        assert reason == "found"


class TestProbeVerdictPropagatesSaturationReason:
    """C8 + B3 test side: the probe's ``evidence["saturation_reason"]``
    reflects the helper's return value for each curve shape. We force
    specific curves by monkeypatching the probe's ``divergence()`` call
    so the tests are deterministic and cheap."""

    def _run_with_curve(self, divs_by_lambda: list[float]) -> dict:
        from dlm_sway.probes import adapter_ablation as ab_mod

        lambdas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
        assert len(lambdas) == len(divs_by_lambda)
        call_count = {"i": 0}

        def _fake_divergence(a, b, *, kind):  # noqa: ANN001 — test-local
            del a, b, kind
            # One prompt × len(lambdas) invocations; round-robin the curve.
            idx = call_count["i"] % len(lambdas)
            call_count["i"] += 1
            return divs_by_lambda[idx]

        backend = _diverging_backend()
        probe, spec = build_probe(
            {
                "name": "abl",
                "kind": "adapter_ablation",
                "prompts": ["q1"],
                "lambdas": lambdas,
                "assert_linearity_gte": 0.0,
                "assert_overshoot_gte": 0.0,
            }
        )
        ctx = RunContext(backend=backend)
        mp = pytest.MonkeyPatch()
        mp.setattr(ab_mod, "divergence", _fake_divergence)
        try:
            result = probe.run(spec, ctx)
        finally:
            mp.undo()
        return dict(result.evidence)

    def test_flat_curve_surfaces_flat_reason(self) -> None:
        ev = self._run_with_curve([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert ev["saturation_reason"] == "flat_curve"
        assert ev["saturation_lambda"] is None

    def test_healthy_monotonic_curve_found(self) -> None:
        ev = self._run_with_curve([0.0, 0.3, 0.6, 0.85, 0.95, 1.0])
        assert ev["saturation_reason"] == "found"
        assert ev["saturation_lambda"] is not None

    def test_overshoot_dip_still_found_via_max_reference(self) -> None:
        """B3 fix: curve peaks at λ=0.75, dips at λ=1.0, recovers at 1.25."""
        ev = self._run_with_curve([0.0, 0.3, 0.6, 0.95, 0.7, 1.0])
        # max=1.0 at λ=1.25 → 0.9*1.0 = 0.9 → smallest λ where div ≥ 0.9
        # is λ=0.75 (div=0.95). Monotonic through 0.75 → "found".
        assert ev["saturation_reason"] == "found"
        assert ev["saturation_lambda"] == 0.75

    def test_non_monotonic_before_saturation(self) -> None:
        ev = self._run_with_curve([0.0, 0.6, 0.4, 0.95, 1.0, 1.05])
        assert ev["saturation_reason"] == "non_monotonic"
        assert ev["saturation_lambda"] == 0.75


def _diverging_backend() -> DummyDifferentialBackend:
    """Backend where base ≠ ft at a few prompts; distributions interpolate
    smoothly under lam-blending in DummyDifferentialBackend.as_scaled_adapter."""
    base = DummyResponses(
        token_dists={
            "q1": TokenDist(
                token_ids=np.array([1, 2, 3], dtype=np.int64),
                logprobs=np.log(np.array([0.9, 0.05, 0.05], dtype=np.float32)),
                vocab_size=100,
            ),
            "q2": TokenDist(
                token_ids=np.array([5, 6], dtype=np.int64),
                logprobs=np.log(np.array([0.8, 0.2], dtype=np.float32)),
                vocab_size=100,
            ),
        }
    )
    ft = DummyResponses(
        token_dists={
            "q1": TokenDist(
                token_ids=np.array([1, 2, 3], dtype=np.int64),
                logprobs=np.log(np.array([0.2, 0.4, 0.4], dtype=np.float32)),
                vocab_size=100,
            ),
            "q2": TokenDist(
                token_ids=np.array([5, 6], dtype=np.int64),
                logprobs=np.log(np.array([0.3, 0.7], dtype=np.float32)),
                vocab_size=100,
            ),
        }
    )
    return DummyDifferentialBackend(base=base, ft=ft)


class TestProbe:
    def test_backend_implements_scalable_protocol(self) -> None:
        backend = _diverging_backend()
        assert isinstance(backend, ScalableDifferentialBackend)

    def test_probe_runs_and_emits_shape_metrics(self) -> None:
        probe, spec = build_probe(
            {
                "name": "abl",
                "kind": "adapter_ablation",
                "prompts": ["q1", "q2"],
                "lambdas": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
                # Very permissive to tolerate the log-space blend of a
                # tiny synthetic fixture.
                "assert_linearity_gte": 0.3,
                "assert_overshoot_gte": 1.0,
            }
        )
        ctx = RunContext(backend=_diverging_backend())
        result = probe.run(spec, ctx)
        assert result.verdict in (Verdict.PASS, Verdict.FAIL)
        assert "lambdas" in result.evidence
        assert "mean_divergence_per_lambda" in result.evidence
        assert len(result.evidence["mean_divergence_per_lambda"]) == 6
        # Divergence should increase as λ grows from 0 toward ft.
        divs = result.evidence["mean_divergence_per_lambda"]
        # λ=0 → 0 divergence from itself. λ>0 should be non-decreasing
        # for the bulk of the curve.
        assert divs[-2] >= divs[0]

    def test_skip_when_backend_not_scalable(self) -> None:
        class _NonScalable:
            def as_base(self):  # noqa: ANN202
                raise NotImplementedError

            def as_finetuned(self):  # noqa: ANN202
                raise NotImplementedError

        probe, spec = build_probe(
            {
                "name": "abl",
                "kind": "adapter_ablation",
                "prompts": ["q1"],
            }
        )
        ctx = RunContext(backend=_NonScalable())  # type: ignore[arg-type]
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.SKIP
        assert "ScalableDifferentialBackend" in result.message

    def test_error_on_empty_prompts(self) -> None:
        backend = _diverging_backend()
        probe, spec = build_probe({"name": "abl", "kind": "adapter_ablation", "prompts": []})
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.ERROR
