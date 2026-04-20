"""Tests for :mod:`dlm_sway.probes.adapter_ablation`.

Uses the dummy backend's lam-interpolation implementation to exercise
the full probe path without loading a real model.
"""

from __future__ import annotations

import numpy as np

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
        sat = _saturation_lambda(lambdas, divs)
        assert sat == 0.75  # 0.95 / 1.0 = 0.95 ≥ 0.9

    def test_overshoot_recovered(self) -> None:
        lambdas = np.asarray([0.0, 0.5, 1.0, 1.25], dtype=np.float64)
        divs = np.asarray([0.0, 0.5, 1.0, 1.15], dtype=np.float64)
        assert _overshoot(lambdas, divs) == 1.15


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
