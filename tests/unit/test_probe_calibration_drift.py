"""Tests for :mod:`dlm_sway.probes.calibration_drift`."""

from __future__ import annotations

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.result import Verdict
from dlm_sway.probes._calibration_pack import BUILT_IN_PACK
from dlm_sway.probes.base import RunContext, build_probe


def _backend(delta_per_token: float) -> DummyDifferentialBackend:
    """Apply a uniform per-token logprob delta across every item."""
    base_lp: dict[tuple[str, str], float] = {}
    ft_lp: dict[tuple[str, str], float] = {}
    for prompt, gold in BUILT_IN_PACK:
        base_lp[(prompt, gold)] = -5.0 * max(len(gold) // 4, 1)
        ft_lp[(prompt, gold)] = base_lp[(prompt, gold)] + delta_per_token * max(len(gold) // 4, 1)
    return DummyDifferentialBackend(
        base=DummyResponses(logprobs=base_lp),
        ft=DummyResponses(logprobs=ft_lp),
    )


class TestCalibrationDrift:
    def test_healthy_when_no_regression(self) -> None:
        backend = _backend(delta_per_token=0.0)  # no drift
        probe, spec = build_probe({"name": "c2", "kind": "calibration_drift"})
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.PASS
        assert result.raw == 0.0  # zero fraction regressed

    def test_fail_on_uniform_large_regression(self) -> None:
        backend = _backend(delta_per_token=-2.0)  # every item regresses
        probe, spec = build_probe({"name": "c2", "kind": "calibration_drift"})
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.FAIL
        assert result.raw == 1.0

    def test_respects_items_limit(self) -> None:
        backend = _backend(delta_per_token=0.0)
        probe, spec = build_probe({"name": "c2", "kind": "calibration_drift", "items_limit": 5})
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        assert result.evidence["total_items"] == 5

    def test_worst_offenders_reported(self) -> None:
        backend = _backend(delta_per_token=-2.0)
        probe, spec = build_probe({"name": "c2", "kind": "calibration_drift"})
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        worst = result.evidence["worst_offenders"]
        assert len(worst) <= 5
        # Each worst-offender record carries prompt/gold/delta fields.
        if worst:
            assert {"prompt", "gold", "delta"} <= set(worst[0].keys())
