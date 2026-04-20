"""Tests for null-adapter calibration.

Covers: dummy backend ``as_null_adapter`` yields a plausibly noisy
view; ``NullAdapterProbe`` populates ``ctx.null_stats`` in a way
downstream probes pick up end-to-end; missing-capability SKIP path.
"""

from __future__ import annotations

import numpy as np

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.result import Verdict
from dlm_sway.core.scoring import NullCalibratedBackend
from dlm_sway.probes.base import RunContext, build_probe
from dlm_sway.suite.runner import run as run_suite
from dlm_sway.suite.spec import SwaySpec


def _diverging_backend() -> DummyDifferentialBackend:
    base = DummyResponses()
    ft = DummyResponses()
    return DummyDifferentialBackend(base=base, ft=ft)


class TestProtocolConformance:
    def test_dummy_is_null_calibrated(self) -> None:
        assert isinstance(_diverging_backend(), NullCalibratedBackend)


class TestAsNullAdapter:
    def test_yields_perturbed_view(self) -> None:
        backend = _diverging_backend()
        with backend.as_base() as base:
            base_dist = base.next_token_dist("hello")
        with backend.as_null_adapter(seed=0) as null:
            null_dist = null.next_token_dist("hello")
        # Some perturbation, but bounded.
        assert not np.allclose(base_dist.logprobs, null_dist.logprobs)

    def test_different_seeds_yield_different_views(self) -> None:
        backend = _diverging_backend()
        with backend.as_null_adapter(seed=1) as v1:
            d1 = v1.next_token_dist("hello")
        with backend.as_null_adapter(seed=2) as v2:
            d2 = v2.next_token_dist("hello")
        assert not np.allclose(d1.logprobs, d2.logprobs)

    def test_view_exclusion_enforced(self) -> None:
        import pytest

        backend = _diverging_backend()
        with backend.as_null_adapter(seed=0), pytest.raises(RuntimeError):
            with backend.as_base():
                pass


class TestProbe:
    def test_populates_null_stats(self) -> None:
        backend = _diverging_backend()
        probe, spec = build_probe(
            {
                "name": "null",
                "kind": "null_adapter",
                "runs": 3,
                "prompts": ["q1", "q2"],
            }
        )
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.PASS
        stats = result.evidence["null_stats"]
        assert "delta_kl" in stats
        assert stats["delta_kl"]["n"] == 3.0
        assert stats["delta_kl"]["std"] > 0.0  # seeded perturbations produce variance

    def test_runner_threads_null_stats_to_subsequent_probes(self) -> None:
        """End-to-end: null_adapter first → delta_kl picks up z-score path."""
        backend = _diverging_backend()
        raw_spec = SwaySpec.model_validate(
            {
                "version": 1,
                "models": {"base": {"base": "b"}, "ft": {"base": "b", "adapter": "/tmp/a"}},
                "suite": [
                    {
                        "name": "null",
                        "kind": "null_adapter",
                        "runs": 3,
                        "prompts": ["p1", "p2"],
                    },
                    {
                        "name": "dk",
                        "kind": "delta_kl",
                        "prompts": ["p1", "p2"],
                        "assert_z_gte": -10.0,  # permissive so we pass regardless
                    },
                ],
            }
        )
        result = run_suite(raw_spec, backend)
        assert len(result.probes) == 2
        null_result = result.probes[0]
        dk_result = result.probes[1]
        assert null_result.verdict == Verdict.PASS
        # The delta_kl probe should have computed a z_score because null_stats was present.
        assert dk_result.z_score is not None, (
            "delta_kl should have z-scored against null baseline, got "
            f"evidence={dk_result.evidence}, message={dk_result.message}"
        )

    def test_skip_when_backend_not_null_calibrated(self) -> None:
        class _Bare:
            def as_base(self):  # noqa: ANN202
                raise NotImplementedError

            def as_finetuned(self):  # noqa: ANN202
                raise NotImplementedError

        probe, spec = build_probe({"name": "null", "kind": "null_adapter"})
        ctx = RunContext(backend=_Bare())  # type: ignore[arg-type]
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.SKIP
        assert "NullCalibratedBackend" in result.message
