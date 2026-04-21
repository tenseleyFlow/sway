"""Smoke test for the two-model differential wrapper.

Covers the ``defaults.differential: false`` code path: the runner
routes ``as_base()`` through one independent backend and
``as_finetuned()`` through another. Proper integration in S04.
"""

from __future__ import annotations

import numpy as np

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.backends.two_model import TwoModelDifferential
from dlm_sway.core.result import Verdict
from dlm_sway.core.scoring import TokenDist
from dlm_sway.suite.runner import run as run_suite
from dlm_sway.suite.spec import SwaySpec


def _backend_with(token: str, prob: float) -> DummyDifferentialBackend:
    """A dummy backend whose next-token dist puts ``prob`` mass on ``token``."""
    other = (1.0 - prob) / 2.0
    dist = TokenDist(
        token_ids=np.array([1, 2, 3], dtype=np.int64),
        logprobs=np.log(np.asarray([prob, other, other], dtype=np.float32)),
        vocab_size=100,
    )
    responses = DummyResponses(token_dists={"q1": dist, "q2": dist})
    return DummyDifferentialBackend(base=responses, ft=responses)


class TestRoutesBaseAndFTToTwoBackends:
    def test_two_backends_produce_distinct_dists(self) -> None:
        """Each side of the wrapper yields its own independent dist."""
        base_backend = _backend_with("a", 0.9)
        ft_backend = _backend_with("a", 0.1)
        wrapper = TwoModelDifferential(base=base_backend, ft=ft_backend)

        with wrapper.as_base() as v:
            base_dist = v.next_token_dist("q1")
        with wrapper.as_finetuned() as v:
            ft_dist = v.next_token_dist("q1")

        # First-token logprob on base should be high (low magnitude),
        # and on ft should be much lower (large negative magnitude).
        assert base_dist.logprobs[0] > ft_dist.logprobs[0], (
            f"expected base[0] > ft[0]; got base={base_dist.logprobs[0]}, ft={ft_dist.logprobs[0]}"
        )

    def test_runner_routes_through_wrapper_end_to_end(self) -> None:
        """A full suite run picks up divergence between the two backends."""
        base_backend = _backend_with("a", 0.9)
        ft_backend = _backend_with("a", 0.1)
        wrapper = TwoModelDifferential(base=base_backend, ft=ft_backend)

        spec = SwaySpec.model_validate(
            {
                "version": 1,
                "models": {
                    "base": {"base": "b"},
                    "ft": {"base": "b", "adapter": "/tmp/a"},
                },
                "defaults": {"differential": False},
                "suite": [
                    {
                        "name": "dk",
                        "kind": "delta_kl",
                        "prompts": ["q1", "q2"],
                        "assert_mean_gte": 0.0,
                    }
                ],
            }
        )
        result = run_suite(spec, wrapper)
        assert len(result.probes) == 1
        dk = result.probes[0]
        assert dk.verdict in (Verdict.PASS, Verdict.FAIL)
        # Real divergence between the two backends — raw must be > 0.
        assert dk.raw is not None
        assert dk.raw > 0.0


class TestPreflightPassthrough:
    def test_preflight_delegates_to_ft_backend(self) -> None:
        base = DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
        ft = DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
        wrapper = TwoModelDifferential(base=base, ft=ft)
        ok, _ = wrapper.preflight_finite_check()
        assert ok is True


class TestConcurrencyFlagComposition:
    """F06 regression — wrapper's ``safe_for_concurrent_views`` is the
    AND of the two inner backends' flags, defaulting to ``False`` when
    either is absent. Before F06, the attribute was missing entirely and
    the runner defaulted to ``False`` even when both inners set ``True``.
    """

    def test_missing_on_both_defaults_false(self) -> None:
        base = DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
        ft = DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
        wrapper = TwoModelDifferential(base=base, ft=ft)
        # Dummy has safe_for_concurrent_views=False by class default.
        assert wrapper.safe_for_concurrent_views is False

    def test_both_true_composes_true(self) -> None:
        class SafeDummy(DummyDifferentialBackend):
            safe_for_concurrent_views = True

        base = SafeDummy(base=DummyResponses(), ft=DummyResponses())
        ft = SafeDummy(base=DummyResponses(), ft=DummyResponses())
        wrapper = TwoModelDifferential(base=base, ft=ft)
        assert wrapper.safe_for_concurrent_views is True

    def test_one_true_one_false_composes_false(self) -> None:
        class SafeDummy(DummyDifferentialBackend):
            safe_for_concurrent_views = True

        base = SafeDummy(base=DummyResponses(), ft=DummyResponses())
        ft = DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
        wrapper = TwoModelDifferential(base=base, ft=ft)
        assert wrapper.safe_for_concurrent_views is False


class TestSpecAcceptsDifferentialFalse:
    def test_loader_accepts_false_then_uses_two_separate(self, tmp_path) -> None:
        """The CLI path: spec.defaults.differential=False routes through
        ``build_two_separate``. We don't exercise the actual HF backend
        here — the unit smoke is that the spec parses + the wrapper
        runs end-to-end on dummy."""
        spec = SwaySpec.model_validate(
            {
                "version": 1,
                "models": {
                    "base": {"base": "b"},
                    "ft": {"base": "b", "adapter": "/tmp/a"},
                },
                "defaults": {"differential": False},
                "suite": [],
            }
        )
        assert spec.defaults.differential is False
