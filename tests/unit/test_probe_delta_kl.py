"""Tests for :mod:`dlm_sway.probes.delta_kl`."""

from __future__ import annotations

import numpy as np

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.result import Verdict
from dlm_sway.core.scoring import TokenDist
from dlm_sway.probes.base import RunContext, build_probe


def _diverging_backend() -> DummyDifferentialBackend:
    """Base peaks tightly on token 1; ft is broad uniform. Real divergence."""
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
                logprobs=np.log(np.array([0.3, 0.35, 0.35], dtype=np.float32)),
                vocab_size=100,
            ),
            "q2": TokenDist(
                token_ids=np.array([5, 6], dtype=np.int64),
                logprobs=np.log(np.array([0.4, 0.6], dtype=np.float32)),
                vocab_size=100,
            ),
        }
    )
    return DummyDifferentialBackend(base=base, ft=ft)


def _identical_backend() -> DummyDifferentialBackend:
    dist = TokenDist(
        token_ids=np.array([1, 2, 3], dtype=np.int64),
        logprobs=np.log(np.array([0.5, 0.3, 0.2], dtype=np.float32)),
        vocab_size=100,
    )
    base = DummyResponses(token_dists={"q1": dist})
    ft = DummyResponses(token_dists={"q1": dist})
    return DummyDifferentialBackend(base=base, ft=ft)


class TestDeltaKL:
    def test_passes_when_distributions_diverge(self) -> None:
        probe, spec = build_probe(
            {
                "name": "dk",
                "kind": "delta_kl",
                "prompts": ["q1", "q2"],
                "assert_mean_gte": 0.01,
            }
        )
        ctx = RunContext(backend=_diverging_backend())
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.PASS
        assert result.raw is not None
        assert result.raw > 0.01
        assert result.evidence["num_prompts"] == 2
        assert len(result.evidence["per_prompt"]) == 2

    def test_fails_when_distributions_identical(self) -> None:
        probe, spec = build_probe(
            {
                "name": "dk",
                "kind": "delta_kl",
                "prompts": ["q1"],
                "assert_mean_gte": 0.01,
            }
        )
        ctx = RunContext(backend=_identical_backend())
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.FAIL
        assert result.raw == 0.0

    def test_z_score_path_when_null_stats_present(self) -> None:
        probe, spec = build_probe(
            {
                "name": "dk",
                "kind": "delta_kl",
                "prompts": ["q1"],
                "assert_z_gte": 2.0,
            }
        )
        null_stats = {"delta_kl": {"mean": 0.01, "std": 0.01, "n": 3.0}}
        ctx = RunContext(backend=_diverging_backend(), null_stats=null_stats)
        result = probe.run(spec, ctx)
        assert result.z_score is not None
        # Our synthetic ft diverges ~0.1+, far above μ=0.01, σ=0.01 → huge z.
        assert result.z_score > 2.0
        assert result.verdict == Verdict.PASS

    def test_error_on_empty_prompts(self) -> None:
        probe, spec = build_probe({"name": "dk", "kind": "delta_kl", "prompts": []})
        ctx = RunContext(backend=_identical_backend())
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.ERROR

    def test_kl_kind_available(self) -> None:
        probe, spec = build_probe(
            {
                "name": "dk",
                "kind": "delta_kl",
                "prompts": ["q1"],
                "divergence": "kl",
                "assert_mean_gte": 0.0,
            }
        )
        ctx = RunContext(backend=_diverging_backend())
        result = probe.run(spec, ctx)
        assert result.evidence["divergence_kind"] == "kl"
