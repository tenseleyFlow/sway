"""Tests for :mod:`dlm_sway.core.scoring`."""

from __future__ import annotations

import math

import numpy as np

from dlm_sway.core.scoring import (
    DifferentialBackend,
    RollingLogprob,
    ScoringBackend,
    TokenDist,
)


class TestRollingLogprob:
    def test_empty_sequence(self) -> None:
        r = RollingLogprob(
            token_ids=np.array([42], dtype=np.int64),
            logprobs=np.array([], dtype=np.float32),
            num_tokens=1,
            total_logprob=0.0,
        )
        assert r.mean_logprob == 0.0
        assert r.perplexity == 1.0

    def test_mean_and_perplexity(self) -> None:
        # Three tokens, two transition logprobs summing to -4.0 → mean -2.0.
        r = RollingLogprob(
            token_ids=np.array([1, 2, 3], dtype=np.int64),
            logprobs=np.array([-1.5, -2.5], dtype=np.float32),
            num_tokens=3,
            total_logprob=-4.0,
        )
        assert math.isclose(r.mean_logprob, -2.0, rel_tol=1e-6)
        assert math.isclose(r.perplexity, math.exp(2.0), rel_tol=1e-6)


class TestTokenDist:
    def test_construction_and_defaults(self) -> None:
        dist = TokenDist(
            token_ids=np.array([1, 2, 3], dtype=np.int64),
            logprobs=np.array([-0.1, -1.0, -3.0], dtype=np.float32),
            vocab_size=50_257,
        )
        assert dist.tail_logprob == 0.0
        assert dist.token_ids.shape == (3,)


class TestProtocols:
    def test_scoring_backend_runtime_checkable(self) -> None:
        class FakeScoring:
            def logprob_of(self, prompt: str, completion: str) -> float:
                return 0.0

            def rolling_logprob(self, text: str) -> RollingLogprob:
                return RollingLogprob(
                    token_ids=np.array([0], dtype=np.int64),
                    logprobs=np.array([], dtype=np.float32),
                    num_tokens=1,
                    total_logprob=0.0,
                )

            def next_token_dist(self, prompt: str, *, top_k: int = 256) -> TokenDist:
                return TokenDist(
                    token_ids=np.array([0], dtype=np.int64),
                    logprobs=np.array([0.0], dtype=np.float32),
                    vocab_size=1,
                )

        assert isinstance(FakeScoring(), ScoringBackend)

    def test_differential_backend_runtime_checkable(self) -> None:
        from contextlib import nullcontext

        class FakeDiff:
            def as_base(self):  # type: ignore[no-untyped-def]
                return nullcontext(object())

            def as_finetuned(self):  # type: ignore[no-untyped-def]
                return nullcontext(object())

        assert isinstance(FakeDiff(), DifferentialBackend)
