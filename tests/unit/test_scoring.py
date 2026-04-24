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
        # B6: default tail_logprob is None ("no tail recorded"), not
        # 0.0 (which now means "tail underflowed to zero, but exists").
        assert dist.tail_logprob is None
        assert dist.token_ids.shape == (3,)

    def test_explicit_tail_distinguishes_zero_from_none(self) -> None:
        """B6: 0.0 means measurable-but-tiny; None means no tail at all."""
        d_no_tail = TokenDist(
            token_ids=np.array([1], dtype=np.int64),
            logprobs=np.array([0.0], dtype=np.float32),
            vocab_size=1,
            tail_logprob=None,
        )
        d_underflow = TokenDist(
            token_ids=np.array([1], dtype=np.int64),
            logprobs=np.array([0.0], dtype=np.float32),
            vocab_size=1,
            tail_logprob=0.0,
        )
        assert d_no_tail.tail_logprob is None
        assert d_underflow.tail_logprob == 0.0


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

            def next_token_dist_batch(
                self,
                prompts,  # type: ignore[no-untyped-def]
                *,
                top_k: int = 256,
            ) -> list[TokenDist]:
                # S23 — Protocol requires the batched method at
                # runtime. Defer to the single-prompt path; enough to
                # satisfy the runtime_checkable isinstance check.
                return [self.next_token_dist(p, top_k=top_k) for p in prompts]

        assert isinstance(FakeScoring(), ScoringBackend)

    def test_differential_backend_runtime_checkable(self) -> None:
        from contextlib import nullcontext

        class FakeDiff:
            def as_base(self):  # type: ignore[no-untyped-def]
                return nullcontext(object())

            def as_finetuned(self):  # type: ignore[no-untyped-def]
                return nullcontext(object())

        assert isinstance(FakeDiff(), DifferentialBackend)
