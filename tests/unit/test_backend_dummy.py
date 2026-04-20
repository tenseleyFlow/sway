"""Tests for :class:`dlm_sway.backends.dummy.DummyDifferentialBackend`.

The dummy backend is used by every downstream probe unit test, so it
gets a thorough own-right test here. Also verifies the view-exclusion
invariant that catches stale-view bugs in probes.
"""

from __future__ import annotations

import numpy as np
import pytest

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.model import Model
from dlm_sway.core.scoring import DifferentialBackend, ScoringBackend


@pytest.fixture
def backend() -> DummyDifferentialBackend:
    base = DummyResponses(
        generations={"hi": "hello"},
        logprobs={("q", "a"): -3.0},
    )
    ft = DummyResponses(
        generations={"hi": "greetings, traveler"},
        logprobs={("q", "a"): -1.2},
    )
    return DummyDifferentialBackend(base=base, ft=ft)


class TestViews:
    def test_as_base_and_as_ft_yield_distinct_generations(
        self, backend: DummyDifferentialBackend
    ) -> None:
        with backend.as_base() as b:
            assert b.generate("hi", max_new_tokens=5) == "hello"
        with backend.as_finetuned() as f:
            assert f.generate("hi", max_new_tokens=5) == "greetings, traveler"

    def test_logprob_differs_between_modes(self, backend: DummyDifferentialBackend) -> None:
        with backend.as_base() as b:
            base_score = b.logprob_of("q", "a")
        with backend.as_finetuned() as f:
            ft_score = f.logprob_of("q", "a")
        assert base_score == -3.0
        assert ft_score == -1.2

    def test_missing_generation_raises_keyerror(self, backend: DummyDifferentialBackend) -> None:
        with backend.as_base() as b, pytest.raises(KeyError, match="no canned generation"):
            b.generate("unconfigured", max_new_tokens=1)

    def test_missing_logprob_default(self, backend: DummyDifferentialBackend) -> None:
        with backend.as_base() as b:
            assert b.logprob_of("nonexistent", "target") == -10.0


class TestRollingLogprob:
    def test_synthesized_when_not_preseeded(self, backend: DummyDifferentialBackend) -> None:
        with backend.as_base() as b:
            r = b.rolling_logprob("a quick brown fox jumps")
        assert r.num_tokens == 5
        assert r.logprobs.size == 4
        assert np.all(r.logprobs == -2.0)

    def test_ft_perplexity_lower_than_base(self, backend: DummyDifferentialBackend) -> None:
        text = "a quick brown fox"
        with backend.as_base() as b:
            pb = b.rolling_logprob(text).perplexity
        with backend.as_finetuned() as f:
            pf = f.rolling_logprob(text).perplexity
        assert pf < pb  # synthesized ft is less perplexed → lower PPL


class TestTokenDist:
    def test_dists_differ_between_modes(self, backend: DummyDifferentialBackend) -> None:
        with backend.as_base() as b:
            base_dist = b.next_token_dist("any prompt")
        with backend.as_finetuned() as f:
            ft_dist = f.next_token_dist("any prompt")
        assert not np.array_equal(base_dist.logprobs, ft_dist.logprobs)


class TestInvariants:
    def test_protocol_satisfaction(self, backend: DummyDifferentialBackend) -> None:
        assert isinstance(backend, DifferentialBackend)
        with backend.as_base() as view:
            assert isinstance(view, Model)
            assert isinstance(view, ScoringBackend)

    def test_nested_views_rejected(self, backend: DummyDifferentialBackend) -> None:
        with backend.as_base(), pytest.raises(RuntimeError, match="view already active"):
            with backend.as_finetuned():
                pass

    def test_sequential_views_fine(self, backend: DummyDifferentialBackend) -> None:
        # Must be able to re-enter after exiting — common pattern in probes.
        with backend.as_base() as b:
            b.logprob_of("q", "a")
        with backend.as_finetuned() as f:
            f.logprob_of("q", "a")
        with backend.as_base() as b:
            b.logprob_of("q", "a")
