"""Tests for :mod:`dlm_sway.mining.outlier_miner`.

Uses the dummy backend's synthesized per-prompt divergences: base is
sharply peaked, ft is broad, and the dummy's ``next_token_dist`` cache
keys on the prompt string — so each candidate prompt produces the same
divergence unless we overlay per-prompt TokenDists. Tests that need
variation build per-prompt TokenDists directly on ``DummyResponses``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.scoring import TokenDist
from dlm_sway.mining.outlier_miner import (
    OutlierCandidate,
    OutlierResult,
    corpus_prompts,
    mine_outliers,
)


def _dist_from_probs(probs: list[float]) -> TokenDist:
    arr = np.asarray(probs, dtype=np.float64)
    arr = arr / arr.sum()
    lp = np.log(arr).astype(np.float32)
    return TokenDist(
        token_ids=np.arange(len(probs), dtype=np.int64),
        logprobs=lp,
        vocab_size=max(1000, len(probs)),
        tail_logprob=None,
    )


class TestMineOutliers:
    def test_ranks_prompts_by_per_prompt_divergence(self) -> None:
        """Six prompts with planted divergences: ``hi*`` have the biggest
        gap, ``lo*`` the smallest, ``mid*`` in between. Top-K = hi rows,
        bottom-K = lo rows."""
        base = _dist_from_probs([0.92, 0.02, 0.02, 0.02, 0.02])
        ft_flat = _dist_from_probs([0.25, 0.20, 0.20, 0.20, 0.15])  # big KL
        ft_mild = _dist_from_probs([0.70, 0.10, 0.10, 0.05, 0.05])  # mid KL
        ft_same = base  # zero KL

        # F04 — need ≥ 2·top_k=4 distinct prompts to clear the guard.
        prompts = ["hi1", "hi2", "mid1", "mid2", "lo1", "lo2"]
        base_dists = dict.fromkeys(prompts, base)
        ft_dists = {
            "hi1": ft_flat,
            "hi2": ft_flat,
            "mid1": ft_mild,
            "mid2": ft_mild,
            "lo1": ft_same,
            "lo2": ft_same,
        }
        backend = DummyDifferentialBackend(
            base=DummyResponses(token_dists=base_dists),
            ft=DummyResponses(token_dists=ft_dists),
        )

        result = mine_outliers(
            probe_kind="delta_kl",
            candidate_prompts=prompts,
            backend=backend,
            top_k=2,
        )

        assert isinstance(result, OutlierResult)
        assert result.probe_kind == "delta_kl"
        # Top is most-positive first; bottom is least-positive first.
        top_prompts = {c.prompt for c in result.top}
        bottom_prompts = {c.prompt for c in result.bottom}
        assert top_prompts == {"hi1", "hi2"}
        assert bottom_prompts == {"lo1", "lo2"}
        # Raw values are finite and positive (JS divergence ≥ 0).
        for c in result.top:
            assert math.isfinite(c.raw)
            assert c.raw >= 0.0

    def test_small_pool_raises_f04_guard(self) -> None:
        """F04 (Audit 03) — pool below ``2·top_k`` distinct prompts
        raises SwayError with an actionable hint. Replaces pre-F04
        'test_top_k_clipped_to_pool_size' which relied on the same
        degenerate single-prompt case the audit flagged as produced
        top=[p], bottom=[p] — identical lists."""
        from dlm_sway.core.errors import SwayError

        base = _dist_from_probs([0.92, 0.02, 0.02, 0.02, 0.02])
        ft = _dist_from_probs([0.25, 0.20, 0.20, 0.20, 0.15])
        backend = DummyDifferentialBackend(
            base=DummyResponses(token_dists={"p": base}),
            ft=DummyResponses(token_dists={"p": ft}),
        )
        with pytest.raises(SwayError, match="below the 2·top_k"):
            mine_outliers(
                probe_kind="delta_kl",
                candidate_prompts=["p"],
                backend=backend,
                top_k=10,
            )

    def test_small_pool_error_suggests_smaller_top_k(self) -> None:
        """The error message includes a concrete ``--top-k N`` hint the
        user can copy into their CLI invocation."""
        from dlm_sway.core.errors import SwayError

        base = _dist_from_probs([0.92, 0.02, 0.02, 0.02, 0.02])
        ft = _dist_from_probs([0.25, 0.20, 0.20, 0.20, 0.15])
        prompts = ["p1", "p2", "p3"]
        backend = DummyDifferentialBackend(
            base=DummyResponses(token_dists=dict.fromkeys(prompts, base)),
            ft=DummyResponses(token_dists=dict.fromkeys(prompts, ft)),
        )
        with pytest.raises(SwayError, match="Pass --top-k 1"):
            mine_outliers(
                probe_kind="delta_kl",
                candidate_prompts=prompts,
                backend=backend,
                top_k=5,
            )

    def test_empty_pool_returns_empty_result(self) -> None:
        backend = DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
        result = mine_outliers(
            probe_kind="delta_kl",
            candidate_prompts=[],
            backend=backend,
            top_k=5,
        )
        assert result.top == []
        assert result.bottom == []

    def test_unsupported_probe_kind_returns_empty(self) -> None:
        """Probes that need a non-``prompts`` spec (leakage, etc.) skip
        every candidate silently. The F04 floor doesn't fire in that
        case because the scored list is empty — empty-result path
        preserved for the unsupported-kind UX."""
        base = _dist_from_probs([0.92, 0.02, 0.02, 0.02, 0.02])
        ft = _dist_from_probs([0.25, 0.20, 0.20, 0.20, 0.15])
        backend = DummyDifferentialBackend(
            base=DummyResponses(token_dists={"p": base}),
            ft=DummyResponses(token_dists={"p": ft}),
        )
        result = mine_outliers(
            probe_kind="leakage",
            candidate_prompts=["p"],
            backend=backend,
            top_k=5,
        )
        assert result.top == []
        assert result.bottom == []

    def test_rejects_top_k_zero(self) -> None:
        backend = DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
        with pytest.raises(ValueError, match="top_k must be positive"):
            mine_outliers(
                probe_kind="delta_kl",
                candidate_prompts=["a"],
                backend=backend,
                top_k=0,
            )


class TestCorpusPrompts:
    def test_pulls_chunks_from_public_domain(self) -> None:
        """``--from-corpus public_domain_en`` yields a list of strings
        long enough for probe scoring."""
        chunks = corpus_prompts("public_domain_en", chunk_chars=512, max_chunks=4)
        assert chunks
        assert len(chunks) <= 4
        for c in chunks:
            assert isinstance(c, str)
            assert len(c) >= 64  # chunk_corpus's minimum

    def test_unknown_corpus_raises(self) -> None:
        with pytest.raises(KeyError):
            corpus_prompts("doesnotexist")


class TestOutlierCandidate:
    def test_is_frozen(self) -> None:
        c = OutlierCandidate(prompt="p", raw=0.5, index=0)
        with pytest.raises(Exception):  # noqa: B017, PT011
            c.raw = 0.0  # type: ignore[misc]
