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
        """Three prompts with planted divergences: ``hi`` has the
        biggest gap, ``lo`` the smallest. Top-1 = hi, bottom-1 = lo."""
        base = _dist_from_probs([0.92, 0.02, 0.02, 0.02, 0.02])
        ft_flat = _dist_from_probs([0.25, 0.20, 0.20, 0.20, 0.15])  # big KL
        ft_mild = _dist_from_probs([0.70, 0.10, 0.10, 0.05, 0.05])  # mid KL
        ft_same = base  # zero KL

        base_dists = {"hi": base, "mid": base, "lo": base}
        ft_dists = {"hi": ft_flat, "mid": ft_mild, "lo": ft_same}
        backend = DummyDifferentialBackend(
            base=DummyResponses(token_dists=base_dists),
            ft=DummyResponses(token_dists=ft_dists),
        )

        result = mine_outliers(
            probe_kind="delta_kl",
            candidate_prompts=["hi", "mid", "lo"],
            backend=backend,
            top_k=3,
        )

        assert isinstance(result, OutlierResult)
        assert result.probe_kind == "delta_kl"
        # Top is ordered most-positive first.
        assert [c.prompt for c in result.top] == ["hi", "mid", "lo"]
        # Bottom is ordered least-positive first.
        assert [c.prompt for c in result.bottom] == ["lo", "mid", "hi"]
        # Raw values are finite and positive (JS divergence ≥ 0).
        for c in result.top:
            assert math.isfinite(c.raw)
            assert c.raw >= 0.0

    def test_top_k_clipped_to_pool_size(self) -> None:
        base = _dist_from_probs([0.92, 0.02, 0.02, 0.02, 0.02])
        ft = _dist_from_probs([0.25, 0.20, 0.20, 0.20, 0.15])
        backend = DummyDifferentialBackend(
            base=DummyResponses(token_dists={"p": base}),
            ft=DummyResponses(token_dists={"p": ft}),
        )
        result = mine_outliers(
            probe_kind="delta_kl",
            candidate_prompts=["p"],
            backend=backend,
            top_k=10,
        )
        assert len(result.top) == 1
        assert len(result.bottom) == 1

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
        every candidate silently. S17 scope is delta_kl; other probes
        are documented as future work."""
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
