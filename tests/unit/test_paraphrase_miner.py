"""Tests for :mod:`dlm_sway.mining.paraphrase_miner`.

We stub the embedder + the nlpaug generator so tests run without the
80-MB MiniLM load or the nlpaug wheel. What's under test is the ranker
and the diversity filter — the infrastructure around them (the
embedder, the generator) is exercised by the integration test.
"""

from __future__ import annotations

import numpy as np
import pytest

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.mining.paraphrase_miner import (
    MiningResult,
    ParaphraseCandidate,
    mine_paraphrases,
)


def _stub_embedder(text_to_vec: dict[str, np.ndarray]):  # type: ignore[no-untyped-def]
    def _encode(texts: list[str]):  # type: ignore[no-untyped-def]
        return np.stack([text_to_vec[t] for t in texts])

    return _encode


@pytest.fixture
def monkeyed_embed(monkeypatch: pytest.MonkeyPatch) -> dict[str, np.ndarray]:
    table: dict[str, np.ndarray] = {}
    monkeypatch.setattr(
        "dlm_sway.mining.paraphrase_miner._load_embedder",
        lambda _model_id: _stub_embedder(table),  # type: ignore[arg-type]
    )
    return table


def _canned_candidates(candidates: list[str]):  # type: ignore[no-untyped-def]
    """Build a generator closure that returns ``candidates`` verbatim."""

    def _gen(_prompt: str, *, n: int, seed: int) -> list[str]:
        del n, seed
        return list(candidates)

    return _gen


class TestRanker:
    def test_gap_sort_ranks_hardest_first(self, monkeyed_embed: dict[str, np.ndarray]) -> None:
        """The paraphrase with the largest verbatim-vs-paraphrase lift
        gap ranks first. Planted: three candidates with known
        ``(base, ft)`` logprob pairs; the ranker must surface them in
        gap-descending order regardless of the generator's input
        order."""
        # Seed + candidates
        prompt = "Q: what is the capital?"
        gold = " Paris"
        candidates = ["Q1", "Q2", "Q3"]
        # Embed all four distinctly so the diversity filter keeps
        # every candidate.
        monkeyed_embed[prompt] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        monkeyed_embed["Q1"] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        monkeyed_embed["Q2"] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        monkeyed_embed["Q3"] = np.array([0.5, 0.5, 0.0], dtype=np.float32)

        # Logprob table — verbatim lift is large; paraphrase lifts
        # vary by design: Q1 lift ≈ 0 (big gap), Q2 lift ≈ verbatim
        # (no gap), Q3 lift somewhere in between.
        base_logprobs = {
            (prompt, gold): -4.0,
            ("Q1", gold): -4.0,  # base unchanged
            ("Q2", gold): -4.0,
            ("Q3", gold): -4.0,
        }
        ft_logprobs = {
            (prompt, gold): -1.0,  # verbatim lift: +3.0
            ("Q1", gold): -4.0,  # paraphrase lift: 0 (big gap of 3)
            ("Q2", gold): -1.0,  # paraphrase lift: +3 (gap of 0)
            ("Q3", gold): -2.5,  # paraphrase lift: +1.5 (gap of 1.5)
        }
        backend = DummyDifferentialBackend(
            base=DummyResponses(logprobs=base_logprobs),
            ft=DummyResponses(logprobs=ft_logprobs),
        )

        result = mine_paraphrases(
            prompt=prompt,
            gold=gold,
            backend=backend,
            generate_candidates=_canned_candidates(candidates),
            n_candidates=3,
            top_k=3,
            seed=0,
        )

        assert isinstance(result, MiningResult)
        assert [c.prompt for c in result.candidates] == ["Q1", "Q3", "Q2"]
        # Gaps are in descending order.
        gaps = [c.gap for c in result.candidates]
        assert gaps == sorted(gaps, reverse=True)
        # Top candidate's gap is meaningfully larger than 0.
        assert result.candidates[0].gap > 0.1

    def test_dedup_drops_verbatim_seed(self, monkeyed_embed: dict[str, np.ndarray]) -> None:
        """If the generator echoes the seed prompt back in its output,
        it's dropped before the ranker sees it — otherwise the seed
        would bubble to the top with gap = 0 and pollute the list."""
        prompt = "prompt"
        gold = " gold"
        # Generator returns the seed + two real candidates; the seed
        # must not appear in the final result.
        candidates = ["prompt", "C1", "C2"]
        for p in candidates + [prompt]:
            monkeyed_embed[p] = (
                np.random.RandomState(hash(p) & 0xFFFFFFFF).randn(4).astype(np.float32)
            )

        base_logprobs = {(p, gold): -3.0 for p in [prompt, "C1", "C2"]}
        ft_logprobs = {(prompt, gold): -1.0, ("C1", gold): -2.5, ("C2", gold): -2.0}
        backend = DummyDifferentialBackend(
            base=DummyResponses(logprobs=base_logprobs),
            ft=DummyResponses(logprobs=ft_logprobs),
        )

        result = mine_paraphrases(
            prompt=prompt,
            gold=gold,
            backend=backend,
            generate_candidates=_canned_candidates(candidates),
            n_candidates=3,
            top_k=3,
            seed=0,
        )
        assert prompt not in {c.prompt for c in result.candidates}

    def test_empty_candidate_list_returns_empty_result(
        self, monkeyed_embed: dict[str, np.ndarray]
    ) -> None:
        del monkeyed_embed
        backend = DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
        result = mine_paraphrases(
            prompt="x",
            gold=" y",
            backend=backend,
            generate_candidates=_canned_candidates([]),
            n_candidates=3,
            top_k=3,
        )
        assert isinstance(result, MiningResult)
        assert result.candidates == []


class TestDiversityFilter:
    def test_keeps_farthest_when_candidates_cluster(
        self, monkeyed_embed: dict[str, np.ndarray]
    ) -> None:
        """Two near-duplicate candidates + one distant one → the
        distant candidate must survive the k=2 diversity filter."""
        prompt = "seed"
        gold = " gold"
        candidates = ["near1", "near2", "far"]
        monkeyed_embed[prompt] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # near1 / near2 collapse onto the same embedding; far lives
        # orthogonally. With k=2, the filter must pick one near + far
        # (not both nears).
        monkeyed_embed["near1"] = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        monkeyed_embed["near2"] = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        monkeyed_embed["far"] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        base_logprobs = {("near1", gold): -3.0, ("near2", gold): -3.0, ("far", gold): -3.0}
        ft_logprobs = {("near1", gold): -2.0, ("near2", gold): -2.0, ("far", gold): -1.0}
        base_logprobs[(prompt, gold)] = -3.0
        ft_logprobs[(prompt, gold)] = -1.0
        backend = DummyDifferentialBackend(
            base=DummyResponses(logprobs=base_logprobs),
            ft=DummyResponses(logprobs=ft_logprobs),
        )

        result = mine_paraphrases(
            prompt=prompt,
            gold=gold,
            backend=backend,
            generate_candidates=_canned_candidates(candidates),
            n_candidates=3,
            top_k=2,
            seed=0,
        )
        chosen = {c.prompt for c in result.candidates}
        assert "far" in chosen
        assert len(chosen & {"near1", "near2"}) == 1


class TestInputValidation:
    def test_rejects_top_k_zero(self, monkeyed_embed: dict[str, np.ndarray]) -> None:
        del monkeyed_embed
        backend = DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
        with pytest.raises(ValueError, match="top_k must be positive"):
            mine_paraphrases(
                prompt="x",
                gold=" y",
                backend=backend,
                generate_candidates=_canned_candidates(["a"]),
                n_candidates=5,
                top_k=0,
            )

    def test_rejects_n_candidates_below_top_k(self, monkeyed_embed: dict[str, np.ndarray]) -> None:
        del monkeyed_embed
        backend = DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
        with pytest.raises(ValueError, match="must be ≥ top_k"):
            mine_paraphrases(
                prompt="x",
                gold=" y",
                backend=backend,
                generate_candidates=_canned_candidates(["a"]),
                n_candidates=2,
                top_k=5,
            )


class TestParaphraseCandidate:
    def test_is_frozen_dataclass(self) -> None:
        c = ParaphraseCandidate(
            prompt="p", gap=0.5, verbatim_lift=1.0, paraphrase_lift=0.5, diversity_rank=0
        )
        with pytest.raises(Exception):  # noqa: B017, PT011 — FrozenInstanceError varies
            c.gap = 0.0  # type: ignore[misc]
