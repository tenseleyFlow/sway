"""Tests for :mod:`dlm_sway.probes.adapter_revert`.

We stub out the embedder so these tests don't need sentence-transformers
installed. The ``probe.py`` SKIP path for the missing-extra case is
covered separately by monkeypatching the importer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.result import Verdict
from dlm_sway.probes.adapter_revert import AdapterRevertProbe
from dlm_sway.probes.base import RunContext, build_probe


def _backend(*, ft_like_base: bool = False) -> DummyDifferentialBackend:
    base = DummyResponses(
        generations={
            "pp1": "cats are mammals",
            "pp2": "cats have fur",
        }
    )
    if ft_like_base:
        ft_gens = dict(base.generations)
    else:
        ft_gens = {
            "pp1": "dolphins are mammals",
            "pp2": "dolphins are smart",
        }
    ft = DummyResponses(generations=ft_gens)
    return DummyDifferentialBackend(base=base, ft=ft)


def _stub_embedder(text_to_vec: dict[str, np.ndarray]):  # type: ignore[no-untyped-def]
    def _encode(texts: list[str]):  # type: ignore[no-untyped-def]
        return np.stack([text_to_vec[t] for t in texts])

    return _encode


@pytest.fixture
def monkeyed_embed(monkeypatch: pytest.MonkeyPatch) -> dict[str, np.ndarray]:
    """Install a stub embedder with a controllable text→vec mapping.

    Tests populate the dict before calling ``probe.run()``.
    """
    table: dict[str, np.ndarray] = {}
    monkeypatch.setattr(
        "dlm_sway.probes.adapter_revert._load_embedder",
        lambda _model_id: _stub_embedder(table),  # type: ignore[arg-type]
    )
    return table


class TestAdapterRevert:
    def test_healthy_adapter_passes(self, monkeyed_embed: dict[str, np.ndarray]) -> None:
        # gold and ft-outputs cluster together, base outputs cluster elsewhere.
        monkeyed_embed["cats are mammals"] = np.array([1.0, 0.0])
        monkeyed_embed["cats have fur"] = np.array([1.0, 0.0])
        monkeyed_embed["dolphins are mammals"] = np.array([0.0, 1.0])
        monkeyed_embed["dolphins are smart"] = np.array([0.0, 1.0])
        monkeyed_embed["the answer is dolphins"] = np.array([0.0, 1.0])  # gold

        probe, spec = build_probe(
            {
                "name": "rev",
                "kind": "adapter_revert",
                "cases": [
                    {
                        "prompt": "anything",
                        "gold": "the answer is dolphins",
                        "paraphrases": ["pp1", "pp2"],
                    }
                ],
                "assert_revert_rate_lt": 0.25,
            }
        )
        ctx = RunContext(backend=_backend(ft_like_base=False))
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.PASS
        assert result.raw == 0.0

    def test_reverting_adapter_fails(self, monkeyed_embed: dict[str, np.ndarray]) -> None:
        # ft matches base (reverted), diverges from gold.
        monkeyed_embed["cats are mammals"] = np.array([1.0, 0.0])
        monkeyed_embed["cats have fur"] = np.array([1.0, 0.0])
        monkeyed_embed["the answer is dolphins"] = np.array([0.0, 1.0])  # gold

        probe, spec = build_probe(
            {
                "name": "rev",
                "kind": "adapter_revert",
                "cases": [
                    {
                        "prompt": "anything",
                        "gold": "the answer is dolphins",
                        "paraphrases": ["pp1", "pp2"],
                    }
                ],
            }
        )
        ctx = RunContext(backend=_backend(ft_like_base=True))
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.FAIL
        assert result.raw == 1.0  # 100% revert

    def test_trivially_similar_cases_dropped(self, monkeyed_embed: dict[str, np.ndarray]) -> None:
        # base and gold are identical → drop.
        v = np.array([1.0, 0.0])
        monkeyed_embed["cats are mammals"] = v
        monkeyed_embed["cats have fur"] = v
        monkeyed_embed["dolphins are mammals"] = np.array([0.0, 1.0])
        monkeyed_embed["dolphins are smart"] = np.array([0.0, 1.0])
        monkeyed_embed["cats are mammals too"] = v  # gold — matches base

        probe, spec = build_probe(
            {
                "name": "rev",
                "kind": "adapter_revert",
                "cases": [
                    {
                        "prompt": "anything",
                        "gold": "cats are mammals too",
                        "paraphrases": ["pp1", "pp2"],
                    }
                ],
            }
        )
        ctx = RunContext(backend=_backend(ft_like_base=False))
        result = probe.run(spec, ctx)
        # Both paraphrase pairs trivially similar → WARN (no separable signal).
        assert result.verdict == Verdict.WARN
        assert result.evidence["dropped_trivial"] == 2

    def test_no_cases_errors(self, monkeyed_embed: dict[str, np.ndarray]) -> None:
        probe, spec = build_probe({"name": "rev", "kind": "adapter_revert", "cases": []})
        ctx = RunContext(backend=_backend())
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.ERROR


class TestMissingSemsim:
    def test_skip_when_sentence_transformers_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from dlm_sway.core.errors import BackendNotAvailableError

        def raiser(_model_id: Any) -> Any:  # type: ignore[no-untyped-def]
            raise BackendNotAvailableError(
                "adapter_revert",
                extra="semsim",
                hint="adapter_revert relies on sentence embeddings.",
            )

        monkeypatch.setattr(
            "dlm_sway.probes.adapter_revert._load_embedder",
            raiser,  # type: ignore[arg-type]
        )
        probe = AdapterRevertProbe()
        spec = probe.spec_cls(
            name="rev",
            cases=[{"prompt": "x", "gold": "y", "paraphrases": ["pp1"]}],  # type: ignore[list-item]
        )
        ctx = RunContext(backend=_backend())
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.SKIP
        assert "semsim" in result.message
