"""Smoke tests for ``sway mine`` — S17 CLI surface.

Follows the test_sway_gate_exit_code pattern: stub ``backends.build``
with a dummy-returning factory so the CLI runs without loading a real
HF model. The paraphrase generator is also stubbed (via monkeypatch
on ``dlm_sway.mining.paraphrase_miner.nlpaug_candidates``) so tests
don't need the nlpaug wheel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.cli.app import app
from dlm_sway.core.scoring import TokenDist


def _programmable_backend() -> DummyDifferentialBackend:
    """Dummy backend with predictable logprobs + TokenDists so both
    mine modes have something to chew on."""
    base_dist = TokenDist(
        token_ids=np.arange(5, dtype=np.int64),
        logprobs=np.log(np.array([0.92, 0.02, 0.02, 0.02, 0.02], dtype=np.float32)),
        vocab_size=1000,
    )
    ft_dist = TokenDist(
        token_ids=np.arange(5, dtype=np.int64),
        logprobs=np.log(np.array([0.25, 0.20, 0.20, 0.20, 0.15], dtype=np.float32)),
        vocab_size=1000,
    )
    prompts = ["p1", "p2", "p3", "p4"]
    base_token_dists = dict.fromkeys(prompts, base_dist)
    ft_token_dists = dict.fromkeys(prompts, ft_dist)
    # Logprobs for the paraphrase case scoring.
    base_lp = {
        ("seed prompt", " gold"): -3.0,
        ("C1", " gold"): -3.0,
        ("C2", " gold"): -3.0,
        ("C3", " gold"): -3.0,
    }
    ft_lp = {
        ("seed prompt", " gold"): -1.0,  # lift +2
        ("C1", " gold"): -3.0,  # no lift → big gap
        ("C2", " gold"): -2.0,  # partial lift → medium gap
        ("C3", " gold"): -1.0,  # full lift → zero gap
    }
    return DummyDifferentialBackend(
        base=DummyResponses(token_dists=base_token_dists, logprobs=base_lp),
        ft=DummyResponses(token_dists=ft_token_dists, logprobs=ft_lp),
    )


@pytest.fixture
def stub_build_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``backends.build`` so the CLI doesn't try to load HF."""

    def _factory(*_args: object, **_kwargs: object) -> DummyDifferentialBackend:
        return _programmable_backend()

    import dlm_sway.backends as backends_mod

    monkeypatch.setattr(backends_mod, "build", _factory)


@pytest.fixture
def stub_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the MiniLM embedder so paraphrase mining doesn't need
    sentence-transformers installed."""
    table: dict[str, np.ndarray] = {
        "seed prompt": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "C1": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "C2": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "C3": np.array([0.5, 0.5, 0.0], dtype=np.float32),
    }

    def _encode(texts: list[str]) -> np.ndarray:
        return np.stack([table[t] for t in texts])

    monkeypatch.setattr(
        "dlm_sway.mining.paraphrase_miner._load_embedder",
        lambda _model_id: _encode,  # type: ignore[arg-type]
    )


@pytest.fixture
def stub_nlpaug(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the nlpaug generator so paraphrase mining doesn't need
    the nlpaug wheel."""

    def _gen(prompt: str, *, n: int, seed: int) -> list[str]:
        del prompt, n, seed
        return ["C1", "C2", "C3"]

    monkeypatch.setattr("dlm_sway.mining.paraphrase_miner.nlpaug_candidates", _gen)


def _write_paraphrase_spec(path: Path) -> None:
    path.write_text(
        """
version: 1
models:
  base: {base: stub, kind: hf, adapter: /tmp/stub}
  ft: {base: stub, kind: hf, adapter: /tmp/stub}
suite:
  - name: pi
    kind: paraphrase_invariance
    cases:
      - prompt: seed prompt
        gold: " gold"
        paraphrases: ["x"]
""".strip()
    )


def _write_delta_kl_spec(path: Path) -> None:
    path.write_text(
        """
version: 1
models:
  base: {base: stub, kind: hf, adapter: /tmp/stub}
  ft: {base: stub, kind: hf, adapter: /tmp/stub}
suite:
  - name: dk
    kind: delta_kl
    prompts: [p1, p2, p3, p4, p5, p6, p7, p8]
    assert_mean_gte: 0.0
""".strip()
    )


class TestMineParaphrase:
    def test_emits_yaml_fragment_with_mined_cases(
        self,
        stub_build_backend: None,  # noqa: ARG002
        stub_embedder: None,  # noqa: ARG002
        stub_nlpaug: None,  # noqa: ARG002
        tmp_path: Path,
    ) -> None:
        spec = tmp_path / "sway.yaml"
        _write_paraphrase_spec(spec)
        out = tmp_path / "mined.yaml"

        result = CliRunner().invoke(
            app,
            [
                "mine",
                str(spec),
                "--mode",
                "paraphrase",
                "--out",
                str(out),
                "--n-candidates",
                "3",
                "--top-k",
                "3",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert out.exists()
        payload: dict[str, Any] = yaml.safe_load(out.read_text())
        assert "mined_cases" in payload
        cases = payload["mined_cases"]
        assert len(cases) == 1
        case = cases[0]
        assert case["prompt"] == "seed prompt"
        assert case["gold"] == " gold"
        # Paraphrases are ranked hardest-first; C1 had the largest gap.
        assert case["paraphrases"][0] == "C1"
        assert "_mining_meta" in case


class TestMineOutliers:
    def test_emits_top_and_bottom(
        self,
        stub_build_backend: None,  # noqa: ARG002
        tmp_path: Path,
    ) -> None:
        spec = tmp_path / "sway.yaml"
        _write_delta_kl_spec(spec)
        out = tmp_path / "outliers.yaml"

        result = CliRunner().invoke(
            app,
            ["mine", str(spec), "--mode", "outliers", "--out", str(out), "--top-k", "3"],
        )
        assert result.exit_code == 0, result.stdout
        assert out.exists()
        payload: dict[str, Any] = yaml.safe_load(out.read_text())
        assert "mined_outliers" in payload
        rollup = payload["mined_outliers"]
        assert rollup["probe_kind"] == "delta_kl"
        assert isinstance(rollup["top"], list)
        assert isinstance(rollup["bottom"], list)
        # All 4 delta_kl prompts get scored; top-K is clipped to 3.
        assert len(rollup["top"]) == 3
        assert len(rollup["bottom"]) == 3

    def test_no_prompts_and_no_corpus_errors(
        self,
        stub_build_backend: None,  # noqa: ARG002
        tmp_path: Path,
    ) -> None:
        """A spec with no delta_kl prompts + no --from-corpus flag
        must exit non-zero with a pointed message rather than write
        an empty YAML."""
        spec = tmp_path / "sway.yaml"
        spec.write_text(
            """
version: 1
models:
  base: {base: stub, kind: hf, adapter: /tmp/stub}
  ft: {base: stub, kind: hf, adapter: /tmp/stub}
suite:
  - name: pi
    kind: paraphrase_invariance
    cases:
      - prompt: x
        gold: " y"
        paraphrases: ["z"]
""".strip()
        )
        out = tmp_path / "outliers.yaml"
        result = CliRunner().invoke(
            app, ["mine", str(spec), "--mode", "outliers", "--out", str(out)]
        )
        assert result.exit_code == 2
        assert not out.exists()
