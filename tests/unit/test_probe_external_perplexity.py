"""Tests for :mod:`dlm_sway.probes.external_perplexity`."""

from __future__ import annotations

import numpy as np
import pytest

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.result import Verdict
from dlm_sway.core.scoring import RollingLogprob
from dlm_sway.probes._external_corpus import (
    available_corpora,
    chunk_corpus,
    load_corpus,
)
from dlm_sway.probes.base import RunContext, build_probe
from dlm_sway.probes.external_perplexity import ExternalPerplexitySpec


def _rolling(text: str, per_tok: float) -> RollingLogprob:
    """Build a uniform rolling logprob whose per-token mean is ``per_tok``."""
    tokens = text.split()
    n = max(len(tokens), 1)
    lp = np.full(max(n - 1, 0), per_tok, dtype=np.float32)
    return RollingLogprob(
        token_ids=np.arange(n, dtype=np.int64),
        logprobs=lp,
        num_tokens=n,
        total_logprob=float(per_tok * max(n - 1, 0)),
    )


def _backend_with_delta(base_per_tok: float, ft_per_tok: float) -> DummyDifferentialBackend:
    """Return a dummy backend where ft - base = (ft_per_tok - base_per_tok) per token.

    The canned rolling map is keyed by raw chunk text, so the probe's
    first-64-chunk slice of the corpus doesn't matter — every chunk
    falls back to the synthesized default (``_compute_rolling_logprob``
    uses mode defaults of -2.0 / -1.5). We override both maps with one
    entry per chunk by preloading the corpus and pre-computing chunk
    texts.
    """
    corpus = load_corpus("public_domain_en")
    chunks = chunk_corpus(corpus, chunk_chars=2048, max_chunks=16)
    base_map = {c: _rolling(c, base_per_tok) for c in chunks}
    ft_map = {c: _rolling(c, ft_per_tok) for c in chunks}
    return DummyDifferentialBackend(
        base=DummyResponses(rolling=base_map),
        ft=DummyResponses(rolling=ft_map),
    )


class TestCorpusLoader:
    def test_public_domain_en_is_available(self) -> None:
        assert "public_domain_en" in available_corpora()

    def test_load_corpus_strips_comments(self) -> None:
        text = load_corpus("public_domain_en")
        # The raw file has `# -- source:` provenance lines; those must
        # not survive into the probe-facing string.
        assert "# --" not in text
        assert text.strip(), "loaded corpus should not be empty"

    def test_load_corpus_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            load_corpus("not_a_real_corpus")

    def test_chunk_corpus_respects_caps(self) -> None:
        text = "A" * 10_000
        chunks = chunk_corpus(text, chunk_chars=1024, max_chunks=4)
        assert len(chunks) == 4
        assert all(len(c) == 1024 for c in chunks)

    def test_chunk_corpus_drops_short_tail(self) -> None:
        # 2100 chars at chunk_chars=1024 → two full chunks + 52-char tail
        # (below the 64-char floor, so it's dropped).
        text = "A" * 2100
        chunks = chunk_corpus(text, chunk_chars=1024, max_chunks=16)
        assert len(chunks) == 2

    def test_chunk_corpus_keeps_long_tail(self) -> None:
        # 2200 chars at chunk_chars=1024 → two full chunks + 152-char tail
        # (above the 64-char floor, kept).
        text = "A" * 2200
        chunks = chunk_corpus(text, chunk_chars=1024, max_chunks=16)
        assert len(chunks) == 3

    @pytest.mark.parametrize("bad", [0, -1])
    def test_chunk_corpus_rejects_nonpositive_chunk_chars(self, bad: int) -> None:
        with pytest.raises(ValueError):
            chunk_corpus("hello world", chunk_chars=bad, max_chunks=4)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_chunk_corpus_rejects_nonpositive_max_chunks(self, bad: int) -> None:
        with pytest.raises(ValueError):
            chunk_corpus("hello world", chunk_chars=1024, max_chunks=bad)


class TestExternalPerplexityProbe:
    def test_pass_when_ft_matches_base(self) -> None:
        """No perplexity shift → mean_delta ≈ 0 → fixed-threshold PASS."""
        backend = _backend_with_delta(base_per_tok=-2.0, ft_per_tok=-2.0)
        probe, spec = build_probe(
            {"name": "ext_ppl", "kind": "external_perplexity", "max_chunks": 4}
        )
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.PASS
        assert result.raw is not None
        assert abs(result.raw) < 1e-6
        # No null stats yet — message should carry the no-calibration note.
        assert "no calibration" in (result.message or "").lower()

    def test_pass_when_ft_improves_base(self) -> None:
        """ft assigns higher logprobs → mean_delta > 0 → PASS."""
        backend = _backend_with_delta(base_per_tok=-2.0, ft_per_tok=-1.5)
        probe, spec = build_probe(
            {"name": "ext_ppl", "kind": "external_perplexity", "max_chunks": 4}
        )
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.PASS
        assert result.raw is not None
        assert result.raw > 0

    def test_fail_on_large_regression(self) -> None:
        """ft raised perplexity by >0.1 nats/tok → fixed-threshold FAIL."""
        backend = _backend_with_delta(base_per_tok=-2.0, ft_per_tok=-2.5)
        probe, spec = build_probe(
            {"name": "ext_ppl", "kind": "external_perplexity", "max_chunks": 4}
        )
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.FAIL
        assert result.raw is not None
        assert result.raw < -0.1

    def test_evidence_carries_per_chunk_deltas(self) -> None:
        backend = _backend_with_delta(base_per_tok=-2.0, ft_per_tok=-1.9)
        probe, spec = build_probe(
            {"name": "ext_ppl", "kind": "external_perplexity", "max_chunks": 3}
        )
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        ev = result.evidence
        assert ev["corpus"] == "public_domain_en"
        assert ev["num_chunks"] == 3
        assert len(ev["per_chunk_delta"]) == 3
        # Every chunk carries the same 0.1 nats/tok improvement by
        # construction of the canned rolling maps.
        for d in ev["per_chunk_delta"]:
            assert abs(d - 0.1) < 1e-5

    def test_unknown_corpus_errors(self) -> None:
        """Spec validation accepts only declared corpora; bypassing it
        via direct construction yields a clean ERROR."""
        backend = _backend_with_delta(base_per_tok=-2.0, ft_per_tok=-2.0)
        spec = ExternalPerplexitySpec.model_construct(
            name="ext",
            kind="external_perplexity",
            corpus="does_not_exist",  # type: ignore[arg-type]
        )
        probe, _ = build_probe({"name": "ext_ppl", "kind": "external_perplexity", "max_chunks": 4})
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.ERROR
        assert "unknown corpus" in (result.message or "")

    def test_respects_max_chunks(self) -> None:
        backend = _backend_with_delta(base_per_tok=-2.0, ft_per_tok=-2.0)
        probe, spec = build_probe(
            {"name": "ext_ppl", "kind": "external_perplexity", "max_chunks": 2}
        )
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        assert result.evidence["num_chunks"] == 2


class TestCalibrateSpec:
    def test_returns_non_none_spec(self) -> None:
        from dlm_sway.probes.external_perplexity import ExternalPerplexityProbe

        backend = _backend_with_delta(base_per_tok=-2.0, ft_per_tok=-2.0)
        ctx = RunContext(backend=backend)
        cal_spec = ExternalPerplexityProbe.calibrate_spec(ctx)
        assert cal_spec is not None
        assert cal_spec.kind == "external_perplexity"
        # Cheap calibration uses ≤4 chunks so null calibration stays fast.
        assert cal_spec.max_chunks <= 4


class TestNullCalibrationEndToEnd:
    def test_runner_threads_null_stats_to_external_perplexity(self) -> None:
        """null_adapter → external_perplexity gets a z_score in the suite."""
        from dlm_sway.suite.runner import run as run_suite
        from dlm_sway.suite.spec import SwaySpec

        backend = _backend_with_delta(base_per_tok=-2.0, ft_per_tok=-1.9)
        raw_spec = SwaySpec.model_validate(
            {
                "version": 1,
                "models": {
                    "base": {"base": "b"},
                    "ft": {"base": "b", "adapter": "/tmp/a"},
                },
                "suite": [
                    {"name": "null", "kind": "null_adapter", "runs": 2, "cache": False},
                    {
                        "name": "ext",
                        "kind": "external_perplexity",
                        "max_chunks": 3,
                        "assert_z_gte": -100.0,  # permissive
                    },
                ],
            }
        )
        result = run_suite(raw_spec, backend)
        assert len(result.probes) == 2
        null_result = result.probes[0]
        ext_result = result.probes[1]
        assert null_result.verdict == Verdict.PASS
        # External-perplexity should have taken the z-score path because
        # null_adapter populated per-kind stats for it.
        assert ext_result.z_score is not None, (
            "external_perplexity should have z-scored against null baseline, "
            f"got evidence={ext_result.evidence}, message={ext_result.message}"
        )
        # Sign-flipped z path puts "lower-is-better" wording in the message.
        assert "lower-is-better" in (ext_result.message or "")
