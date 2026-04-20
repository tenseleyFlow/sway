"""Tests for :mod:`dlm_sway.probes.style_fingerprint`."""

from __future__ import annotations

import numpy as np

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.result import Verdict
from dlm_sway.probes.base import RunContext, build_probe
from dlm_sway.probes.style_fingerprint import fingerprint


class TestFingerprint:
    def test_zero_vector_for_empty(self) -> None:
        fp = fingerprint("")
        assert fp.shape == (6,)
        assert np.allclose(fp, 0.0)

    def test_non_zero_for_normal_text(self) -> None:
        fp = fingerprint("This is a sentence. This is another one. A third.")
        assert fp.shape == (6,)
        assert fp[0] > 0  # mean sentence length
        assert fp[2] > 0  # TTR
        assert fp[3] > 0  # avg word length

    def test_distinct_styles_distinct_fingerprints(self) -> None:
        terse = "Go. Now. Quick."
        verbose = (
            "We must, with all deliberate speed and measured consideration, "
            "proceed expeditiously towards the elaborated and carefully "
            "constructed resolution of the foregoing matter."
        )
        assert not np.allclose(fingerprint(terse), fingerprint(verbose))


def _backend_with_samples(base: list[str], ft: list[str]) -> DummyDifferentialBackend:
    return DummyDifferentialBackend(
        base=DummyResponses(generations={f"p{i}": s for i, s in enumerate(base)}),
        ft=DummyResponses(generations={f"p{i}": s for i, s in enumerate(ft)}),
    )


class TestProbe:
    def test_pass_when_ft_drifts_toward_doc(self) -> None:
        base_samples = ["Short. Plain. Words."] * 2
        ft_samples = [
            "Wherein many clauses conjoin themselves, through extended "
            "ruminations, unto a meandering whole of considerable length."
        ] * 2
        doc = (
            "Wherein many clauses conjoin themselves, through extended "
            "ruminations, unto a meandering whole of considerable length. "
            "Further elaboration, no less copious, follows apace."
        )
        backend = _backend_with_samples(base_samples, ft_samples)
        probe, spec = build_probe(
            {
                "name": "c1",
                "kind": "style_fingerprint",
                "prompts": ["p0", "p1"],
                "doc_reference": doc,
                "max_new_tokens": 32,
                "assert_shift_gte": 0.2,
            }
        )
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.PASS
        assert result.raw is not None
        assert result.raw > 0.2

    def test_fail_when_no_stylistic_shift(self) -> None:
        base_samples = ["Short. Plain. Words."] * 2
        ft_samples = ["Short. Plain. Words."] * 2
        doc = "Wherein clauses conjoin into meandering wholes of length."
        backend = _backend_with_samples(base_samples, ft_samples)
        probe, spec = build_probe(
            {
                "name": "c1",
                "kind": "style_fingerprint",
                "prompts": ["p0", "p1"],
                "doc_reference": doc,
                "assert_shift_gte": 0.25,
            }
        )
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.FAIL

    def test_skip_without_doc_reference(self) -> None:
        backend = _backend_with_samples(["x"], ["y"])
        probe, spec = build_probe(
            {
                "name": "c1",
                "kind": "style_fingerprint",
                "prompts": ["p0"],
            }
        )
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.SKIP

    def test_error_on_empty_prompts(self) -> None:
        backend = _backend_with_samples([], [])
        probe, spec = build_probe(
            {
                "name": "c1",
                "kind": "style_fingerprint",
                "prompts": [],
                "doc_reference": "doc",
            }
        )
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.ERROR
