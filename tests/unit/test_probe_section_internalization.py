"""Tests for :mod:`dlm_sway.probes.section_internalization` (the flagship B1)."""

from __future__ import annotations

import numpy as np

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.result import Verdict
from dlm_sway.core.scoring import RollingLogprob
from dlm_sway.core.sections import Section, SectionProbe
from dlm_sway.probes.base import RunContext, build_probe


def _rolling(mean_lp: float, n: int = 10) -> RollingLogprob:
    lp = np.full(n - 1, mean_lp, dtype=np.float32)
    return RollingLogprob(
        token_ids=np.arange(n, dtype=np.int64),
        logprobs=lp,
        num_tokens=n,
        total_logprob=float(lp.sum()),
    )


def _section(sid: str, kind: str = "prose", content: str = "content", probes=()) -> Section:
    return Section(id=sid, kind=kind, content=content, probes=tuple(probes))  # type: ignore[arg-type]


def test_skip_without_sections() -> None:
    probe, spec = build_probe({"name": "sis", "kind": "section_internalization"})
    backend = DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
    ctx = RunContext(backend=backend)
    result = probe.run(spec, ctx)
    assert result.verdict == Verdict.SKIP


def test_skip_with_single_section() -> None:
    probe, spec = build_probe({"name": "sis", "kind": "section_internalization"})
    backend = DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
    ctx = RunContext(backend=backend, sections=(_section("a"),))
    result = probe.run(spec, ctx)
    assert result.verdict == Verdict.SKIP


def test_pass_when_each_section_gets_distinct_lift() -> None:
    # Build a dummy backend where the ft is much lower-PPL than base on
    # every section's content — uniform lift, but leak-check math
    # yields ~zero differential leak so all sections pass.
    content_a = "aaa " * 10
    content_b = "bbb " * 10

    base = DummyResponses(rolling={content_a: _rolling(-3.0), content_b: _rolling(-3.0)})
    ft = DummyResponses(rolling={content_a: _rolling(-1.0), content_b: _rolling(-2.5)})
    backend = DummyDifferentialBackend(base=base, ft=ft)

    sections = (
        _section("a", content=content_a),
        _section("b", content=content_b),
    )
    probe, spec = build_probe(
        {
            "name": "sis",
            "kind": "section_internalization",
            "per_section_threshold": 0.05,
        }
    )
    ctx = RunContext(backend=backend, sections=sections)
    result = probe.run(spec, ctx)
    assert result.verdict in (Verdict.PASS, Verdict.FAIL)
    assert "per_section" in result.evidence
    assert len(result.evidence["per_section"]) == 2


def test_instruction_uses_logprob_of() -> None:
    # Instruction sections contribute their probe Q/A pairs; feed
    # logprobs so the ft view comes out cheaper than base.
    probes_a = (SectionProbe(prompt="Qa", gold="Aa"),)
    probes_b = (SectionProbe(prompt="Qb", gold="Ab"),)
    base = DummyResponses(logprobs={("Qa", "Aa"): -10.0, ("Qb", "Ab"): -10.0})
    ft = DummyResponses(logprobs={("Qa", "Aa"): -3.0, ("Qb", "Ab"): -8.0})
    backend = DummyDifferentialBackend(base=base, ft=ft)

    sections = (
        _section("a", kind="instruction", content="...", probes=probes_a),
        _section("b", kind="instruction", content="...", probes=probes_b),
    )
    probe, spec = build_probe(
        {"name": "sis", "kind": "section_internalization", "per_section_threshold": 0.05}
    )
    ctx = RunContext(backend=backend, sections=sections)
    result = probe.run(spec, ctx)
    per = result.evidence["per_section"]
    # Section A got much more lift than B, so effective_sis(a) > effective_sis(b).
    sis_by_id = {row["section_id"]: row["effective_sis"] for row in per}
    assert sis_by_id["a"] > sis_by_id["b"]
