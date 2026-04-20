"""Tests for :mod:`dlm_sway.probes.preference_flip`."""

from __future__ import annotations

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.result import Verdict
from dlm_sway.core.sections import Section, SectionPreference
from dlm_sway.probes.base import RunContext, build_probe


def _backend(pairs: list[tuple[str, str, str, float, float]]) -> DummyDifferentialBackend:
    """``pairs`` = list of (prompt, chosen, rejected, base_margin, ft_margin).

    We distribute the margin half to the chosen and half (negative) to
    the rejected, which is enough to make logprob_of(chosen)-logprob_of(rejected)
    equal the requested margin.
    """
    base_lp: dict[tuple[str, str], float] = {}
    ft_lp: dict[tuple[str, str], float] = {}
    for prompt, chosen, rejected, base_m, ft_m in pairs:
        base_lp[(prompt, chosen)] = base_m / 2
        base_lp[(prompt, rejected)] = -base_m / 2
        ft_lp[(prompt, chosen)] = ft_m / 2
        ft_lp[(prompt, rejected)] = -ft_m / 2
    return DummyDifferentialBackend(
        base=DummyResponses(logprobs=base_lp),
        ft=DummyResponses(logprobs=ft_lp),
    )


def test_pass_when_base_wrong_flipped() -> None:
    backend = _backend(
        [
            ("p1", "good1", "bad1", -2.0, 2.0),  # base wrong, ft flips
            ("p2", "good2", "bad2", -1.5, 1.0),  # base wrong, ft flips
            ("p3", "good3", "bad3", -0.5, 0.8),  # base wrong, ft flips
            ("p4", "good4", "bad4", 1.0, 2.0),  # base already right (no contribution)
        ]
    )
    triples = [
        {"prompt": p, "chosen": c, "rejected": r}
        for (p, c, r, _, _) in [
            ("p1", "good1", "bad1", 0, 0),
            ("p2", "good2", "bad2", 0, 0),
            ("p3", "good3", "bad3", 0, 0),
            ("p4", "good4", "bad4", 0, 0),
        ]
    ]
    probe, spec = build_probe(
        {
            "name": "pf",
            "kind": "preference_flip",
            "triples": triples,
            "assert_flip_rate_gte": 0.7,
            "min_triples_for_decision": 3,
        }
    )
    ctx = RunContext(backend=backend)
    result = probe.run(spec, ctx)
    assert result.verdict == Verdict.PASS
    assert result.raw == 1.0  # 3/3 flipped


def test_fail_when_base_wrong_not_flipped() -> None:
    backend = _backend(
        [
            ("p1", "good1", "bad1", -2.0, -1.5),  # base wrong, ft still wrong
            ("p2", "good2", "bad2", -1.5, -1.0),  # base wrong, ft still wrong
            ("p3", "good3", "bad3", -0.5, 0.8),  # base wrong, ft flips
        ]
    )
    triples = [
        {"prompt": p, "chosen": c, "rejected": r}
        for p, c, r in [
            ("p1", "good1", "bad1"),
            ("p2", "good2", "bad2"),
            ("p3", "good3", "bad3"),
        ]
    ]
    probe, spec = build_probe(
        {
            "name": "pf",
            "kind": "preference_flip",
            "triples": triples,
            "assert_flip_rate_gte": 0.7,
            "min_triples_for_decision": 3,
        }
    )
    ctx = RunContext(backend=backend)
    result = probe.run(spec, ctx)
    assert result.verdict == Verdict.FAIL
    assert result.raw is not None
    assert result.raw < 0.7


def test_skip_when_no_triples_anywhere() -> None:
    probe, spec = build_probe({"name": "pf", "kind": "preference_flip"})
    backend = _backend([])
    ctx = RunContext(backend=backend)
    result = probe.run(spec, ctx)
    assert result.verdict == Verdict.SKIP


def test_warn_when_too_few_base_wrong() -> None:
    backend = _backend(
        [
            ("p1", "good1", "bad1", 1.0, 2.0),  # base right
            ("p2", "good2", "bad2", 0.5, 1.0),  # base right
            ("p3", "good3", "bad3", -0.5, 0.5),  # base wrong
        ]
    )
    triples = [
        {"prompt": p, "chosen": c, "rejected": r}
        for p, c, r in [
            ("p1", "good1", "bad1"),
            ("p2", "good2", "bad2"),
            ("p3", "good3", "bad3"),
        ]
    ]
    probe, spec = build_probe(
        {
            "name": "pf",
            "kind": "preference_flip",
            "triples": triples,
            "min_triples_for_decision": 3,
        }
    )
    ctx = RunContext(backend=backend)
    result = probe.run(spec, ctx)
    assert result.verdict == Verdict.WARN


def test_triples_pulled_from_sections() -> None:
    pref_section = Section(
        id="p1",
        kind="preference",
        content="...",
        preferences=(
            SectionPreference(prompt="q1", chosen="good", rejected="bad"),
            SectionPreference(prompt="q2", chosen="good2", rejected="bad2"),
            SectionPreference(prompt="q3", chosen="good3", rejected="bad3"),
        ),
    )
    backend = _backend(
        [
            ("q1", "good", "bad", -1.0, 1.0),
            ("q2", "good2", "bad2", -1.0, 1.0),
            ("q3", "good3", "bad3", -1.0, 1.0),
        ]
    )
    probe, spec = build_probe(
        {
            "name": "pf",
            "kind": "preference_flip",
            "assert_flip_rate_gte": 0.7,
            "min_triples_for_decision": 3,
        }
    )
    ctx = RunContext(backend=backend, sections=(pref_section,))
    result = probe.run(spec, ctx)
    assert result.verdict == Verdict.PASS
