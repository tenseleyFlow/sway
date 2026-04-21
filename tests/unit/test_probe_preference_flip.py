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


def test_warn_branch_score_formula_pinned() -> None:
    """C10: pin the WARN-branch numerical formula so a refactor notices.

    Formula: ``score = clip(0.5 + mean_delta / 4.0, 0, 1)`` where
    ``mean_delta`` is the average of ``(ft_margin - base_margin)`` over
    every triple. With deltas [+1.0, +0.5, +1.0] the mean is 0.8333…
    so the score is 0.5 + 0.8333.../4 = 0.70833….
    """
    import math

    backend = _backend(
        [
            ("p1", "good1", "bad1", 1.0, 2.0),  # delta=+1.0 (base right)
            ("p2", "good2", "bad2", 0.5, 1.0),  # delta=+0.5 (base right)
            ("p3", "good3", "bad3", -0.5, 0.5),  # delta=+1.0 (base wrong)
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
    expected_mean_delta = (1.0 + 0.5 + 1.0) / 3.0
    expected_score = 0.5 + expected_mean_delta / 4.0
    assert result.raw is not None
    assert math.isclose(result.raw, expected_mean_delta, rel_tol=1e-9)
    assert result.score is not None
    assert math.isclose(result.score, expected_score, rel_tol=1e-9)

    # Evidence mirrors the raw metric so report consumers can render it.
    assert math.isclose(result.evidence["mean_margin_delta"], expected_mean_delta, rel_tol=1e-9)
    assert result.evidence["base_wrong"] == 1
    assert result.evidence["total"] == 3


def test_one_bad_triple_does_not_kill_the_batch() -> None:
    """B14: a triple that raises ProbeError is dropped, not propagated.

    The remaining triples still produce a verdict; the dropped count
    surfaces in evidence so a user can see what got skipped.
    """
    from dlm_sway.core.errors import ProbeError

    backend = _backend(
        [
            ("p1", "good1", "bad1", -2.0, 2.0),
            ("p2", "good2", "bad2", -1.5, 1.0),
            ("p3", "good3", "bad3", -0.5, 0.8),
        ]
    )

    # Wrap the backend's logprob_of so the second triple raises.
    raising = {"p2"}
    original_as_base = backend.as_base
    original_as_finetuned = backend.as_finetuned

    def _raising_view(view_cm):
        from contextlib import contextmanager

        @contextmanager
        def _wrap():
            with view_cm() as view:
                orig = view.logprob_of

                def fenced(prompt, completion):
                    if prompt in raising:
                        raise ProbeError("logprob_of", f"simulated failure on {prompt!r}")
                    return orig(prompt, completion)

                view.logprob_of = fenced  # type: ignore[method-assign]
                yield view

        return _wrap

    backend.as_base = _raising_view(original_as_base)  # type: ignore[method-assign]
    backend.as_finetuned = _raising_view(original_as_finetuned)  # type: ignore[method-assign]

    triples = [
        {"prompt": p, "chosen": c, "rejected": r}
        for p, c, r in [("p1", "good1", "bad1"), ("p2", "good2", "bad2"), ("p3", "good3", "bad3")]
    ]
    probe, spec = build_probe(
        {
            "name": "pf",
            "kind": "preference_flip",
            "triples": triples,
            "assert_flip_rate_gte": 0.7,
            "min_triples_for_decision": 2,
        }
    )
    ctx = RunContext(backend=backend)
    result = probe.run(spec, ctx)

    assert result.verdict == Verdict.PASS  # the two surviving triples both flipped
    assert result.evidence["dropped_triples"] == 1
    assert any("p2" in reason for reason in result.evidence["dropped_reasons"])


def test_all_triples_failing_yields_error() -> None:
    """When every triple raises, the probe routes to ERROR with an explanation."""
    from contextlib import contextmanager

    from dlm_sway.core.errors import ProbeError

    backend = _backend([("p1", "g", "b", 0.0, 0.0)])
    inner_as_base = backend.as_base  # capture before monkeypatching

    @contextmanager
    def _always_raise():
        with inner_as_base() as view:

            def _raises(*_a, **_k):
                raise ProbeError("logprob_of", "always")

            view.logprob_of = _raises  # type: ignore[method-assign]
            yield view

    backend.as_base = _always_raise  # type: ignore[method-assign]
    backend.as_finetuned = _always_raise  # type: ignore[method-assign]

    probe, spec = build_probe(
        {
            "name": "pf",
            "kind": "preference_flip",
            "triples": [{"prompt": "p1", "chosen": "g", "rejected": "b"}],
        }
    )
    result = probe.run(spec, RunContext(backend=backend))
    assert result.verdict == Verdict.ERROR
    assert result.evidence["dropped_triples"] == 1


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
