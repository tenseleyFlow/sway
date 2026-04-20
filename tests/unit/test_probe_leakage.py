"""Tests for :mod:`dlm_sway.probes.leakage`."""

from __future__ import annotations

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.result import Verdict
from dlm_sway.core.sections import Section
from dlm_sway.probes.base import RunContext, build_probe
from dlm_sway.probes.leakage import _fragility, _lcs_ratio, _perturb


class TestLCS:
    def test_identical_returns_one(self) -> None:
        assert _lcs_ratio("abcdef", "abcdef") == 1.0

    def test_disjoint_returns_low(self) -> None:
        assert _lcs_ratio("abc", "xyz") < 0.3

    def test_empty_returns_zero(self) -> None:
        assert _lcs_ratio("", "abc") == 0.0


class TestPerturb:
    def test_typo_swaps_first_two(self) -> None:
        assert _perturb("hello", "typo") == "ehllo"

    def test_case_flip_inverts_first_alpha(self) -> None:
        assert _perturb("abc", "case_flip") == "Abc"
        assert _perturb("ABC", "case_flip") == "aBC"

    def test_drop_punct_removes_punct(self) -> None:
        assert _perturb("a, b. c!", "drop_punct") == "a b c"


class TestFragility:
    def test_zero_when_clean_zero(self) -> None:
        assert _fragility(0.0, 0.0) == 0.0

    def test_expected_when_perturbed_dropped(self) -> None:
        import pytest as _pt

        assert _fragility(0.8, 0.2) == _pt.approx(0.75)


def _prose_section(sid: str, content: str) -> Section:
    return Section(id=sid, kind="prose", content=content)


def _backend(*, ft_recall: float, ft_perturbed_recall: float) -> DummyDifferentialBackend:
    """Build a backend whose ft generate() returns a controlled prefix of ``target``.

    The target is "aaa..." (200 chars) so we can measure LCS ratio
    against it deterministically.
    """
    content = ("The capital of France is Paris. " * 30).strip()
    # Generate a fraction of the target to hit the desired recall.
    target = content[128 : 128 + 256]
    ft_full = target[: int(ft_recall * len(target))]
    ft_pert = target[: int(ft_perturbed_recall * len(target))]

    base = DummyResponses()
    ft = DummyResponses(
        generations={
            content[:128]: ft_full,
            # perturbations of the first 128 chars hit these three:
            **{_perturb(content[:128], p): ft_pert for p in ("typo", "case_flip", "drop_punct")},
        }
    )
    return DummyDifferentialBackend(base=base, ft=ft), content


class TestProbe:
    def test_skip_without_sections(self) -> None:
        backend, _ = _backend(ft_recall=0.0, ft_perturbed_recall=0.0)
        probe, spec = build_probe({"name": "c3", "kind": "leakage"})
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.SKIP

    def test_pass_when_no_leak(self) -> None:
        backend, content = _backend(ft_recall=0.0, ft_perturbed_recall=0.0)
        probe, spec = build_probe(
            {
                "name": "c3",
                "kind": "leakage",
                "prefix_chars": 128,
                "continuation_chars": 256,
            }
        )
        ctx = RunContext(backend=backend, sections=(_prose_section("a", content),))
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.PASS

    def test_fail_when_strong_low_fragility_leak(self) -> None:
        backend, content = _backend(ft_recall=0.95, ft_perturbed_recall=0.9)
        probe, spec = build_probe(
            {
                "name": "c3",
                "kind": "leakage",
                "prefix_chars": 128,
                "continuation_chars": 256,
                "assert_recall_lt": 0.5,
                "min_fragility": 0.3,
            }
        )
        ctx = RunContext(backend=backend, sections=(_prose_section("a", content),))
        result = probe.run(spec, ctx)
        # High recall + low fragility → fail.
        assert result.verdict == Verdict.FAIL
