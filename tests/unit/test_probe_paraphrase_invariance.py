"""Tests for :mod:`dlm_sway.probes.paraphrase_invariance`."""

from __future__ import annotations

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.result import Verdict
from dlm_sway.probes.base import RunContext, build_probe


def _backend(*, par_lift_fraction: float, verb_lift: float = 10.0) -> DummyDifferentialBackend:
    """Return a backend with tunable verbatim/paraphrase lifts.

    The ft view adds ``verb_lift`` nats to the verbatim (Q,A) logprob
    and ``par_lift_fraction * verb_lift`` to paraphrase logprobs.
    """
    base = DummyResponses(
        logprobs={
            ("Q", "A"): -20.0,
            ("Q_par1", "A"): -20.0,
            ("Q_par2", "A"): -20.0,
        }
    )
    ft = DummyResponses(
        logprobs={
            ("Q", "A"): -20.0 + verb_lift,
            ("Q_par1", "A"): -20.0 + par_lift_fraction * verb_lift,
            ("Q_par2", "A"): -20.0 + par_lift_fraction * verb_lift,
        }
    )
    return DummyDifferentialBackend(base=base, ft=ft)


def test_pass_when_generalizing() -> None:
    # High paraphrase lift + high verbatim → healthy generalization.
    backend = _backend(par_lift_fraction=0.9)
    probe, spec = build_probe(
        {
            "name": "pi",
            "kind": "paraphrase_invariance",
            "intent": "generalize",
            "min_verbatim_lift": 0.05,
            "min_generalization_ratio": 0.5,
            "cases": [{"prompt": "Q", "gold": "A", "paraphrases": ["Q_par1", "Q_par2"]}],
        }
    )
    ctx = RunContext(backend=backend)
    result = probe.run(spec, ctx)
    assert result.verdict == Verdict.PASS
    assert result.raw is not None
    assert result.raw >= 0.5


def test_fails_when_only_memorized_but_intent_generalize() -> None:
    backend = _backend(par_lift_fraction=0.0)
    probe, spec = build_probe(
        {
            "name": "pi",
            "kind": "paraphrase_invariance",
            "intent": "generalize",
            "min_verbatim_lift": 0.05,
            "cases": [{"prompt": "Q", "gold": "A", "paraphrases": ["Q_par1"]}],
        }
    )
    ctx = RunContext(backend=backend)
    result = probe.run(spec, ctx)
    assert result.verdict == Verdict.FAIL


def test_passes_memorize_intent_when_only_memorized() -> None:
    backend = _backend(par_lift_fraction=0.0)
    probe, spec = build_probe(
        {
            "name": "pi",
            "kind": "paraphrase_invariance",
            "intent": "memorize",
            "min_verbatim_lift": 0.05,
            "max_generalization_ratio_if_memorize": 0.3,
            "cases": [{"prompt": "Q", "gold": "A", "paraphrases": ["Q_par1"]}],
        }
    )
    ctx = RunContext(backend=backend)
    result = probe.run(spec, ctx)
    assert result.verdict == Verdict.PASS


def test_error_on_empty_cases() -> None:
    probe, spec = build_probe({"name": "pi", "kind": "paraphrase_invariance", "cases": []})
    backend = _backend(par_lift_fraction=0.9)
    ctx = RunContext(backend=backend)
    result = probe.run(spec, ctx)
    assert result.verdict == Verdict.ERROR
