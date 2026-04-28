"""Tests for :mod:`dlm_sway.probes.multi_turn_coherence`."""

from __future__ import annotations

import math

import numpy as np

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.result import Verdict
from dlm_sway.core.scoring import TokenDist
from dlm_sway.probes.base import RunContext, build_probe
from dlm_sway.probes.multi_turn_coherence import (
    _ascii_sparkline,
    _chat_template_of,
    _fallback_format,
    _fit_half_life_turns,
    _verdict_from_half_life,
)

# ---------------------------------------------------------------------------
# Fake tokenizer + dist-injection helpers
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal stand-in for an HF tokenizer with a chat_template.

    Implements just enough of the surface :mod:`multi_turn_coherence`
    consults: the ``chat_template`` attribute and an
    ``apply_chat_template`` method that returns a concatenated
    role-marker string. Real HF chat templates are Jinja-rendered;
    ours is a deterministic string concat so per-turn comparisons
    are reproducible.
    """

    chat_template = "fake-jinja-template-string"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        del tokenize  # we always return text
        parts = [f"{m['role']}::{m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("assistant::")
        return "|".join(parts)


def _attach_tokenizer(
    backend: DummyDifferentialBackend, tokenizer: object | None
) -> DummyDifferentialBackend:
    """Slot a tokenizer onto the dummy backend the same way HF does.

    multi_turn_coherence reads ``ctx.backend._tokenizer`` to find the
    chat template — mirrors prompt_collapse's _peek_backend_tokenizer.
    """
    backend._tokenizer = tokenizer  # type: ignore[attr-defined]
    return backend


def _decay_dist(value: float, *, broad: bool, k: int = 8) -> TokenDist:
    """Build a TokenDist whose top-k logprobs encode a known signal.

    ``value`` controls how peaked the distribution is — feeding two
    such dists into the divergence helper produces a deterministic
    KL we can plant per-turn for the curve-fit tests.
    """
    if broad:
        # Roughly uniform with tiny perturbation (clears the
        # _UNIFORM_LOGPROB_TOL guard).
        lp = np.full(k, -math.log(k), dtype=np.float32)
        lp += np.linspace(-1e-4, 1e-4, k, dtype=np.float32)
    else:
        # Sharp: most mass on first token, with `value` controlling
        # how peaked. Larger value ⇒ sharper.
        lp = np.array([-0.01 * value] + [-1.0 - value] * (k - 1), dtype=np.float32)
    return TokenDist(
        token_ids=np.arange(k, dtype=np.int64),
        logprobs=lp,
        vocab_size=1000,
        tail_logprob=None,
    )


# ---------------------------------------------------------------------------
# Skip / error paths
# ---------------------------------------------------------------------------


class TestSkipPaths:
    def test_skips_when_no_tokenizer(self) -> None:
        """Dummy backend has no tokenizer ⇒ probe SKIPs cleanly."""
        backend = DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
        probe, spec = build_probe(
            {
                "name": "mtc",
                "kind": "multi_turn_coherence_decay",
                "prompts": ["hello"],
            }
        )
        result = probe.run(spec, RunContext(backend=backend))
        assert result.verdict == Verdict.SKIP
        assert "chat_template" in result.message

    def test_skips_when_tokenizer_lacks_chat_template(self) -> None:
        backend = DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())

        class _BareTokenizer:
            chat_template = None

        _attach_tokenizer(backend, _BareTokenizer())
        probe, spec = build_probe(
            {
                "name": "mtc",
                "kind": "multi_turn_coherence_decay",
                "prompts": ["hello"],
            }
        )
        result = probe.run(spec, RunContext(backend=backend))
        assert result.verdict == Verdict.SKIP

    def test_errors_when_no_prompts(self) -> None:
        backend = DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
        _attach_tokenizer(backend, _FakeTokenizer())
        probe, spec = build_probe({"name": "mtc", "kind": "multi_turn_coherence_decay"})
        result = probe.run(spec, RunContext(backend=backend))
        assert result.verdict == Verdict.ERROR


# ---------------------------------------------------------------------------
# Happy path: planted decay → curve fit recovers half-life
# ---------------------------------------------------------------------------


class TestHappyPath:
    """End-to-end probe runs against the dummy backend.

    Tests verify *wiring* — the probe correctly drives the chat loop,
    asks for next-token-dists at the right positions, and produces a
    finalized result with the documented evidence shape. The exact
    KL values depend on the dummy backend's synthetic-distribution
    math; tests assert the curve is monotonic / non-monotonic / flat
    as intended rather than pinning specific KL targets. Curve-fit
    correctness is covered separately in :class:`TestFitHalfLife`.
    """

    def test_decreasing_curve_yields_finite_half_life(self) -> None:
        """Decreasing planted KLs → 'ok' fit status, finite half-life."""
        backend = _backend_with_decreasing_dists(
            prompt="hello", max_turns=4, sharpness_per_turn=[3.0, 1.0, 0.3]
        )
        probe, spec = build_probe(
            {
                "name": "mtc",
                "kind": "multi_turn_coherence_decay",
                "prompts": ["hello"],
                "max_turns": 4,
                "assert_half_life_turns": 0.5,
            }
        )
        result = probe.run(spec, RunContext(backend=backend))
        assert result.verdict in {Verdict.PASS, Verdict.FAIL}, result.message
        assert result.evidence["fit_status"] in {"ok", "stable"}
        per_turn = result.evidence["per_turn_kls"]
        assert len(per_turn) == 3
        # Monotonic decrease — verifies the probe wired turns in order.
        assert per_turn[0] > per_turn[1] > per_turn[2]
        assert result.raw is not None
        assert result.raw > 0.0

    def test_flat_curve_marked_stable(self) -> None:
        """Same dists at every turn → fit_status=stable → PASS."""
        backend = _backend_with_decreasing_dists(
            prompt="hello", max_turns=4, sharpness_per_turn=[1.0, 1.0, 1.0]
        )
        probe, spec = build_probe(
            {
                "name": "mtc",
                "kind": "multi_turn_coherence_decay",
                "prompts": ["hello"],
                "max_turns": 4,
            }
        )
        result = probe.run(spec, RunContext(backend=backend))
        assert result.verdict == Verdict.PASS
        assert result.evidence["fit_status"] == "stable"
        assert result.score == 1.0
        assert "held coherence" in result.message

    def test_growing_curve_warns(self) -> None:
        """Increasing planted KLs → fit_status=non_monotonic → WARN."""
        backend = _backend_with_decreasing_dists(
            prompt="hello", max_turns=4, sharpness_per_turn=[0.3, 1.0, 3.0]
        )
        probe, spec = build_probe(
            {
                "name": "mtc",
                "kind": "multi_turn_coherence_decay",
                "prompts": ["hello"],
                "max_turns": 4,
            }
        )
        result = probe.run(spec, RunContext(backend=backend))
        assert result.verdict == Verdict.WARN
        assert result.evidence["fit_status"] == "non_monotonic"

    def test_evidence_carries_turns_and_sparkline(self) -> None:
        backend = _backend_with_decreasing_dists(
            prompt="hello", max_turns=4, sharpness_per_turn=[2.0, 1.0, 0.5]
        )
        probe, spec = build_probe(
            {
                "name": "mtc",
                "kind": "multi_turn_coherence_decay",
                "prompts": ["hello"],
                "max_turns": 4,
            }
        )
        result = probe.run(spec, RunContext(backend=backend))
        assert result.evidence["turns_axis"] == [2.0, 3.0, 4.0]
        assert result.evidence["max_turns"] == 4
        assert result.evidence["num_prompts"] == 1
        assert isinstance(result.evidence["sparkline"], str)
        assert len(result.evidence["sparkline"]) == 3


# ---------------------------------------------------------------------------
# Curve fit unit tests (math only, no backend)
# ---------------------------------------------------------------------------


class TestFitHalfLife:
    def test_clean_exponential_recovers_half_life(self) -> None:
        # y = exp(-0.693 * x) ⇒ half-life = 1 turn
        turns = np.array([2.0, 3.0, 4.0])
        kls = np.exp(-math.log(2.0) * turns)
        h, status = _fit_half_life_turns(turns, kls, max_turns=4)
        assert status == "ok"
        assert h is not None
        assert math.isclose(h, 1.0, rel_tol=1e-3)

    def test_stable_returns_saturation(self) -> None:
        turns = np.array([2.0, 3.0, 4.0])
        kls = np.array([0.5, 0.5, 0.5])
        h, status = _fit_half_life_turns(turns, kls, max_turns=4)
        assert status == "stable"
        assert h == 40.0  # max_turns * 10

    def test_growing_returns_non_monotonic(self) -> None:
        turns = np.array([2.0, 3.0, 4.0])
        kls = np.array([0.1, 0.3, 0.9])
        h, status = _fit_half_life_turns(turns, kls, max_turns=4)
        assert status == "non_monotonic"
        assert h is None

    def test_all_zero_returns_degenerate(self) -> None:
        turns = np.array([2.0, 3.0, 4.0])
        kls = np.array([0.0, 0.0, 0.0])
        h, status = _fit_half_life_turns(turns, kls, max_turns=4)
        assert status == "degenerate"
        assert h == 0.0

    def test_partial_zero_drops_zero_points(self) -> None:
        """One zero-KL turn: drop it, fit on the remaining positives."""
        turns = np.array([2.0, 3.0, 4.0])
        kls = np.array([0.4, 0.0, 0.1])
        h, status = _fit_half_life_turns(turns, kls, max_turns=4)
        assert status == "ok"
        assert h is not None
        assert h > 0.0


# ---------------------------------------------------------------------------
# Verdict mapping (math-free)
# ---------------------------------------------------------------------------


class TestVerdictMapping:
    def test_pass_on_half_life_above_target(self) -> None:
        v, s, msg = _verdict_from_half_life(
            half_life=3.0,
            fit_status="ok",
            target=2.0,
            mean_kls=[0.4, 0.2, 0.1],
            turns_axis=[2.0, 3.0, 4.0],
        )
        assert v == Verdict.PASS
        assert s == 1.0
        assert "half-life=3.00" in msg

    def test_fail_on_half_life_below_target(self) -> None:
        v, s, msg = _verdict_from_half_life(
            half_life=0.5,
            fit_status="ok",
            target=2.0,
            mean_kls=[0.4, 0.1, 0.02],
            turns_axis=[2.0, 3.0, 4.0],
        )
        assert v == Verdict.FAIL
        assert 0.0 < s < 1.0


# ---------------------------------------------------------------------------
# Sparkline + fallback formatter
# ---------------------------------------------------------------------------


class TestSparkline:
    def test_renders_one_char_per_value(self) -> None:
        out = _ascii_sparkline([0.4, 0.2, 0.1])
        assert len(out) == 3
        # Decreasing input ⇒ first bar should be the tallest
        assert out[0] >= out[-1]

    def test_flat_input_renders_uniform_mid(self) -> None:
        out = _ascii_sparkline([0.5, 0.5, 0.5])
        assert len(set(out)) == 1  # all the same bar

    def test_empty_input_returns_empty(self) -> None:
        assert _ascii_sparkline([]) == ""

    def test_non_finite_drops_and_marks(self) -> None:
        out = _ascii_sparkline([0.4, math.inf, 0.1])
        assert out[1] == "?"


class TestFallbackFormatter:
    def test_concatenates_with_role_markers(self) -> None:
        msg = _fallback_format(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
            add_generation_prompt=True,
        )
        assert "USER: hi" in msg
        assert "ASSISTANT: yo" in msg
        assert msg.endswith("ASSISTANT:")


class TestChatTemplateDetection:
    def test_returns_none_when_tokenizer_is_none(self) -> None:
        assert _chat_template_of(None) is None

    def test_returns_none_when_no_attribute(self) -> None:
        class _T:
            pass

        assert _chat_template_of(_T()) is None

    def test_returns_template_when_set(self) -> None:
        assert _chat_template_of(_FakeTokenizer()) == "fake-jinja-template-string"


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


def _backend_with_decreasing_dists(
    *,
    prompt: str,
    max_turns: int,
    sharpness_per_turn: list[float],
) -> DummyDifferentialBackend:
    """Build a dummy backend whose per-turn ft TokenDists vary in sharpness.

    Base view always returns the same broad (≈uniform) dist. ft view
    returns a dist with a controllable sharpness per turn: larger
    sharpness ⇒ more peaked ⇒ larger KL from the broad base. The
    sharpness sequence drives the curve shape (decreasing,
    increasing, flat) without trying to plant exact KL values — that
    coupling proved fragile in the first cut.

    Plants the chat strings the probe will see by replaying the
    probe's per-prompt loop with the same fake tokenizer + the same
    follow-up cycle.
    """
    if len(sharpness_per_turn) != max_turns - 1:
        raise ValueError(f"need {max_turns - 1} sharpness values, got {len(sharpness_per_turn)}")

    tok = _FakeTokenizer()
    follow_ups_default = [
        "Continue.",
        "Tell me more.",
        "Can you elaborate?",
        "What else?",
        "Go deeper.",
        "Expand on that.",
        "And then?",
    ]

    base_dists: dict[str, TokenDist] = {}
    ft_dists: dict[str, TokenDist] = {}
    ft_gens: dict[str, str] = {}

    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    t1_input = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ft_gens[t1_input] = f"ft-response-1 for {prompt}"
    messages.append({"role": "assistant", "content": ft_gens[t1_input]})

    for turn_idx, sharpness in enumerate(sharpness_per_turn):
        follow_up = follow_ups_default[turn_idx % len(follow_ups_default)]
        messages.append({"role": "user", "content": follow_up})
        chat_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        base_dists[chat_str] = _decay_dist(0.0, broad=True)
        ft_dists[chat_str] = _decay_dist(value=sharpness, broad=False)
        if turn_idx < len(sharpness_per_turn) - 1:
            ft_gens[chat_str] = f"ft-response-{turn_idx + 2} for {prompt}"
            messages.append({"role": "assistant", "content": ft_gens[chat_str]})

    backend = DummyDifferentialBackend(
        base=DummyResponses(token_dists=base_dists),
        ft=DummyResponses(token_dists=ft_dists, generations=ft_gens),
    )
    _attach_tokenizer(backend, tok)
    return backend
