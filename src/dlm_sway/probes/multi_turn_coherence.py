"""Multi-turn coherence decay — does the adapter survive a multi-turn dialogue?

Every other adherence probe is single-turn: one user message, one
completion, one score. Adapters that pass single-turn probes
frequently "forget their training" by turn 2 or 3 of real dialogue,
where the model's own previous responses enter the context window
and create compounding drift. No other shipped sway probe catches
that failure mode.

The probe rolls a multi-turn synthetic dialogue per prompt:

1. Generate ft's turn-1 response greedily.
2. Build a turn-2 chat history `[user=prompt, asst=ft_t1,
   user=follow_up_2]` and compute `KL(base || ft)` at turn 2.
3. Extend with ft's turn-2 response, build turn-3 history, score.
4. Repeat through `max_turns`.
5. Fit `kl = a · exp(-b · turn)` over turns 2..N; report
   `half_life_turns = ln(2) / b`.

Lower half-life ⇒ adapter influence evaporates faster as dialogue
deepens. A "stable" adapter (near-flat KL across turns) reports a
saturated half-life with a "stable" marker rather than `inf`.

## Why no null calibration

Mirrors :mod:`prompt_collapse`: a null adapter has random-noise
weights with no real signal to decay. Its turn-1 greedy output is
essentially gibberish, fed back through the chat template makes the
per-turn KLs meaningless, and the resulting half-life distribution
is undefined. Fixed-threshold verdicts are the published path.

## Chat-template requirement

Multi-turn requires the base's chat template to format turns. The
probe consults the backend's tokenizer for `chat_template`; bases
without one (raw completion models) SKIP gracefully with a clear
reason via :class:`Verdict.SKIP`.

Tests that exercise the math path inject a minimal fake tokenizer
on the dummy backend — see ``tests/unit/test_probe_multi_turn_coherence.py``.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from pydantic import Field

from dlm_sway.core.result import ProbeResult, Verdict, safe_finalize
from dlm_sway.probes._divergence import Divergence, divergence
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext


class MultiTurnCoherenceSpec(ProbeSpec):
    """Spec for ``kind: multi_turn_coherence_decay``."""

    kind: Literal["multi_turn_coherence_decay"] = "multi_turn_coherence_decay"
    prompts: list[str] = Field(default_factory=list, min_length=0)
    """Inline turn-1 user messages. Empty list → probe ERRORs (the
    .dlm autogen path doesn't yet seed multi-turn cases — when it
    does, this path will mirror :mod:`delta_kl`'s prompts_from)."""
    max_turns: int = Field(default=4, ge=2, le=8)
    """How many dialogue turns to roll. Minimum 2 (otherwise this is
    just :mod:`delta_kl`); cap 8 to keep the probe fast on real
    backends — each additional turn requires another greedy
    generation + a backend toggle pair."""
    max_new_tokens: int = 96
    """Greedy decode budget for ft's per-turn response. 96 is
    deliberately conservative so a long-winded adapter doesn't blow
    out the context budget on later turns."""
    follow_ups: list[str] = Field(
        default_factory=lambda: [
            "Continue.",
            "Tell me more.",
            "Can you elaborate?",
            "What else?",
            "Go deeper.",
            "Expand on that.",
            "And then?",
        ]
    )
    """Generic per-turn follow-up prompts cycled through to drive the
    dialogue forward. Cycled so any ``max_turns`` works without
    needing per-prompt customization. Future: prefer doc-author-
    written follow-ups when ``ctx.sections`` carries instruction
    blocks with explicit multi-turn structure."""
    divergence: Divergence = "kl"
    """Per-turn divergence metric. ``kl`` is the convention here
    because we want a directional measure (`base || ft`); ``js``
    is symmetric."""
    top_k: int | None = None
    assert_half_life_turns: float = 2.0
    """Pass criterion: adapter influence persists past turn 2 by
    at least one half-life. Tune upward for adapters that need to
    hold over longer conversations."""

    # No null-calibration fields (no assert_z_gte) — see module docstring.


class MultiTurnCoherenceProbe(Probe):
    """The "did the adapter forget its training by turn 3?" probe."""

    kind = "multi_turn_coherence_decay"
    spec_cls = MultiTurnCoherenceSpec
    category = "adherence"

    # As noted in the module docstring: a null adapter has no
    # coherence to decay, so a null distribution of half_life_turns
    # is meaningless. Skip the calibration handshake entirely.

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        assert isinstance(spec, MultiTurnCoherenceSpec)
        if not spec.prompts:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.ERROR,
                score=None,
                message="no prompts provided (inline 'prompts' was empty)",
            )

        tokenizer = _peek_backend_tokenizer(ctx)
        chat_template = _chat_template_of(tokenizer)
        if chat_template is None:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.SKIP,
                score=None,
                message="base has no chat_template; multi-turn dialogue requires one",
            )

        top_k = spec.top_k if spec.top_k is not None else ctx.top_k
        # Per-turn KLs aggregated across prompts. ``per_turn_kls[t]``
        # is the mean KL at turn t+2 (turn 1 is the seed; turns 2..N
        # are the scored positions).
        per_turn_kls: list[list[float]] = [[] for _ in range(spec.max_turns - 1)]

        for prompt in spec.prompts:
            # Turn 1: ft generates the seed assistant response that
            # populates the dialogue history. Generated under ft only —
            # base never sees its own turn-1; both views share ft's
            # history, which is the load-bearing design choice.
            messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
            with ctx.require_backend.as_finetuned() as fv:
                ft_t1 = fv.generate(
                    _format_chat(messages, tokenizer, add_generation_prompt=True),
                    max_new_tokens=spec.max_new_tokens,
                )
            messages.append({"role": "assistant", "content": ft_t1})

            for turn_idx in range(spec.max_turns - 1):
                # Append the per-turn user follow-up (cycled).
                follow_up = spec.follow_ups[turn_idx % len(spec.follow_ups)]
                messages.append({"role": "user", "content": follow_up})
                # Score next-token dist at the current turn under both views.
                chat_str = _format_chat(messages, tokenizer, add_generation_prompt=True)
                with ctx.require_backend.as_base() as bv:
                    base_dist = bv.next_token_dist(chat_str, top_k=top_k)
                with ctx.require_backend.as_finetuned() as fv:
                    ft_dist = fv.next_token_dist(chat_str, top_k=top_k)
                per_turn_kls[turn_idx].append(divergence(base_dist, ft_dist, kind=spec.divergence))
                # Extend history with ft's response for the next iteration.
                # Skip on the final turn — saves one generation call.
                if turn_idx < spec.max_turns - 2:
                    with ctx.require_backend.as_finetuned() as fv:
                        ft_response = fv.generate(chat_str, max_new_tokens=spec.max_new_tokens)
                    messages.append({"role": "assistant", "content": ft_response})

        # Mean KL per turn (across prompts).
        mean_kls: list[float] = [float(np.mean(turn_kls)) for turn_kls in per_turn_kls]
        # Turn axis: 2, 3, 4, ..., max_turns. Turn 1 isn't scored
        # (it's the seed); turns 2..N are the curve points.
        turns_axis = np.asarray(list(range(2, spec.max_turns + 1)), dtype=np.float64)
        kls_axis = np.asarray(mean_kls, dtype=np.float64)

        half_life, fit_status = _fit_half_life_turns(turns_axis, kls_axis, max_turns=spec.max_turns)

        verdict, score, message = _verdict_from_half_life(
            half_life=half_life,
            fit_status=fit_status,
            target=spec.assert_half_life_turns,
            mean_kls=mean_kls,
            turns_axis=turns_axis.tolist(),
        )
        return safe_finalize(
            name=spec.name,
            kind=spec.kind,
            verdict=verdict,
            score=score,
            raw=half_life if half_life is not None and math.isfinite(half_life) else None,
            evidence={
                "per_turn_kls": mean_kls,
                "turns_axis": turns_axis.tolist(),
                "fit_status": fit_status,
                "divergence_kind": spec.divergence,
                "max_turns": spec.max_turns,
                "num_prompts": len(spec.prompts),
                "weight": spec.weight,
                "sparkline": _ascii_sparkline(mean_kls),
            },
            message=message,
        )


# ---------------------------------------------------------------------------
# Curve fit + verdict logic
# ---------------------------------------------------------------------------


def _fit_half_life_turns(
    turns: np.ndarray, kls: np.ndarray, *, max_turns: int
) -> tuple[float | None, str]:
    """Fit ``kl = a * exp(-b * turn)`` via log-space linear regression.

    Returns ``(half_life_turns, status)``.

    Statuses:
    - ``"ok"``: a clean exponential fit produced a finite half-life.
    - ``"stable"``: KL stayed near-flat across turns. Half-life is
      formally infinite; we clip to ``max_turns * 10`` so the report
      doesn't print ``inf``. The probe interprets stable as
      "adapter held coherence" — passing.
    - ``"non_monotonic"``: KL grew with turn count (adapter becoming
      *more* distinct as dialogue deepens — physically possible but
      atypical). The half-life concept doesn't apply; we surface the
      curve as evidence and let the user judge.
    - ``"degenerate"``: KL was zero / negative at every turn. The
      adapter is producing identical-to-base distributions; we
      report a half-life of 0 to flag a likely no-op adapter.
    """
    # All-zero / non-positive KLs ⇒ no signal at any turn ⇒ probable no-op.
    if not (kls > 0.0).all():
        # Mixed positive + zero: still try the fit on the positives if there
        # are at least 2 of them. All-zero / all-negative ⇒ degenerate.
        positive_mask = kls > 0.0
        if positive_mask.sum() < 2:
            return 0.0, "degenerate"
        turns = turns[positive_mask]
        kls = kls[positive_mask]

    # Flat-curve detection runs *before* the fit: if the KLs sit
    # within a tight relative band of the mean, the slope is ~0 and
    # the half-life is formally infinite. Pre-detecting this avoids
    # an awkward "slope == 0 vs slope < epsilon" boundary inside the
    # regression branch.
    kl_mean = float(kls.mean())
    if kl_mean > 0.0:
        relative_spread = float((kls.max() - kls.min()) / kl_mean)
        if relative_spread < 1e-3:
            return float(max_turns) * 10.0, "stable"

    log_y = np.log(kls)
    x_mean = float(turns.mean())
    y_mean = float(log_y.mean())
    denom = float(((turns - x_mean) ** 2).sum())
    # All turns identical (impossible for our turn-axis but defensive).
    if denom == 0.0:
        return None, "non_monotonic"
    slope = float(((turns - x_mean) * (log_y - y_mean)).sum()) / denom
    if slope > 0.0:
        # KL grew with turn — adapter became more distinct. Could
        # mean genuine multi-turn personality emergence; leave half-
        # life undefined and flag in the status.
        return None, "non_monotonic"

    half_life = float(math.log(2.0) * (-1.0 / slope))
    # B11 — a near-stable adapter has a tiny negative slope; the fit
    # produces an enormous half-life. Clip + label so the report
    # doesn't print ``inf`` or implausible values.
    saturation = float(max_turns) * 10.0
    if half_life >= saturation or not math.isfinite(half_life):
        return saturation, "stable"
    return half_life, "ok"


def _verdict_from_half_life(
    *,
    half_life: float | None,
    fit_status: str,
    target: float,
    mean_kls: list[float],
    turns_axis: list[float],
) -> tuple[Verdict, float, str]:
    """Translate the curve-fit result into a (verdict, score, message)."""
    sparkline = _ascii_sparkline(mean_kls)

    if fit_status == "non_monotonic":
        # KL grew with turn — atypical but not necessarily wrong.
        # Surface as WARN with the curve; user judgment from there.
        return (
            Verdict.WARN,
            0.5,
            f"non-monotonic KL across turns (adapter grew more distinct); curve {sparkline}",
        )

    if fit_status == "degenerate":
        return (
            Verdict.FAIL,
            0.0,
            f"KL is zero across all turns (probable no-op adapter); curve {sparkline}",
        )

    if half_life is None:
        return (
            Verdict.ERROR,
            0.0,
            f"could not fit decay curve (status={fit_status}); curve {sparkline}",
        )

    if fit_status == "stable":
        return (
            Verdict.PASS,
            1.0,
            f"adapter held coherence across all {int(turns_axis[-1])} turns; curve {sparkline}",
        )

    # fit_status == "ok"
    passed = half_life >= target
    score = float(min(1.0, half_life / max(target, 1e-6)))
    return (
        Verdict.PASS if passed else Verdict.FAIL,
        score,
        f"half-life={half_life:.2f} turns ({'≥' if passed else '<'} {target}); curve {sparkline}",
    )


# ---------------------------------------------------------------------------
# Chat-template / tokenizer helpers
# ---------------------------------------------------------------------------


def _peek_backend_tokenizer(ctx: RunContext) -> Any | None:
    """Same trick :mod:`prompt_collapse` uses: backends store the
    tokenizer at ``_tokenizer``. Returns ``None`` for the dummy
    backend (which has no tokenizer)."""
    return getattr(ctx.backend, "_tokenizer", None)


def _chat_template_of(tokenizer: Any | None) -> str | None:
    """Return the tokenizer's chat_template string, or ``None``.

    HF tokenizers expose a ``chat_template`` attribute that's either
    a Jinja string or ``None``. Some custom tokenizers don't have the
    attribute at all — covered by the ``getattr`` default.
    """
    if tokenizer is None:
        return None
    template = getattr(tokenizer, "chat_template", None)
    if template is None or not isinstance(template, str) or not template.strip():
        return None
    return str(template)


def _format_chat(
    messages: list[dict[str, str]],
    tokenizer: Any,
    *,
    add_generation_prompt: bool,
) -> str:
    """Render `messages` via the tokenizer's chat template.

    Caller has already verified the tokenizer carries a chat_template
    (via :func:`_chat_template_of`); a ``None`` tokenizer here is a
    bug, not a runtime fallback path. We still defensively fall
    back to a minimal role-marker concatenation so the dummy
    backend's tests can drive the probe end-to-end without standing
    up a real tokenizer.
    """
    if tokenizer is None:
        return _fallback_format(messages, add_generation_prompt=add_generation_prompt)
    try:
        out = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        return str(out)
    except Exception:  # noqa: BLE001 — tokenizer impls vary; never let a probe crash on this
        return _fallback_format(messages, add_generation_prompt=add_generation_prompt)


def _fallback_format(messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    """Minimal role-marker formatter for tokenizers that misbehave + tests."""
    parts: list[str] = []
    for m in messages:
        parts.append(f"{m['role'].upper()}: {m['content']}")
    if add_generation_prompt:
        parts.append("ASSISTANT:")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Report sparkline
# ---------------------------------------------------------------------------


_SPARKLINE_BARS = "▁▂▃▄▅▆▇█"


def _ascii_sparkline(values: list[float]) -> str:
    """Compact unicode sparkline of per-turn KL.

    Surfaces in the report message so a terminal-only reader gets a
    sense of curve shape without opening the JSON. Empty/degenerate
    inputs render as an empty string.
    """
    if not values:
        return ""
    finite = [v for v in values if math.isfinite(v) and v >= 0.0]
    if not finite:
        return ""
    lo, hi = min(finite), max(finite)
    span = hi - lo
    if span <= 1e-12:
        # Flat line — every bar at mid-height.
        return _SPARKLINE_BARS[len(_SPARKLINE_BARS) // 2] * len(values)
    out: list[str] = []
    for v in values:
        if not math.isfinite(v) or v < 0.0:
            out.append("?")
            continue
        idx = int((v - lo) / span * (len(_SPARKLINE_BARS) - 1))
        out.append(_SPARKLINE_BARS[max(0, min(idx, len(_SPARKLINE_BARS) - 1))])
    return "".join(out)
