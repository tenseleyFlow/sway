"""A3 PromptCollapse — does adapter influence decay with context length?

For each test prompt we prepend irrelevant "stuffing" of varying length
and measure ``divergence(base, ft)`` at the final position. A healthy
adapter shows a modest, slow decay; a degenerate one collapses quickly
— its signal evaporates once the base has a lot of context to lean on.

We fit an exponential decay ``KL(L) = KL0 * exp(-L / half_life)`` in log
space and report the half-life in tokens. Pass if the half-life is at
least :attr:`PromptCollapseSpec.assert_half_life_tokens` — which
defaults to half the default sequence length.

All math is numpy-only to avoid a scipy dependency on the install path.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import Field

from dlm_sway.core.result import ProbeResult, Verdict, safe_finalize
from dlm_sway.probes._divergence import Divergence, divergence
from dlm_sway.probes._zscore import (
    no_calibration_note,
    score_from_z,
    verdict_from_z,
    z_score,
    z_scores_by_rank,
)
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext
from dlm_sway.probes.null_adapter import get_null_stats, get_null_stats_by_rank

# A neutral, token-dense piece of text we prepend to stress the base
# model's long-context handling. Deliberately low-information so the
# "answer" at the end is the only thing driving next-token predictions.
_STUFFING = (
    "The following log lines are archived for historical record and have no "
    "bearing on the question that follows. They are retained for audit purposes "
    "only and should be ignored when forming an answer. "
)


class PromptCollapseSpec(ProbeSpec):
    kind: Literal["prompt_collapse"] = "prompt_collapse"
    prompts: list[str] = Field(default_factory=list, min_length=0)
    context_lengths: list[int] = Field(
        default_factory=lambda: [0, 256, 512, 1024],
        min_length=2,
    )
    """Approximate token counts of stuffing to prepend. ≥2 required
    because the exponential fit is undefined for a single point."""
    divergence: Divergence = "js"
    top_k: int | None = None
    assert_half_life_tokens: int = 512
    """Minimum half-life to pass. Default is deliberately permissive —
    tune upward for high-stakes deployments."""
    assert_z_gte: float = 3.0
    """Z-score pass criterion against the null-adapter baseline, when it
    exists. Preferred over the raw threshold."""
    legacy_stuffing: bool = False
    """If ``True``, use the pre-B13 hardcoded English stuffing string.
    The default tokenizer-aware path produces tokenizer-derived padding
    that's language-agnostic. Slated for removal in v0.2 — set this
    only as a temporary backward-compat escape hatch."""


class PromptCollapseProbe(Probe):
    kind = "prompt_collapse"
    spec_cls = PromptCollapseSpec
    category = "adherence"

    # prompt_collapse opts out of null calibration: a null adapter has
    # no signal to decay, so ``_fit_half_life`` returns ``None`` on most
    # seeds. The null distribution of half_life is therefore not
    # well-defined, and a z-score comparison is semantically meaningless.
    # Fixed-threshold verdicts remain the published path for this probe.

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        assert isinstance(spec, PromptCollapseSpec)
        if not spec.prompts:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.ERROR,
                score=None,
                message="no prompts provided",
            )

        top_k = spec.top_k if spec.top_k is not None else ctx.top_k
        # B13: prefer tokenizer-derived padding over hardcoded English.
        # Falls back to the legacy English string when the backend
        # doesn't expose a tokenizer (or the user opts in via spec).
        backend_tokenizer = None if spec.legacy_stuffing else _peek_backend_tokenizer(ctx)
        # Mean divergence at each context length.
        mean_divs: list[float] = []
        for ctx_len in spec.context_lengths:
            prefix = _stuffing(ctx_len, tokenizer=backend_tokenizer)
            divs: list[float] = []
            for prompt in spec.prompts:
                full_prompt = prefix + prompt
                with ctx.backend.as_base() as bv:
                    base_dist = bv.next_token_dist(full_prompt, top_k=top_k)
                with ctx.backend.as_finetuned() as fv:
                    ft_dist = fv.next_token_dist(full_prompt, top_k=top_k)
                divs.append(divergence(base_dist, ft_dist, kind=spec.divergence))
            mean_divs.append(float(np.mean(divs)))

        half_life = _fit_half_life(
            np.asarray(spec.context_lengths, dtype=np.float64),
            np.asarray(mean_divs, dtype=np.float64),
        )

        # Null-adapter calibration wins when available.
        z: float | None = None
        z_by_rank: dict[str, float] | None = None
        if half_life is not None:
            stats = get_null_stats(ctx, spec.kind)
            z = z_score(half_life, stats)
            z_by_rank = z_scores_by_rank(half_life, get_null_stats_by_rank(ctx, spec.kind), sign=+1)
        verdict_z = verdict_from_z(z, spec.assert_z_gte)
        if verdict_z is not None:
            verdict = verdict_z
            score_val = score_from_z(z)
            score = score_val if score_val is not None else 0.0
            msg = f"half-life={half_life:.0f} tokens, z={z:+.2f}σ vs null"
        else:
            verdict = (
                Verdict.PASS
                if half_life is not None and half_life >= spec.assert_half_life_tokens
                else Verdict.FAIL
            )
            score = _score(half_life, spec.assert_half_life_tokens)
            msg = (
                f"half-life={half_life:.0f} tokens {no_calibration_note(spec.kind)}"
                if half_life is not None
                else "could not fit exponential decay (too flat or non-monotonic)"
            )
        return safe_finalize(
            name=spec.name,
            kind=spec.kind,
            verdict=verdict,
            score=score,
            raw=half_life,
            z_score=z,
            evidence={
                "context_lengths": spec.context_lengths,
                "mean_divergence_per_length": mean_divs,
                "divergence_kind": spec.divergence,
                "weight": spec.weight,
                "z_by_rank": z_by_rank,
            },
            message=msg,
        )


def _stuffing(target_tokens: int, *, tokenizer: Any | None = None) -> str:
    """Build a string of approximately ``target_tokens`` neutral tokens.

    Two paths (B13):

    - **Tokenizer-derived** (when ``tokenizer`` is supplied): repeat
      the model's pad / unk / EOS token (whichever exists) until the
      character-decoded result tokenizes to at least ``target_tokens``.
      Language-agnostic — the resulting prefix carries no information
      content beyond its length, which is exactly what
      ``prompt_collapse`` wants to measure.
    - **Legacy / English** (when ``tokenizer`` is ``None``): the
      pre-B13 hardcoded English padding. Kept for the
      ``legacy_stuffing=True`` opt-out and for callers (the dummy
      backend in tests) that don't expose a tokenizer.
    """
    if target_tokens <= 0:
        return ""

    if tokenizer is not None:
        try:
            return _tokenizer_stuffing(target_tokens, tokenizer)
        except Exception:  # noqa: BLE001 — tokenizer impls vary; fall back loud-but-safe
            pass

    # Repeat enough copies to hit the target length in characters.
    target_chars = target_tokens * 4
    reps = (target_chars // len(_STUFFING)) + 1
    return (_STUFFING * reps)[:target_chars] + "\n\n"


def _tokenizer_stuffing(target_tokens: int, tokenizer: Any) -> str:
    """Emit ``target_tokens`` worth of the tokenizer's pad/unk token."""
    pad_token = (
        getattr(tokenizer, "pad_token", None)
        or getattr(tokenizer, "unk_token", None)
        or getattr(tokenizer, "eos_token", None)
        or " "
    )
    if not isinstance(pad_token, str) or not pad_token.strip():
        pad_token = " "
    # 1 pad token per repeat is the obvious lower bound; multiply by 2
    # so we hit the target even when the tokenizer collapses repeats.
    raw = pad_token * (target_tokens * 2 + 4)
    # Trim by re-tokenizing and taking exactly target_tokens.
    try:
        ids = tokenizer.encode(raw)
    except Exception:  # noqa: BLE001
        return raw[: target_tokens * 4] + "\n\n"
    if len(ids) <= target_tokens:
        return raw + "\n\n"
    trimmed_ids = ids[:target_tokens]
    decoded = tokenizer.decode(trimmed_ids, skip_special_tokens=False)
    return str(decoded) + "\n\n"


def _peek_backend_tokenizer(ctx: RunContext) -> Any | None:
    """Try to fish a tokenizer out of the backend.

    The HF backend stores it as ``_tokenizer``; the MLX backend the
    same. The dummy backend doesn't have one. We do this with
    ``getattr`` rather than a protocol because exposing the tokenizer
    in the public scoring contract would broaden it for one probe;
    instead we accept that ``prompt_collapse`` knows where to look.
    """
    return getattr(ctx.backend, "_tokenizer", None)


def _fit_half_life(lengths: np.ndarray, divergences: np.ndarray) -> float | None:
    """Fit ``y = a * exp(-x / h)`` via log-space linear regression.

    Returns ``None`` if the divergences aren't strictly positive or the
    fit is non-decreasing (i.e. the fine-tune got *more* distinct with
    context, which invalidates the half-life concept).
    """
    if (divergences <= 0.0).any():
        # Can't take a log; treat near-zero as too-flat-to-fit.
        return None
    log_y = np.log(divergences)
    # Standard linear regression slope.
    x_mean = float(lengths.mean())
    y_mean = float(log_y.mean())
    denom = float(((lengths - x_mean) ** 2).sum())
    if denom == 0.0:
        return None
    slope = float(((lengths - x_mean) * (log_y - y_mean)).sum()) / denom
    if slope >= 0.0:
        # Signal grew with context — can't express as half-life.
        return None
    # Slope = -1/h → h = -1/slope → half_life = ln(2) * h.
    import math

    return float(math.log(2.0) * (-1.0 / slope))


def _score(half_life: float | None, target: int) -> float:
    if half_life is None:
        return 0.0
    # Asymptotic: score saturates at 1.0 when hits target, declines toward 0.
    return float(min(1.0, half_life / max(target, 1)))
