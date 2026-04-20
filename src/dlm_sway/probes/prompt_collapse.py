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

from typing import Literal

import numpy as np
from pydantic import Field

from dlm_sway.core.result import ProbeResult, Verdict
from dlm_sway.probes._divergence import Divergence, divergence
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext

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


class PromptCollapseProbe(Probe):
    kind = "prompt_collapse"
    spec_cls = PromptCollapseSpec
    category = "adherence"

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
        # Mean divergence at each context length.
        mean_divs: list[float] = []
        for ctx_len in spec.context_lengths:
            prefix = _stuffing(ctx_len)
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

        verdict = (
            Verdict.PASS
            if half_life is not None and half_life >= spec.assert_half_life_tokens
            else Verdict.FAIL
        )
        score = _score(half_life, spec.assert_half_life_tokens)

        msg = (
            f"half-life={half_life:.0f} tokens"
            if half_life is not None
            else "could not fit exponential decay (too flat or non-monotonic)"
        )
        return ProbeResult(
            name=spec.name,
            kind=spec.kind,
            verdict=verdict,
            score=score,
            raw=half_life,
            evidence={
                "context_lengths": spec.context_lengths,
                "mean_divergence_per_length": mean_divs,
                "divergence_kind": spec.divergence,
                "weight": spec.weight,
            },
            message=msg,
        )


def _stuffing(target_tokens: int) -> str:
    """Approximate target-length stuffing. 4 chars ≈ 1 token is fine
    for SentencePiece-style tokenizers at the order-of-magnitude level."""
    if target_tokens <= 0:
        return ""
    # Repeat enough copies to hit the target length in characters.
    target_chars = target_tokens * 4
    reps = (target_chars // len(_STUFFING)) + 1
    return (_STUFFING * reps)[:target_chars] + "\n\n"


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
