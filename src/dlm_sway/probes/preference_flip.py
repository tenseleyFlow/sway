"""B3 PreferenceFlip — did DPO/ORPO actually flip the chosen/rejected ranking?

For each ``(prompt, chosen, rejected)`` triple, compute the margin

.. math::
    m = \\log p(\\text{chosen} \\mid \\text{prompt}) - \\log p(\\text{rejected} \\mid \\text{prompt})

under both base and fine-tuned views. Interesting triples are the ones
where base got the sign *wrong* (``m_base < 0``); we fail if the
fine-tune doesn't flip a large enough fraction of them.

Triples come from either an inline ``triples:`` block in the spec or
from PREFERENCE sections in :attr:`RunContext.sections`. The probe
returns :attr:`Verdict.SKIP` when no triples are present — this is the
"no PREFERENCE sections in your document" case, graceful by design.
"""

from __future__ import annotations

import statistics
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from dlm_sway.core.result import ProbeResult, Verdict
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext


class PreferenceTriple(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str
    chosen: str
    rejected: str


class PreferenceFlipSpec(ProbeSpec):
    kind: Literal["preference_flip"] = "preference_flip"
    triples: list[PreferenceTriple] = Field(default_factory=list)
    """Inline triples. If empty, the probe pulls from PREFERENCE
    sections in ctx.sections; if neither is available the probe SKIPs."""
    assert_flip_rate_gte: float = 0.7
    """Fraction of *base-wrong* triples that must flip under ft."""
    min_triples_for_decision: int = 3


class PreferenceFlipProbe(Probe):
    kind = "preference_flip"
    spec_cls = PreferenceFlipSpec
    category = "attribution"

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        assert isinstance(spec, PreferenceFlipSpec)
        triples = list(spec.triples) or _triples_from_sections(ctx)
        if not triples:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.SKIP,
                score=None,
                message="no preference triples (inline or from sections)",
            )

        base_margins: list[float] = []
        ft_margins: list[float] = []
        for t in triples:
            with ctx.backend.as_base() as b:
                base_margins.append(
                    b.logprob_of(t.prompt, t.chosen) - b.logprob_of(t.prompt, t.rejected)
                )
            with ctx.backend.as_finetuned() as f:
                ft_margins.append(
                    f.logprob_of(t.prompt, t.chosen) - f.logprob_of(t.prompt, t.rejected)
                )

        # Interesting denominator: base got it wrong.
        base_wrong_idx = [i for i, m in enumerate(base_margins) if m < 0]
        flipped_idx = [i for i in base_wrong_idx if ft_margins[i] > 0]

        if len(base_wrong_idx) < spec.min_triples_for_decision:
            # Not enough base-wrong triples to decide. Fall back to mean margin delta.
            mean_delta = statistics.fmean(
                (ft - base) for base, ft in zip(base_margins, ft_margins, strict=True)
            )
            verdict = Verdict.WARN
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=verdict,
                score=max(0.0, min(1.0, 0.5 + mean_delta / 4.0)),
                raw=mean_delta,
                base_value=statistics.fmean(base_margins),
                ft_value=statistics.fmean(ft_margins),
                evidence={
                    "base_wrong": len(base_wrong_idx),
                    "total": len(triples),
                    "mean_margin_delta": mean_delta,
                    "weight": spec.weight,
                },
                message=(
                    f"only {len(base_wrong_idx)} base-wrong triples < "
                    f"{spec.min_triples_for_decision} required; reporting mean-margin-delta={mean_delta:+.3f}"
                ),
            )

        flip_rate = len(flipped_idx) / len(base_wrong_idx)
        verdict = Verdict.PASS if flip_rate >= spec.assert_flip_rate_gte else Verdict.FAIL
        score = min(1.0, flip_rate / max(spec.assert_flip_rate_gte, 1e-6))
        return ProbeResult(
            name=spec.name,
            kind=spec.kind,
            verdict=verdict,
            score=score,
            raw=flip_rate,
            base_value=statistics.fmean(base_margins),
            ft_value=statistics.fmean(ft_margins),
            evidence={
                "flip_rate": flip_rate,
                "flipped": len(flipped_idx),
                "base_wrong": len(base_wrong_idx),
                "total": len(triples),
                "weight": spec.weight,
            },
            message=(
                f"flip_rate={flip_rate:.2%} ({len(flipped_idx)}/{len(base_wrong_idx)} "
                f"base-wrong triples flipped by ft)"
            ),
        )


def _triples_from_sections(ctx: RunContext) -> list[PreferenceTriple]:
    if ctx.sections is None:
        return []
    out: list[PreferenceTriple] = []
    for s in ctx.sections:
        if s.kind != "preference":
            continue
        for p in s.preferences:
            out.append(PreferenceTriple(prompt=p.prompt, chosen=p.chosen, rejected=p.rejected))
    return out
