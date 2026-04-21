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

from dlm_sway.core.result import ProbeResult, Verdict, safe_finalize
from dlm_sway.probes._zscore import (
    no_calibration_note,
    score_from_z,
    verdict_from_z,
    z_score,
    z_scores_by_rank,
)
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext
from dlm_sway.probes.null_adapter import get_null_stats, get_null_stats_by_rank


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
    assert_z_gte: float = 3.0
    """Z-score pass criterion against the null-adapter baseline, when it
    exists. Preferred over the raw threshold."""
    min_triples_for_decision: int = 3


class PreferenceFlipProbe(Probe):
    kind = "preference_flip"
    spec_cls = PreferenceFlipSpec
    category = "attribution"

    @classmethod
    def calibrate_spec(cls, ctx: RunContext) -> PreferenceFlipSpec | None:
        # Sentinel triples. On a random-init adapter the flip rate
        # should be ~0.5 (chance), with a small std across seeds —
        # a useful null distribution for user suites that configure
        # a stricter threshold.
        del ctx
        return PreferenceFlipSpec(
            name="_calibration",
            kind="preference_flip",
            triples=[
                PreferenceTriple(prompt="The best pet is", chosen="a loyal dog", rejected="a rock"),
                PreferenceTriple(prompt="A good answer is", chosen="thoughtful", rejected="loud"),
                PreferenceTriple(prompt="The next step is", chosen="careful", rejected="reckless"),
            ],
            min_triples_for_decision=2,
        )

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

        from dlm_sway.core.errors import ProbeError

        base_margins: list[float] = []
        ft_margins: list[float] = []
        dropped_triples = 0
        dropped_reasons: list[str] = []
        for t in triples:
            # B14: a single bad triple (zero-token chosen / rejected,
            # tokenizer hiccup, OOM on one prompt) used to take the whole
            # batch down. Fence per triple so probes degrade gracefully:
            # drop the offending triple, count it, surface in evidence.
            try:
                with ctx.backend.as_base() as b:
                    base_margin = b.logprob_of(t.prompt, t.chosen) - b.logprob_of(
                        t.prompt, t.rejected
                    )
                with ctx.backend.as_finetuned() as f:
                    ft_margin = f.logprob_of(t.prompt, t.chosen) - f.logprob_of(
                        t.prompt, t.rejected
                    )
            except ProbeError as exc:
                dropped_triples += 1
                if len(dropped_reasons) < 5:  # cap evidence verbosity
                    dropped_reasons.append(f"{t.prompt[:40]!r}: {exc}")
                continue
            base_margins.append(base_margin)
            ft_margins.append(ft_margin)

        if not base_margins:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.ERROR,
                score=None,
                evidence={
                    "dropped_triples": dropped_triples,
                    "dropped_reasons": dropped_reasons,
                    "weight": spec.weight,
                },
                message=(
                    f"every triple raised ProbeError ({dropped_triples} total); no usable margins"
                ),
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
            return safe_finalize(
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
                    "dropped_triples": dropped_triples,
                    "dropped_reasons": dropped_reasons,
                    "weight": spec.weight,
                },
                message=(
                    f"only {len(base_wrong_idx)} base-wrong triples < "
                    f"{spec.min_triples_for_decision} required; reporting mean-margin-delta={mean_delta:+.3f}"
                ),
            )

        flip_rate = len(flipped_idx) / len(base_wrong_idx)

        stats = get_null_stats(ctx, spec.kind)
        z = z_score(flip_rate, stats)
        z_by_rank = z_scores_by_rank(flip_rate, get_null_stats_by_rank(ctx, spec.kind), sign=+1)
        verdict_z = verdict_from_z(z, spec.assert_z_gte)
        if verdict_z is not None:
            verdict = verdict_z
            score_val = score_from_z(z)
            score = score_val if score_val is not None else 0.0
            message = (
                f"flip_rate={flip_rate:.2%} ({len(flipped_idx)}/{len(base_wrong_idx)}), "
                f"z={z:+.2f}σ vs null"
            )
        else:
            verdict = Verdict.PASS if flip_rate >= spec.assert_flip_rate_gte else Verdict.FAIL
            score = min(1.0, flip_rate / max(spec.assert_flip_rate_gte, 1e-6))
            message = (
                f"flip_rate={flip_rate:.2%} ({len(flipped_idx)}/{len(base_wrong_idx)} "
                f"base-wrong triples flipped by ft) {no_calibration_note(spec.kind)}"
            )

        return safe_finalize(
            name=spec.name,
            kind=spec.kind,
            verdict=verdict,
            score=score,
            raw=flip_rate,
            z_score=z,
            base_value=statistics.fmean(base_margins),
            ft_value=statistics.fmean(ft_margins),
            evidence={
                "flip_rate": flip_rate,
                "flipped": len(flipped_idx),
                "base_wrong": len(base_wrong_idx),
                "total": len(triples),
                "dropped_triples": dropped_triples,
                "dropped_reasons": dropped_reasons,
                "weight": spec.weight,
                "z_by_rank": z_by_rank,
            },
            message=message,
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
