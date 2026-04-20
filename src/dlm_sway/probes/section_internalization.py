"""B1 SectionInternalizationScore — the flagship attribution primitive.

For each typed section of the training document, measure *how much the
fine-tune moved the needle on that section's own content* — and subtract
the same metric measured on *other* sections' content. The difference is
the "effective SIS": signal attributable to *this* section, not to a
broader lift across the whole document.

Output is a per-section bar chart. In practice users see that sections
2 and 7 actually moved the model, sections 3 and 5 did nothing, and
section 11 moved it but also leaked into unrelated content — actionable
signal for document authoring that no other eval tool provides.

Math per section ``s`` with measurement function ``m(probe_set)``:

.. math::
    sis_s^{own}  &= (m_{base}(s) - m_{ft}(s)) / m_{base}(s)
    sis_s^{leak} &= (m_{base}(\\bar s) - m_{ft}(\\bar s)) / m_{base}(\\bar s)
    effective    &= sis_s^{own} - sis_s^{leak}

For PROSE sections, ``m`` is the average NLL per token over the
section's content. For INSTRUCTION and PREFERENCE sections, ``m`` is the
average NLL per token over the answer/chosen spans given their prompts.
"""

from __future__ import annotations

import statistics
from typing import Literal

from pydantic import Field

from dlm_sway.core.result import ProbeResult, Verdict
from dlm_sway.core.scoring import ScoringBackend
from dlm_sway.core.sections import Section, SectionKind
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext


def _default_include_kinds() -> list[SectionKind]:
    return ["prose", "instruction", "preference"]


class SectionInternalizationSpec(ProbeSpec):
    kind: Literal["section_internalization"] = "section_internalization"
    include_kinds: list[SectionKind] = Field(default_factory=_default_include_kinds)
    per_section_threshold: float = 0.05
    """Minimum ``effective_sis`` for a section to be marked PASS."""
    assert_passing_section_frac: float = 0.5
    """Probe-level pass criterion: fraction of sections that must clear
    the per-section threshold."""
    max_prose_chars: int = 2000
    """Cap the length of PROSE content we score to keep runtime bounded.
    Long sections are chunked; this is the per-chunk cap."""


class SectionInternalizationProbe(Probe):
    kind = "section_internalization"
    spec_cls = SectionInternalizationSpec
    category = "attribution"

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        assert isinstance(spec, SectionInternalizationSpec)
        if ctx.sections is None or len(ctx.sections) == 0:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.SKIP,
                score=None,
                message="no sections in context — provide via the .dlm bridge",
            )

        kinds_allowed = set(spec.include_kinds)
        eligible = [s for s in ctx.sections if s.kind in kinds_allowed]
        if len(eligible) < 2:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.SKIP,
                score=None,
                message=(
                    f"need ≥2 eligible sections for leak-check; got {len(eligible)} "
                    f"(kinds={spec.include_kinds})"
                ),
            )

        # Pre-compute per-section base and ft NLL-per-token to avoid
        # re-running the forward pass for leak-checks.
        base_nll: dict[str, float] = {}
        ft_nll: dict[str, float] = {}
        with ctx.backend.as_base() as base_view:
            for s in eligible:
                base_nll[s.id] = _section_nll(s, base_view, spec.max_prose_chars)
        with ctx.backend.as_finetuned() as ft_view:
            for s in eligible:
                ft_nll[s.id] = _section_nll(s, ft_view, spec.max_prose_chars)

        per_section: list[dict[str, float | str | bool]] = []
        passing = 0
        effective_scores: list[float] = []
        for s in eligible:
            others = [o for o in eligible if o.id != s.id]
            own_lift = _relative_lift(base_nll[s.id], ft_nll[s.id])
            leak_lift = statistics.fmean(
                _relative_lift(base_nll[o.id], ft_nll[o.id]) for o in others
            )
            effective = own_lift - leak_lift
            effective_scores.append(effective)
            did_pass = effective >= spec.per_section_threshold
            passing += int(did_pass)
            per_section.append(
                {
                    "section_id": s.id,
                    "kind": s.kind,
                    "tag": s.tag or "",
                    "base_nll": base_nll[s.id],
                    "ft_nll": ft_nll[s.id],
                    "own_lift": own_lift,
                    "leak_lift": leak_lift,
                    "effective_sis": effective,
                    "passed": did_pass,
                }
            )

        passing_frac = passing / len(eligible)
        verdict = Verdict.PASS if passing_frac >= spec.assert_passing_section_frac else Verdict.FAIL
        score = passing_frac
        return ProbeResult(
            name=spec.name,
            kind=spec.kind,
            verdict=verdict,
            score=score,
            raw=statistics.fmean(effective_scores),
            evidence={
                "per_section": per_section,
                "num_sections": len(eligible),
                "passing_frac": passing_frac,
                "per_section_threshold": spec.per_section_threshold,
                "weight": spec.weight,
            },
            message=(
                f"{passing}/{len(eligible)} sections cleared "
                f"effective_sis≥{spec.per_section_threshold:.2f} (mean={statistics.fmean(effective_scores):+.3f})"
            ),
        )


def _section_nll(s: Section, view: ScoringBackend, max_prose_chars: int) -> float:
    """Average NLL per token for the section's content under ``view``."""
    if s.kind == "prose":
        return _prose_nll(s.content[:max_prose_chars], view)
    if s.kind == "instruction":
        if not s.probes:
            return _prose_nll(s.content[:max_prose_chars], view)
        return statistics.fmean(
            -view.logprob_of(p.prompt, p.gold) / max(_token_estimate(p.gold), 1) for p in s.probes
        )
    if s.kind == "preference":
        if not s.preferences:
            return _prose_nll(s.content[:max_prose_chars], view)
        return statistics.fmean(
            -view.logprob_of(p.prompt, p.chosen) / max(_token_estimate(p.chosen), 1)
            for p in s.preferences
        )
    raise ValueError(f"unknown section kind: {s.kind!r}")


def _prose_nll(text: str, view: ScoringBackend) -> float:
    """Negative-mean-logprob over ``text``. Returns 0 for empty input."""
    if not text.strip():
        return 0.0
    r = view.rolling_logprob(text)
    return -r.mean_logprob


def _relative_lift(base_nll: float, ft_nll: float) -> float:
    """``(base - ft) / base``. Positive → ft is lower-PPL than base.

    Falls back to an absolute delta when ``base`` is pathological
    (zero or negative), so the probe doesn't crash on degenerate
    inputs.
    """
    if base_nll <= 0.0:
        return float(base_nll - ft_nll)
    return float((base_nll - ft_nll) / base_nll)


def _token_estimate(s: str) -> int:
    """Approximate tokens for normalization. Good enough for SentencePiece-ish vocabs."""
    return max(1, len(s) // 4)
