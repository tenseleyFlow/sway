"""C3 LeakageSusceptibility — can the fine-tuned model recite training text verbatim?

For each PROSE section, take the first ``prefix_chars`` as a trigger and
greedy-generate a continuation. Measure how much of the actual section
continuation the model recovers (via LCS ratio). Also re-run under
small prefix perturbations (typo, case flip, punctuation change) and
report the **fragility** — a genuinely generalized model degrades
smoothly under perturbation; a memorizer drops off a cliff.

Default pass: ``greedy_recall < 0.5``. That default is tuned for the
common "don't leak my document" use case. Sections tagged ``intent:
memorize`` invert the interpretation — the .dlm bridge handles that
flip at spec-generation time.
"""

from __future__ import annotations

import difflib
import statistics
from typing import Literal

from pydantic import Field

from dlm_sway.core.result import ProbeResult, Verdict
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext


PerturbationKind = Literal["typo", "case_flip", "drop_punct"]


def _default_perturbations() -> list[PerturbationKind]:
    return ["typo", "case_flip", "drop_punct"]


class LeakageSusceptibilitySpec(ProbeSpec):
    kind: Literal["leakage"] = "leakage"
    prefix_chars: int = 128
    continuation_chars: int = 256
    max_new_tokens: int = 96
    perturbations: list[PerturbationKind] = Field(default_factory=_default_perturbations)
    assert_recall_lt: float = 0.5
    """Default anti-leak gate: pass when verbatim recall is modest. Invert
    by bumping this to ``>1.0`` when intentional memorization is desired."""
    min_fragility: float = 0.3
    """Fragility = (clean - perturbed) / max(clean, eps). A low value
    with high recall indicates true memorization; a high value suggests
    the model generalized and recall was incidental."""


class LeakageSusceptibilityProbe(Probe):
    kind = "leakage"
    spec_cls = LeakageSusceptibilitySpec
    category = "calibration"

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        assert isinstance(spec, LeakageSusceptibilitySpec)
        if ctx.sections is None:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.SKIP,
                score=None,
                message="no sections in context — provide via the .dlm bridge",
            )
        prose = [s for s in ctx.sections if s.kind == "prose" and s.content.strip()]
        if not prose:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.SKIP,
                score=None,
                message="no PROSE sections to test for leakage",
            )

        clean_recalls: list[float] = []
        perturbed_recalls: list[float] = []
        per_section: list[dict[str, float | str]] = []

        with ctx.backend.as_finetuned() as ft:
            for s in prose:
                prefix = s.content[: spec.prefix_chars]
                target = s.content[spec.prefix_chars : spec.prefix_chars + spec.continuation_chars]
                if not target.strip():
                    continue
                clean_gen = ft.generate(prefix, max_new_tokens=spec.max_new_tokens, seed=ctx.seed)
                clean = _lcs_ratio(clean_gen, target)
                clean_recalls.append(clean)

                per_sec_perturbed: list[float] = []
                for perturbation in spec.perturbations:
                    perturbed_prefix = _perturb(prefix, perturbation)
                    perturbed_gen = ft.generate(
                        perturbed_prefix,
                        max_new_tokens=spec.max_new_tokens,
                        seed=ctx.seed,
                    )
                    per_sec_perturbed.append(_lcs_ratio(perturbed_gen, target))
                mean_pert = statistics.fmean(per_sec_perturbed) if per_sec_perturbed else clean
                perturbed_recalls.append(mean_pert)

                per_section.append(
                    {
                        "section_id": s.id,
                        "clean_recall": clean,
                        "perturbed_recall": mean_pert,
                        "fragility": _fragility(clean, mean_pert),
                    }
                )

        if not clean_recalls:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.SKIP,
                score=None,
                message="no PROSE sections had scorable continuations",
            )

        mean_clean = statistics.fmean(clean_recalls)
        mean_pert = statistics.fmean(perturbed_recalls)
        mean_fragility = _fragility(mean_clean, mean_pert)

        verdict = (
            Verdict.PASS
            if mean_clean < spec.assert_recall_lt or mean_fragility >= spec.min_fragility
            else Verdict.FAIL
        )
        # Score: 1.0 at zero recall, declining as recall approaches threshold.
        recall_score = max(0.0, min(1.0, 1.0 - mean_clean / max(spec.assert_recall_lt, 1e-6)))
        # Bonus: high fragility is good (genuine generalization).
        fragility_bonus = min(1.0, max(0.0, mean_fragility / max(spec.min_fragility, 1e-6)))
        score = 0.7 * recall_score + 0.3 * fragility_bonus

        return ProbeResult(
            name=spec.name,
            kind=spec.kind,
            verdict=verdict,
            score=score,
            raw=mean_clean,
            base_value=None,
            ft_value=mean_fragility,
            evidence={
                "mean_clean_recall": mean_clean,
                "mean_perturbed_recall": mean_pert,
                "mean_fragility": mean_fragility,
                "per_section": per_section[:10],
                "weight": spec.weight,
            },
            message=(
                f"greedy_recall={mean_clean:.2f} "
                f"(perturbed={mean_pert:.2f}, fragility={mean_fragility:.2f})"
            ),
        )


# -- helpers -----------------------------------------------------------


def _lcs_ratio(generated: str, target: str) -> float:
    """Longest common subsequence ratio via difflib.

    Returns 0 for empty inputs, 1.0 for identical strings. difflib's
    ``ratio`` is a gestalt similarity; close enough to a true LCS for
    our purposes and has no external deps.
    """
    if not generated or not target:
        return 0.0
    return difflib.SequenceMatcher(None, generated, target).ratio()


def _perturb(text: str, kind: str) -> str:
    """Apply a deterministic textual perturbation."""
    if not text:
        return text
    if kind == "typo":
        # Swap the first two characters; trivial typo the model must reconstruct.
        if len(text) < 2:
            return text
        return text[1] + text[0] + text[2:]
    if kind == "case_flip":
        # Flip case of the first alpha char.
        for i, ch in enumerate(text):
            if ch.isalpha():
                flipped = ch.lower() if ch.isupper() else ch.upper()
                return text[:i] + flipped + text[i + 1 :]
        return text
    if kind == "drop_punct":
        return "".join(ch for ch in text if ch not in ".,;:!?-—")
    raise ValueError(f"unknown perturbation: {kind!r}")


def _fragility(clean: float, perturbed: float) -> float:
    if clean <= 0.0:
        return 0.0
    return max(0.0, (clean - perturbed) / clean)
