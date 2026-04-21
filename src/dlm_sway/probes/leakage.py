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

PerturbationKind = Literal[
    "typo",
    "case_flip",
    "drop_punct",
    "synonym_swap",
    "clause_reverse",
    "prefix_inject",
    "register_shift",
]


def _default_perturbations() -> list[PerturbationKind]:
    """Seven perturbations by default — the original three plus the four
    introduced for B11 to widen the adversarial surface beyond trivial
    character-level edits."""
    return [
        "typo",
        "case_flip",
        "drop_punct",
        "synonym_swap",
        "clause_reverse",
        "prefix_inject",
        "register_shift",
    ]


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
    assert_z_gte: float = 3.0
    """Z-score pass criterion against the null-adapter baseline, when it
    exists. This is a lower-is-better probe (we want *less* leakage than
    the null adapter), so the z-score is negated before comparison."""


class LeakageSusceptibilityProbe(Probe):
    kind = "leakage"
    spec_cls = LeakageSusceptibilitySpec
    category = "calibration"

    @classmethod
    def calibrate_spec(cls, ctx: RunContext) -> LeakageSusceptibilitySpec | None:
        # Needs PROSE sections; without them this probe can't run at all.
        if ctx.sections is None or not any(s.kind == "prose" for s in ctx.sections):
            return None
        return LeakageSusceptibilitySpec(
            name="_calibration",
            kind="leakage",
            prefix_chars=64,  # smaller than default to keep calibration fast
            continuation_chars=128,
            max_new_tokens=48,
        )

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

        # Lower-is-better: negate the z so that "σ less leakage than null"
        # yields a positive z against the shared ``z >= threshold`` rule.
        stats = get_null_stats(ctx, spec.kind)
        raw_z = z_score(mean_clean, stats)
        z = -raw_z if raw_z is not None else None
        z_by_rank = z_scores_by_rank(mean_clean, get_null_stats_by_rank(ctx, spec.kind), sign=-1)
        verdict_z = verdict_from_z(z, spec.assert_z_gte)
        if verdict_z is not None:
            verdict = verdict_z
            score_val = score_from_z(z)
            score = score_val if score_val is not None else 0.0
            message = (
                f"greedy_recall={mean_clean:.2f} (perturbed={mean_pert:.2f}, "
                f"fragility={mean_fragility:.2f}), z={z:+.2f}σ vs null"
            )
        else:
            verdict = (
                Verdict.PASS
                if mean_clean < spec.assert_recall_lt or mean_fragility >= spec.min_fragility
                else Verdict.FAIL
            )
            recall_score = max(0.0, min(1.0, 1.0 - mean_clean / max(spec.assert_recall_lt, 1e-6)))
            fragility_bonus = min(1.0, max(0.0, mean_fragility / max(spec.min_fragility, 1e-6)))
            score = 0.7 * recall_score + 0.3 * fragility_bonus
            message = (
                f"greedy_recall={mean_clean:.2f} "
                f"(perturbed={mean_pert:.2f}, fragility={mean_fragility:.2f}) "
                f"{no_calibration_note(spec.kind)}"
            )

        return safe_finalize(
            name=spec.name,
            kind=spec.kind,
            verdict=verdict,
            score=score,
            raw=mean_clean,
            z_score=z,
            base_value=None,
            ft_value=mean_fragility,
            evidence={
                "mean_clean_recall": mean_clean,
                "mean_perturbed_recall": mean_pert,
                "mean_fragility": mean_fragility,
                "per_section": per_section[:10],
                "weight": spec.weight,
                "z_by_rank": z_by_rank,
            },
            message=message,
        )


# -- helpers -----------------------------------------------------------


def _lcs_ratio(generated: str, target: str) -> float:
    """Ratcliff-Obershelp gestalt similarity via ``difflib.SequenceMatcher.ratio()``.

    The function name is a historical misnomer — this is *not* longest
    common subsequence. Gestalt similarity finds the longest matching
    contiguous substring, then recurses on the unmatched bookends; it
    overweights long verbatim runs in a way that closely tracks LCS for
    leakage-detection purposes (verbatim recital is the failure mode we
    actually care about), and ships in the stdlib with no external dep.

    Returns 0 for empty inputs, 1.0 for identical strings. Renaming the
    function would be a breaking change for any external consumer —
    deferred until a v0.2 cleanup pass.
    """
    if not generated or not target:
        return 0.0
    return difflib.SequenceMatcher(None, generated, target).ratio()


#: Hand-curated synonym pairs for ``synonym_swap`` (B11). Picked for
#: high-frequency content words that commonly anchor leading sentences;
#: deliberately small + deterministic so we don't carry a WordNet dep.
_SYNONYM_PAIRS: dict[str, str] = {
    "important": "significant",
    "interesting": "notable",
    "good": "fine",
    "great": "excellent",
    "small": "tiny",
    "large": "big",
    "fast": "quick",
    "slow": "sluggish",
    "begin": "start",
    "end": "finish",
    "show": "demonstrate",
    "use": "employ",
    "make": "create",
    "help": "assist",
    "find": "discover",
    "many": "numerous",
    "few": "several",
    "old": "ancient",
    "new": "recent",
    "true": "valid",
    "false": "incorrect",
    "easy": "simple",
    "hard": "difficult",
    "strong": "robust",
    "weak": "fragile",
    "fact": "truth",
    "idea": "concept",
    "result": "outcome",
    "method": "approach",
    "system": "framework",
    "common": "ordinary",
    "rare": "uncommon",
    "thing": "object",
    "person": "individual",
    "place": "location",
    "time": "moment",
    "way": "manner",
    "work": "labor",
    "study": "examine",
    "know": "understand",
    "think": "consider",
    "say": "state",
    "tell": "inform",
    "ask": "inquire",
    "give": "provide",
    "take": "obtain",
    "see": "observe",
    "look": "view",
    "feel": "sense",
    "want": "desire",
}


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
    if kind == "synonym_swap":
        # Replace the *first table-matching* word in the text. ``re.subn``
        # with ``count=1`` would replace the first regex match regardless,
        # so we walk matches and stop at the first one in the table —
        # otherwise text starting with "The" or "This" would never get
        # perturbed.
        import re as _re

        for match in _re.finditer(r"\b[A-Za-z]+\b", text):
            word = match.group(0)
            replacement = _SYNONYM_PAIRS.get(word.lower())
            if replacement is None:
                continue
            if word[0].isupper():
                replacement = replacement.capitalize()
            return text[: match.start()] + replacement + text[match.end() :]
        return text
    if kind == "clause_reverse":
        # Split on the first comma or " and "; swap the two halves.
        for sep in (", ", " and ", " but ", " or "):
            idx = text.find(sep)
            if idx > 0:
                left, right = text[:idx], text[idx + len(sep) :]
                return f"{right.rstrip('.!?')}{sep}{left}"
        return text
    if kind == "prefix_inject":
        # Prepend a neutral lead-in. The doc text doesn't begin with
        # "I think that" so a memorizing model is forced to reconcile.
        return "I think that " + text[0].lower() + text[1:] if text else text
    if kind == "register_shift":
        # Lower-case the first 30 chars (or upper-case if already lower).
        head, tail = text[:30], text[30:]
        shifted = head.lower() if any(ch.isupper() for ch in head) else head.upper()
        return shifted + tail
    raise ValueError(f"unknown perturbation: {kind!r}")


def _fragility(clean: float, perturbed: float) -> float:
    if clean <= 0.0:
        return 0.0
    return max(0.0, (clean - perturbed) / clean)
