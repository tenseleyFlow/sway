"""B2 ParaphraseInvariance — memorization vs generalization, per case.

For each ``(prompt, gold, paraphrases)`` test case:

- ``verbatim_lift``:  Δ-per-token = logprob_ft(prompt, gold) - logprob_base(prompt, gold)
- ``paraphrase_lift``: mean Δ-per-token over the paraphrased prompts

A model that memorized the exact prompt has high ``verbatim_lift`` but
near-zero ``paraphrase_lift``. A model that learned the underlying
*pattern* has both values positive and close to each other.

We report:

- ``generalization_ratio = paraphrase_lift / max(verbatim_lift, eps)``
- ``verbatim_score``: whether the adapter significantly moved the
  verbatim-prompt logprob (sanity check)

The pass criterion depends on the stated intent: by default we require
both high verbatim lift and high generalization ratio. If the spec's
``intent`` is ``"memorize"``, the ratio requirement inverts — we *want*
verbatim >> paraphrase.
"""

from __future__ import annotations

import statistics
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from dlm_sway.core.result import ProbeResult, Verdict
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext

Intent = Literal["generalize", "memorize", "both"]


class ParaphraseCase(BaseModel):
    """One paraphrase-invariance case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str
    gold: str
    paraphrases: list[str] = Field(default_factory=list, min_length=1)


class ParaphraseInvarianceSpec(ProbeSpec):
    kind: Literal["paraphrase_invariance"] = "paraphrase_invariance"
    cases: list[ParaphraseCase] = Field(default_factory=list)
    intent: Intent = "generalize"
    min_verbatim_lift: float = 0.2
    min_generalization_ratio: float = 0.5
    max_generalization_ratio_if_memorize: float = 0.5


class ParaphraseInvarianceProbe(Probe):
    kind = "paraphrase_invariance"
    spec_cls = ParaphraseInvarianceSpec
    category = "attribution"

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        assert isinstance(spec, ParaphraseInvarianceSpec)
        if not spec.cases:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.ERROR,
                score=None,
                message="no cases provided",
            )

        verbatim_lifts: list[float] = []
        paraphrase_lifts: list[float] = []
        per_case: list[dict[str, float | str]] = []

        for case in spec.cases:
            tokens = max(_token_estimate(case.gold), 1)
            with ctx.backend.as_base() as b:
                lp_base_verb = b.logprob_of(case.prompt, case.gold) / tokens
                lp_base_par = [b.logprob_of(p, case.gold) / tokens for p in case.paraphrases]
            with ctx.backend.as_finetuned() as f:
                lp_ft_verb = f.logprob_of(case.prompt, case.gold) / tokens
                lp_ft_par = [f.logprob_of(p, case.gold) / tokens for p in case.paraphrases]

            verb_lift = lp_ft_verb - lp_base_verb
            par_lift = statistics.fmean(
                (ft - base) for base, ft in zip(lp_base_par, lp_ft_par, strict=True)
            )
            verbatim_lifts.append(verb_lift)
            paraphrase_lifts.append(par_lift)
            per_case.append(
                {
                    "prompt": case.prompt[:80],
                    "verbatim_lift": verb_lift,
                    "paraphrase_lift": par_lift,
                }
            )

        mean_verb = statistics.fmean(verbatim_lifts)
        mean_par = statistics.fmean(paraphrase_lifts)
        ratio = mean_par / mean_verb if abs(mean_verb) > 1e-9 else 0.0

        verdict, score, msg = _decide(spec, mean_verb, mean_par, ratio)

        return ProbeResult(
            name=spec.name,
            kind=spec.kind,
            verdict=verdict,
            score=score,
            raw=ratio,
            base_value=mean_verb,
            ft_value=mean_par,
            evidence={
                "verbatim_lift_mean": mean_verb,
                "paraphrase_lift_mean": mean_par,
                "generalization_ratio": ratio,
                "intent": spec.intent,
                "per_case": per_case[:8],
                "weight": spec.weight,
            },
            message=msg,
        )


def _decide(
    spec: ParaphraseInvarianceSpec, verb: float, par: float, ratio: float
) -> tuple[Verdict, float, str]:
    """Apply the intent-aware pass rule and return (verdict, score, message)."""
    base_msg = f"verb={verb:+.3f}, para={par:+.3f}, ratio={ratio:.2f}"
    if spec.intent == "memorize":
        verd = (
            Verdict.PASS
            if verb >= spec.min_verbatim_lift and ratio <= spec.max_generalization_ratio_if_memorize
            else Verdict.FAIL
        )
        score = min(1.0, max(0.0, verb / max(spec.min_verbatim_lift, 1e-6)))
        return verd, score, f"{base_msg} — intent=memorize"
    # Default: generalize (or "both")
    passed = verb >= spec.min_verbatim_lift and ratio >= spec.min_generalization_ratio
    verd = Verdict.PASS if passed else Verdict.FAIL
    gen_component = min(1.0, max(0.0, ratio / max(spec.min_generalization_ratio, 1e-6)))
    verb_component = min(1.0, max(0.0, verb / max(spec.min_verbatim_lift, 1e-6)))
    score = 0.5 * gen_component + 0.5 * verb_component
    return verd, score, f"{base_msg} — intent={spec.intent}"


def _token_estimate(s: str) -> int:
    return max(1, len(s) // 4)
