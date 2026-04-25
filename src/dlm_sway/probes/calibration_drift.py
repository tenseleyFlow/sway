"""C2 CalibrationDrift — did we break general knowledge while fitting the doc?

The classic small-doc fine-tune failure mode: the adapter learned the
document so well that it forgot the world. C2 catches this by scoring
base and ft on a packaged set of general-knowledge completions (the
``BUILT_IN_PACK`` — a 30-item seed of public-domain grade-school facts)
and flagging items whose per-token logprob regressed significantly.

A healthy fine-tune: some items drift slightly (mild confidence shift,
normal), but essentially none regress below a nat of slack. An over-fit
fine-tune: 20%+ of items regress, the adapter has torched its ability
to answer anything outside the document.

Pass when ``fraction_regressed < assert_fraction_regressed_lt`` AND
``mean_delta_nats >= assert_mean_delta_gte``. Both thresholds default
to values that trigger on genuine damage but tolerate normal drift.
"""

from __future__ import annotations

import statistics
from typing import Literal

from pydantic import Field

from dlm_sway.core.result import ProbeResult, Verdict, safe_finalize
from dlm_sway.core.stats import bootstrap_ci
from dlm_sway.probes._calibration_pack import BUILT_IN_PACK
from dlm_sway.probes._zscore import (
    no_calibration_note,
    score_from_z,
    verdict_from_z,
    z_score,
    z_scores_by_rank,
)
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext
from dlm_sway.probes.null_adapter import get_null_stats, get_null_stats_by_rank


class CalibrationItemSpec(ProbeSpec):
    """Not used directly — documents the shape of an item override."""

    kind: Literal["__calibration_item"] = "__calibration_item"
    prompt: str = ""
    gold: str = ""


class CalibrationDriftSpec(ProbeSpec):
    kind: Literal["calibration_drift"] = "calibration_drift"
    pack: Literal["builtin"] = "builtin"
    """Source of items. ``"builtin"`` uses :data:`BUILT_IN_PACK`. Custom
    packs will ship via a file reference in a later milestone."""
    items_limit: int | None = None
    """Truncate the pack to the first ``N`` items (for fast runs).

    The built-in pack is 200 items as of Sprint 05 (B12). Setting
    ``items_limit=30`` reproduces the resolution of the original 30-item
    pack, where each regressed item moves the fraction by ~3.3 pp; full
    200 items moves it by ~0.5 pp, which is the resolution the default
    ``assert_fraction_regressed_lt=0.15`` gate was tuned for."""
    assert_fraction_regressed_lt: float = 0.15
    assert_mean_delta_gte: float = -0.5
    """Mean per-token logprob delta (ft − base) across the pack. Slightly
    negative is tolerable; deeply negative is not."""
    assert_z_gte: float = 3.0
    """Z-score pass criterion against the null-adapter baseline, when it
    exists. This is a lower-is-better probe (we want *fewer* regressions
    than the null adapter), so the z-score is negated before comparison:
    the adapter must be at least ``assert_z_gte`` σ *below* the null
    baseline's ``frac_regressed`` distribution."""
    regression_nats: float = 1.0
    """How many nats worse an item must get to count as regressed."""
    items: list[tuple[str, str]] = Field(default_factory=list)
    """Optional inline override of the packaged items."""


class CalibrationDriftProbe(Probe):
    kind = "calibration_drift"
    spec_cls = CalibrationDriftSpec
    category = "calibration"

    @classmethod
    def calibrate_spec(cls, ctx: RunContext) -> CalibrationDriftSpec | None:
        del ctx
        # Use the built-in pack but capped to keep calibration fast.
        return CalibrationDriftSpec(
            name="_calibration",
            kind="calibration_drift",
            items_limit=10,
        )

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        assert isinstance(spec, CalibrationDriftSpec)
        items = list(spec.items) if spec.items else list(BUILT_IN_PACK)
        if spec.items_limit is not None:
            items = items[: spec.items_limit]
        if not items:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.ERROR,
                score=None,
                message="no calibration items",
            )

        deltas: list[float] = []
        regressed = 0
        worst: list[dict[str, float | str]] = []

        for prompt, gold in items:
            tokens = max(_token_estimate(gold), 1)
            with ctx.require_backend.as_base() as b:
                lp_base = b.logprob_of(prompt, gold) / tokens
            with ctx.require_backend.as_finetuned() as f:
                lp_ft = f.logprob_of(prompt, gold) / tokens
            delta = lp_ft - lp_base
            deltas.append(delta)
            if delta < -spec.regression_nats:
                regressed += 1
                worst.append({"prompt": prompt, "gold": gold, "delta": delta})

        # Surface the worst offenders — up to 5.
        worst.sort(key=lambda d: float(d["delta"]))
        worst = worst[:5]

        frac_regressed = regressed / len(items)
        mean_delta = statistics.fmean(deltas)
        # S14 F9: CI on the fraction — bootstrap over per-item
        # regression indicators (1 if regressed, 0 otherwise). The
        # bootstrap mean = frac_regressed by construction, so the CI
        # brackets the reported ``raw`` value.
        regression_indicators = [1.0 if d < -spec.regression_nats else 0.0 for d in deltas]
        ci_95 = bootstrap_ci(regression_indicators, seed=ctx.seed)

        # Lower-is-better probe: negate the z-score so the shared
        # ``z >= assert_z_gte`` semantics mean "adapter is σ *better*
        # than null".
        stats = get_null_stats(ctx, spec.kind)
        raw_z = z_score(frac_regressed, stats)
        z = -raw_z if raw_z is not None else None
        z_by_rank = z_scores_by_rank(
            frac_regressed, get_null_stats_by_rank(ctx, spec.kind), sign=-1
        )
        verdict_z = verdict_from_z(z, spec.assert_z_gte)
        if verdict_z is not None:
            verdict = verdict_z
            score_val = score_from_z(z)
            score = score_val if score_val is not None else 0.0
            message = (
                f"{regressed}/{len(items)} items regressed "
                f"(frac={frac_regressed:.1%}), mean_delta={mean_delta:+.3f} "
                f"nats/tok, z={z:+.2f}σ vs null"
            )
        else:
            passed = (
                frac_regressed < spec.assert_fraction_regressed_lt
                and mean_delta >= spec.assert_mean_delta_gte
            )
            verdict = Verdict.PASS if passed else Verdict.FAIL
            regress_component = max(
                0.0, 1.0 - frac_regressed / max(spec.assert_fraction_regressed_lt, 1e-6)
            )
            drift_component = max(0.0, min(1.0, (mean_delta + 1.0) / 1.5))
            score = 0.6 * regress_component + 0.4 * drift_component
            message = (
                f"{regressed}/{len(items)} items regressed >{spec.regression_nats:.1f} nats "
                f"(frac={frac_regressed:.1%}), mean_delta={mean_delta:+.3f} nats/tok "
                f"{no_calibration_note(spec.kind)}"
            )

        return safe_finalize(
            name=spec.name,
            kind=spec.kind,
            verdict=verdict,
            score=score,
            raw=frac_regressed,
            z_score=z,
            base_value=None,
            ft_value=mean_delta,
            evidence={
                "fraction_regressed": frac_regressed,
                "mean_delta_nats": mean_delta,
                "regressed_count": regressed,
                "total_items": len(items),
                "worst_offenders": worst,
                "regression_nats_threshold": spec.regression_nats,
                "weight": spec.weight,
                "z_by_rank": z_by_rank,
                "raw_ci_95": list(ci_95) if ci_95 is not None else None,
            },
            message=message,
            ci_95=ci_95,
        )


def _token_estimate(s: str) -> int:
    return max(1, len(s) // 4)
