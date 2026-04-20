"""A1 DeltaKL — the simplest adherence probe.

For each prompt, compute the JS (default) or KL divergence between the
base and fine-tuned model's next-token distributions at the position
after the prompt. Aggregate across prompts with a mean.

*What it tells you:* whether the adapter is distinguishable from the base
on things the document cares about. A zero-divergence result is a red
flag — the adapter is ignored.

*What it can't tell you:* whether the change is semantically *correct*.
Direction and correctness are what :mod:`dir`, :mod:`adapter_revert`,
and the attribution probes cover.
"""

from __future__ import annotations

import statistics
from typing import Literal

from pydantic import Field

from dlm_sway.core.result import ProbeResult, Verdict
from dlm_sway.probes._divergence import Divergence, divergence, js_ln2
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext
from dlm_sway.probes.null_adapter import get_null_stats


class DeltaKLSpec(ProbeSpec):
    """Spec for ``kind: delta_kl``."""

    kind: Literal["delta_kl"] = "delta_kl"
    prompts: list[str] = Field(default_factory=list, min_length=0)
    """Inline prompts. At least one of ``prompts`` / ``prompts_from`` must
    be non-empty at run time; the prompts-from path is wired via
    :mod:`dlm_sway.integrations.dlm.autogen`."""
    divergence: Divergence = "js"
    top_k: int | None = None
    """Override the suite-wide ``top_k``. ``None`` → use ``ctx.top_k``."""
    assert_mean_gte: float = 0.02
    """Fixed-threshold pass criterion when no null stats are available."""
    assert_z_gte: float = 3.0
    """Z-score pass criterion against the null-adapter baseline, when it
    exists. The more principled metric — prefer this over the raw
    threshold."""


class DeltaKLProbe(Probe):
    """The canonical "is the adapter changing anything?" probe."""

    kind = "delta_kl"
    spec_cls = DeltaKLSpec
    category = "adherence"

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        assert isinstance(spec, DeltaKLSpec)
        if not spec.prompts:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.ERROR,
                score=None,
                message="no prompts provided (inline 'prompts' was empty)",
            )

        top_k = spec.top_k if spec.top_k is not None else ctx.top_k
        divergences: list[float] = []
        for prompt in spec.prompts:
            with ctx.backend.as_base() as base_view:
                base_dist = base_view.next_token_dist(prompt, top_k=top_k)
            with ctx.backend.as_finetuned() as ft_view:
                ft_dist = ft_view.next_token_dist(prompt, top_k=top_k)
            divergences.append(divergence(base_dist, ft_dist, kind=spec.divergence))

        raw_mean = statistics.fmean(divergences)
        raw_max = max(divergences)

        # Null-adapter calibration wins when available.
        null = get_null_stats(ctx, spec.kind)
        z = None
        if null is not None and null.get("std", 0.0) > 0.0:
            z = (raw_mean - null["mean"]) / null["std"]
            verdict = Verdict.PASS if z >= spec.assert_z_gte else Verdict.FAIL
            message = f"mean {spec.divergence}={raw_mean:.4f}, z={z:+.2f}σ vs null"
        else:
            verdict = Verdict.PASS if raw_mean >= spec.assert_mean_gte else Verdict.FAIL
            message = (
                f"mean {spec.divergence}={raw_mean:.4f} "
                f"({'≥' if verdict == Verdict.PASS else '<'} {spec.assert_mean_gte})"
            )

        # Normalized score for composite: JS is bounded by ln(2), so
        # sigmoid-ish on (z, or raw / bound) keeps the number in [0, 1].
        if z is not None:
            score = _sigmoid(z / 3.0)
        else:
            bound = js_ln2() if spec.divergence == "js" else 1.0
            score = min(1.0, raw_mean / bound) if bound > 0.0 else 0.0

        return ProbeResult(
            name=spec.name,
            kind=spec.kind,
            verdict=verdict,
            score=score,
            raw=raw_mean,
            z_score=z,
            evidence={
                "divergence_kind": spec.divergence,
                "per_prompt": divergences,
                "max": raw_max,
                "num_prompts": len(spec.prompts),
                "weight": spec.weight,
            },
            message=message,
        )


def _sigmoid(x: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-x))
