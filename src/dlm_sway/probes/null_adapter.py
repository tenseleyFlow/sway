"""Null-adapter baseline probe.

Every numeric primitive reports its raw metric *and* a z-score against a
null-adapter distribution. This probe is the runtime engine that
establishes that distribution — it builds random-init "null" adapters
(structurally identical to the real adapter but with weights drawn from
a Gaussian) and measures how much signal they produce.

The resulting ``(mean, std, n)`` per kind is attached to this probe's
``evidence["null_stats"]``. The runner picks it up and threads it into
:attr:`RunContext.null_stats`, where every downstream probe can read it
and turn a raw metric into a z-score.

Backends that don't implement :class:`~dlm_sway.core.scoring.NullCalibratedBackend`
cause this probe to :attr:`Verdict.SKIP` — downstream probes fall back
to their fixed thresholds in that case.
"""

from __future__ import annotations

import statistics
from typing import Literal

from pydantic import Field

from dlm_sway.core.result import ProbeResult, Verdict
from dlm_sway.core.scoring import NullCalibratedBackend
from dlm_sway.probes._divergence import divergence
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext


class NullAdapterSpec(ProbeSpec):
    """Spec for ``kind: null_adapter``.

    Authors place this probe **first** in the suite so its output
    populates :attr:`RunContext.null_stats` before subsequent probes
    consult it.
    """

    kind: Literal["null_adapter"] = "null_adapter"
    runs: int = Field(default=3, ge=1, le=10)
    """Number of independent null adapters to evaluate. Three is the
    smallest that yields a usable std; more is better but quickly
    dominates suite runtime."""
    prompts: list[str] = Field(default_factory=list)
    """Prompt set for null calibration. Keep small — calibration runs
    ``runs × len(prompts)`` forward passes. 4–8 prompts is typical.
    If empty, a minimal built-in prompt set is used so the probe
    always produces stats."""
    init_scale: float = 0.02
    """Stddev of the zero-mean Gaussian used to fill lora_A/lora_B."""
    seed_base: int = 1000
    """First seed; successive runs use ``seed_base + run_idx``."""


_DEFAULT_PROMPTS: tuple[str, ...] = (
    "The quick brown fox",
    "Once upon a time",
    "In this document we explain",
    "The key takeaway is",
    "An important point to remember",
)


class NullAdapterProbe(Probe):
    """Populate ``ctx.null_stats``; report a :attr:`Verdict.PASS` verdict itself.

    The probe never fails on its own terms — its *job* is calibration.
    Downstream probes pick up :attr:`RunContext.null_stats` keyed by
    probe kind (``delta_kl``, ``adapter_ablation`` …) and use the
    populated mean/std to z-score their own raw metrics.
    """

    kind = "null_adapter"
    spec_cls = NullAdapterSpec
    category = "baseline"

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        assert isinstance(spec, NullAdapterSpec)
        if not isinstance(ctx.backend, NullCalibratedBackend):
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.SKIP,
                score=None,
                message=(
                    "backend does not implement NullCalibratedBackend — "
                    "numeric probes will fall back to fixed thresholds"
                ),
            )
        prompts = list(spec.prompts) or list(_DEFAULT_PROMPTS)

        per_seed_means: list[float] = []
        for run_idx in range(spec.runs):
            seed = spec.seed_base + run_idx
            per_prompt: list[float] = []
            for prompt in prompts:
                with ctx.backend.as_base() as base_view:
                    base_dist = base_view.next_token_dist(prompt, top_k=ctx.top_k)
                with ctx.backend.as_null_adapter(seed, init_scale=spec.init_scale) as null_view:
                    null_dist = null_view.next_token_dist(prompt, top_k=ctx.top_k)
                per_prompt.append(divergence(base_dist, null_dist, kind="js"))
            per_seed_means.append(statistics.fmean(per_prompt) if per_prompt else 0.0)

        mean = statistics.fmean(per_seed_means)
        std = statistics.pstdev(per_seed_means) if len(per_seed_means) > 1 else 0.0

        # Publish per-kind stats. delta_kl is the primary kind; other
        # divergence-based probes (adapter_ablation) share this scale.
        null_stats = {
            "delta_kl": {"mean": mean, "std": max(std, 1e-6), "n": float(spec.runs)},
            "adapter_ablation": {"mean": mean, "std": max(std, 1e-6), "n": float(spec.runs)},
        }

        return ProbeResult(
            name=spec.name,
            kind=spec.kind,
            verdict=Verdict.PASS,
            score=1.0,
            raw=mean,
            evidence={
                "null_stats": null_stats,
                "per_seed_mean_js": per_seed_means,
                "init_scale": spec.init_scale,
                "runs": spec.runs,
                "num_prompts": len(prompts),
                "weight": spec.weight,
            },
            message=(
                f"null JS divergence μ={mean:.4f} ± {std:.4f} "
                f"(over {spec.runs} seeds × {len(prompts)} prompts) — "
                f"downstream probes will z-score against this baseline"
            ),
        )


def get_null_stats(ctx: RunContext, probe_kind: str) -> dict[str, float] | None:
    """Look up null-adapter stats for ``probe_kind``.

    Returns ``{"mean": …, "std": …, "n": …}`` when calibration ran for
    this kind, else ``None``. Probes treat ``None`` as "fall back to the
    fixed threshold from your spec."
    """
    return ctx.null_stats.get(probe_kind)
