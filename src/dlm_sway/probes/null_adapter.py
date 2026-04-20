"""Null-adapter baseline probe.

Every numeric primitive reports its raw metric *and* a z-score against a
null-adapter distribution. This probe is the runtime engine that
establishes that distribution — running each configured primitive
against a series of random-init-style "null" adapters (structurally
identical to the real adapter but with weights indistinguishable from
noise) and caching the resulting ``(mean, std, n)`` per primitive kind.

The heavy lifting — materializing random-init LoRAs on the loaded model
and running probes with them — lives in the HF backend (later
milestone). For now this module ships the spec + the lookup API that
probes will use to z-score their results once stats are populated.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from dlm_sway.core.result import ProbeResult, Verdict
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext


class NullAdapterSpec(ProbeSpec):
    """Spec for ``kind: null_adapter``.

    This is a meta-probe: it doesn't test the adapter, it calibrates
    *other* probes. Place it first in the suite so its output is in
    :attr:`~dlm_sway.probes.base.RunContext.null_stats` when later
    probes run.
    """

    kind: Literal["null_adapter"] = "null_adapter"
    runs: int = Field(default=3, ge=1, le=10)
    """Number of independent null adapters to evaluate. Three is the
    smallest that gives a usable std estimate; more is better but quickly
    dominates suite runtime."""
    rank: int | None = None
    """LoRA rank for the null adapter. ``None`` → match the real adapter."""
    alpha: int | None = None
    """LoRA alpha. ``None`` → match the real adapter."""
    init_scale: float = 0.02
    """Standard deviation of the zero-mean Gaussian used to init
    lora_A/lora_B. Matches typical post-init scale."""


class NullAdapterProbe(Probe):
    """Populate ``ctx.null_stats``; report a :attr:`Verdict.SKIP` verdict itself.

    The probe never fails on its own terms — its *job* is calibration,
    not judgment. Downstream probes consult
    :meth:`get_null_stats` to turn their raw metric into a z-score.
    """

    kind = "null_adapter"
    spec_cls = NullAdapterSpec
    category = "baseline"

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        # Concrete null-adapter materialization is backend-specific. For
        # the HF backend it will build random-init LoRAs with matched
        # rank/alpha. That path is wired in a later milestone; this probe
        # currently reports SKIP so suite composition stays stable.
        del ctx  # unused until HF-level materialization lands
        assert isinstance(spec, NullAdapterSpec)
        return ProbeResult(
            name=spec.name,
            kind=spec.kind,
            verdict=Verdict.SKIP,
            score=None,
            message=(
                "null-adapter calibration pending — downstream probes will fall back to "
                "fixed thresholds until the backend-level materialization lands"
            ),
            evidence={"runs": spec.runs, "rank": spec.rank, "alpha": spec.alpha},
        )


def get_null_stats(ctx: RunContext, probe_kind: str) -> dict[str, float] | None:
    """Look up null-adapter stats for ``probe_kind``.

    Returns ``{"mean": …, "std": …, "n": …}`` when calibration ran for
    this kind, else ``None``. Probes should treat ``None`` as "fall back
    to the fixed threshold from your spec."
    """
    return ctx.null_stats.get(probe_kind)
