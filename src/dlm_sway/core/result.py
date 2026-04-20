"""Probe and suite result types.

Every numeric probe ultimately returns a :class:`ProbeResult`. The suite
runner collects them into a :class:`SuiteResult` and the scorer folds
that into a single :class:`SwayScore` with transparent per-component
weights.

These dataclasses are deliberately plain — no pydantic — because they
cross probe/backend boundaries hundreds of times per run and a free
``model_validate`` on every construction would dominate the runtime of
cheap probes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class Verdict(StrEnum):
    """Outcome of a single probe against its assertion."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """The result of running one probe.

    Attributes
    ----------
    name:
        User-facing name from the spec (unique within a suite).
    kind:
        Probe discriminator (``delta_kl``, ``section_internalization`` …).
    verdict:
        Pass / fail / warn / skip / error.
    score:
        Normalized [0, 1] score. ``sigmoid(z_vs_null / 3)`` for numeric
        probes; 1.0 / 0.0 for binary ones. ``None`` for :attr:`Verdict.SKIP`.
    raw:
        The raw metric value (e.g. KL=0.083). Probe-specific units.
    z_score:
        Standard deviations above the null-adapter baseline. ``None``
        when no null calibration was run.
    base_value:
        The metric evaluated on the base model, when meaningful.
    ft_value:
        The metric evaluated on the fine-tuned model, when meaningful.
    evidence:
        Small structured payload for the report — prompts, example
        completions, per-section breakdowns. Kept bounded (<10 KB) so
        suite JSON stays under a megabyte.
    message:
        One-line diagnostic. Surfaces in the terminal report.
    duration_s:
        Wall time to execute.
    """

    name: str
    kind: str
    verdict: Verdict
    score: float | None
    raw: float | None = None
    z_score: float | None = None
    base_value: float | None = None
    ft_value: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    duration_s: float = 0.0


@dataclass(frozen=True, slots=True)
class SuiteResult:
    """A full run of a sway.yaml suite."""

    spec_path: str
    started_at: datetime
    finished_at: datetime
    base_model_id: str
    adapter_id: str
    sway_version: str
    probes: tuple[ProbeResult, ...] = ()
    null_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    """Per-primitive null-adapter baseline stats (mean, std, runs). Used
    to turn raw metrics into z-scores when rendering the report."""

    @property
    def wall_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()


# Component weights for the composite score. Overridable in sway.yaml.
DEFAULT_COMPONENT_WEIGHTS: dict[str, float] = {
    "adherence": 0.30,
    "attribution": 0.35,
    "calibration": 0.20,
    "ablation": 0.15,
}


@dataclass(frozen=True, slots=True)
class SwayScore:
    """Composite score with a transparent per-component breakdown."""

    overall: float
    components: dict[str, float]
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_COMPONENT_WEIGHTS))
    band: str = ""
    findings: tuple[str, ...] = ()

    @staticmethod
    def band_for(overall: float) -> str:
        """Map a score to a human-readable band.

        Bands (from the plan):
          - <0.3  : indistinguishable from noise
          - 0.3–0.6 : partial fit
          - 0.6–0.85: healthy
          - >0.85 : suspiciously good (possible overfit / memorization)
        """
        if overall < 0.3:
            return "noise"
        if overall < 0.6:
            return "partial"
        if overall <= 0.85:
            return "healthy"
        return "suspicious"


def utcnow() -> datetime:
    """Timezone-aware UTC timestamp (used by the runner)."""
    return datetime.now(UTC)
