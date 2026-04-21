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

import math
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
class DeterminismReport:
    """Serializable view of what seeding the runner accomplished.

    Mirrors :class:`dlm_sway.core.determinism.DeterminismSummary` but
    lives here so :class:`SuiteResult` doesn't pull ``determinism`` as
    an import-time dependency of ``core.result``.
    """

    class_: str
    seed: int
    notes: tuple[str, ...] = ()


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
    determinism: DeterminismReport | None = None
    """Classification of the determinism regime the suite ran under, from
    :func:`dlm_sway.core.determinism.seed_everything`. ``None`` when the
    caller bypassed seeding (e.g., unit tests constructing a
    ``SuiteResult`` directly)."""
    backend_stats: dict[str, float | int] = field(default_factory=dict)
    """Forward-pass + cache counters from the backend
    (:class:`dlm_sway.backends._instrumentation.BackendStats.to_dict`).
    Populated by the runner at suite-end; ``{}`` when the backend
    doesn't expose instrumentation (custom backends, pre-S07 snapshots)."""

    @property
    def wall_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()


# Component weights for the composite score. Overridable in sway.yaml.
# ``baseline`` is listed with weight 0.0 so the null-calibration row
# appears in the report for transparency but contributes nothing to the
# composite — it's an informational category, not a judgment one.
DEFAULT_COMPONENT_WEIGHTS: dict[str, float] = {
    "adherence": 0.30,
    "attribution": 0.35,
    "calibration": 0.20,
    "ablation": 0.15,
    "baseline": 0.0,
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


def safe_finalize(
    *,
    name: str,
    kind: str,
    verdict: Verdict,
    score: float | None = None,
    raw: float | None = None,
    z_score: float | None = None,
    base_value: float | None = None,
    ft_value: float | None = None,
    evidence: dict[str, Any] | None = None,
    message: str = "",
    duration_s: float = 0.0,
    critical_fields: tuple[str, ...] = ("raw",),
) -> ProbeResult:
    """Build a :class:`ProbeResult` with defense against non-finite metrics.

    Probes hand their candidate result kwargs here instead of constructing
    a :class:`ProbeResult` directly. The helper inspects every numeric
    field and classifies it:

    - **Critical field non-finite** (any field named in ``critical_fields``
      whose value is ``NaN`` or ``±inf``): the whole probe result is
      converted to :attr:`Verdict.ERROR` with all scalar fields nulled out,
      the offending values are preserved under
      ``evidence["non_finite_inputs"]``, and the message explains which
      field(s) were non-finite.
    - **Non-critical field non-finite**: nulled out silently (set to
      ``None``), and the field name appended to
      ``evidence["defensively_nulled"]`` so a report reader can see what
      happened.
    - **Everything finite**: passthrough, no change.

    The default ``critical_fields = ("raw",)`` reflects the design stance:
    ``raw`` is the probe's ground-truth metric; a non-finite ``raw`` means
    the probe cannot make a meaningful statement. Probes that care about
    other fields (e.g., probes whose ``z_score`` is load-bearing) pass a
    broader tuple.

    This helper is the single shared guardrail sprint 01 installs against
    the +11639σ class of bug, where NaN logprobs flowed silently through
    to a PASS verdict. Every numeric probe is expected to finalize through
    this function.
    """
    numeric_kwargs: dict[str, float | None] = {
        "score": score,
        "raw": raw,
        "z_score": z_score,
        "base_value": base_value,
        "ft_value": ft_value,
    }

    non_finite: dict[str, float] = {}
    for fname, v in numeric_kwargs.items():
        if isinstance(v, int | float) and not isinstance(v, bool) and not math.isfinite(float(v)):
            non_finite[fname] = float(v)

    ev: dict[str, Any] = dict(evidence) if evidence is not None else {}

    critical_non_finite = {k: v for k, v in non_finite.items() if k in critical_fields}
    if critical_non_finite:
        ev["non_finite_inputs"] = non_finite
        return ProbeResult(
            name=name,
            kind=kind,
            verdict=Verdict.ERROR,
            score=None,
            raw=None,
            z_score=None,
            base_value=None,
            ft_value=None,
            evidence=ev,
            message=(
                f"non-finite critical field(s): {', '.join(sorted(critical_non_finite))} "
                f"— probe cannot produce a meaningful result"
            ),
            duration_s=duration_s,
        )

    if non_finite:
        ev.setdefault("defensively_nulled", []).extend(sorted(non_finite))
        for fname in non_finite:
            numeric_kwargs[fname] = None

    return ProbeResult(
        name=name,
        kind=kind,
        verdict=verdict,
        score=numeric_kwargs["score"],
        raw=numeric_kwargs["raw"],
        z_score=numeric_kwargs["z_score"],
        base_value=numeric_kwargs["base_value"],
        ft_value=numeric_kwargs["ft_value"],
        evidence=ev,
        message=message,
        duration_s=duration_s,
    )
