"""Composite :class:`~dlm_sway.core.result.SwayScore` from a suite result.

The score is a weighted mean over four categories
(adherence / attribution / calibration / ablation). Each category's
value is the weighted mean of its pass/score values (with SKIP/ERROR
excluded so a broken probe doesn't silently depress the composite).

All weighting is explicit, user-overridable, and surfaced in the report
alongside the number — no black-box scoring.
"""

from __future__ import annotations

from dlm_sway.core.result import (
    DEFAULT_COMPONENT_WEIGHTS,
    ProbeResult,
    SuiteResult,
    SwayScore,
    Verdict,
)
from dlm_sway.probes.base import registry


def compute(
    suite: SuiteResult,
    *,
    weights: dict[str, float] | None = None,
) -> SwayScore:
    """Fold a :class:`SuiteResult` into a :class:`SwayScore`."""
    w = weights if weights is not None else dict(DEFAULT_COMPONENT_WEIGHTS)
    registered = registry()

    # Bucket probes by their declared category.
    buckets: dict[str, list[ProbeResult]] = {k: [] for k in w}
    for r in suite.probes:
        if r.verdict in {Verdict.SKIP, Verdict.ERROR}:
            continue
        if r.score is None:
            continue
        probe_cls = registered.get(r.kind)
        category = probe_cls.category if probe_cls is not None else "adherence"
        buckets.setdefault(category, []).append(r)

    component_scores: dict[str, float] = {}
    for cat, probes in buckets.items():
        if not probes:
            component_scores[cat] = 0.0
            continue
        total_w = sum(max(_spec_weight(p), 0.0) for p in probes) or 1.0
        weighted = sum(max(_spec_weight(p), 0.0) * (p.score or 0.0) for p in probes)
        component_scores[cat] = weighted / total_w

    # Fold to composite, weighted by the user's category weights, but
    # ignoring components that had no contributing probes (so a
    # PREFERENCE-free document doesn't get penalized for missing B3).
    active_weights = {k: v for k, v in w.items() if buckets.get(k)}
    total_w = sum(active_weights.values()) or 1.0
    overall = sum(active_weights[k] * component_scores[k] for k in active_weights) / total_w

    findings = _findings(suite, component_scores)

    return SwayScore(
        overall=overall,
        components=component_scores,
        weights=w,
        band=SwayScore.band_for(overall),
        findings=findings,
    )


def _spec_weight(result: ProbeResult) -> float:
    """Recover a probe's declared weight from its ``evidence`` payload.

    The runner stores ``spec.weight`` on evidence so the scorer can read
    it without re-validating specs. Falls back to 1.0 when absent (older
    runs, custom probes, etc).
    """
    w = result.evidence.get("weight")
    if isinstance(w, int | float):
        return float(w)
    return 1.0


def _findings(suite: SuiteResult, components: dict[str, float]) -> tuple[str, ...]:
    """Surface the 2–3 most diagnostic notes for the terminal report."""
    notes: list[str] = []

    failed = [r for r in suite.probes if r.verdict == Verdict.FAIL]
    if failed:
        top = failed[0]
        notes.append(
            f"{top.name} ({top.kind}) failed" + (f": {top.message}" if top.message else "")
        )

    for cat, score in components.items():
        if score < 0.3 and components.get(cat, 1.0) != 0.0:
            notes.append(f"{cat} score is {score:.2f} — below the noise threshold")

    errors = [r for r in suite.probes if r.verdict == Verdict.ERROR]
    if errors:
        notes.append(f"{len(errors)} probe(s) errored — see full report for details")

    return tuple(notes[:5])


__all__ = ["compute"]
