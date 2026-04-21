"""Shared z-score math for numeric probes.

Every numeric probe computes ``(raw - mean) / std`` against a null-adapter
baseline and converts the z-score to a verdict + normalized score. S02
centralizes this math so probes don't each reinvent it (historical bug:
``delta_kl`` had bespoke z-score code while every other numeric probe
ignored null stats entirely — Audit 01 finding P02).

The helpers here are tiny but load-bearing — they're the one place the
"null calibration won / fixed-threshold fallback" decision is made.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TypedDict

from dlm_sway.core.result import Verdict


class NullStats(TypedDict):
    """Per-kind null-adapter baseline stats published by ``NullAdapterProbe``."""

    mean: float
    std: float
    n: float


#: Minimum ``std`` the z-score path accepts. Below this we treat the null
#: distribution as too degenerate to divide by — probes fall back to the
#: fixed-threshold path rather than emit runaway z-scores.
MIN_STD: float = 1e-6


def z_score(raw: float, stats: Mapping[str, float] | None) -> float | None:
    """Compute ``(raw - mean) / std`` against a null-adapter baseline.

    Returns ``None`` when:

    - ``stats`` is missing (no calibration ran for this kind)
    - ``stats["degenerate"]`` is truthy (F02 Audit 03 — null ran but
      was too narrow to calibrate against: ``runs: 1``, or multi-seed
      raws that collapsed to an effectively-zero variance)
    - ``std`` is below :data:`MIN_STD` (belt-and-suspenders guard
      for stats dicts that predate the ``degenerate`` field)
    - ``raw`` or ``mean`` is non-finite

    Callers that get ``None`` are expected to fall back to their probe's
    fixed-threshold path — and surface ``(no calibration)`` in the
    report so the user knows the z-score path didn't fire.
    """
    if stats is None:
        return None
    mean = stats.get("mean")
    std = stats.get("std", 0.0)
    if mean is None or std is None:
        return None
    if not (math.isfinite(raw) and math.isfinite(mean) and math.isfinite(std)):
        return None
    # ``degenerate`` is stored as a float (1.0 / 0.0) so the stats
    # dict stays Mapping[str, float] across every consumer.
    if stats.get("degenerate", 0.0) >= 0.5:
        return None
    if std < MIN_STD:
        return None
    return float((raw - mean) / std)


def verdict_from_z(z: float | None, threshold: float) -> Verdict | None:
    """Map a z-score to ``PASS``/``FAIL`` against a threshold.

    Returns ``None`` when ``z`` is ``None`` (no calibration) so the
    caller knows to use the fixed-threshold verdict path instead.

    Higher-z-is-better is the convention — the adapter's raw metric
    should be *above* the null distribution for the probe to pass.
    """
    if z is None:
        return None
    return Verdict.PASS if z >= threshold else Verdict.FAIL


def score_from_z(z: float | None) -> float | None:
    """Map a z-score to a normalized ``[0, 1]`` composite score.

    ``sigmoid(z / 3)`` is the shape: z=0 → 0.5, z=3 → ≈0.88, z=-3 → ≈0.12.
    Returns ``None`` when ``z`` is ``None``.

    The /3 divisor centers the knee at "3σ above null" — the convention
    we publish in the README for "the adapter is significantly swayed".
    """
    if z is None:
        return None
    # Guard against extreme z values overflowing math.exp.
    clamped = max(-50.0, min(50.0, z / 3.0))
    return 1.0 / (1.0 + math.exp(-clamped))


def no_calibration_note(probe_kind: str) -> str:
    """The visible annotation probes add to messages when falling back.

    Surfaces in the terminal and markdown reports so users can see which
    probes used fixed thresholds. Matches the string the S02.6 report
    code looks for when formatting rows.
    """
    return f"(no calibration for {probe_kind})"


def z_scores_by_rank(
    raw: float,
    stats_by_rank: Mapping[str, Mapping[str, float]] | None,
    *,
    sign: int = 1,
) -> dict[str, float] | None:
    """Compute per-rank z-scores for a probe's raw metric.

    Parameters
    ----------
    raw:
        The probe's raw metric at the real adapter.
    stats_by_rank:
        ``{rank_key: null_stats}`` from
        :func:`dlm_sway.probes.null_adapter.get_null_stats_by_rank`.
        ``None`` short-circuits to ``None``.
    sign:
        ``+1`` for higher-is-better probes (default), ``-1`` for
        lower-is-better. Applied after the raw z computation so each
        probe keeps its existing sign convention unchanged.

    Returns
    -------
    ``{rank_key: z}`` with only the ranks that produced a finite z
    (divergent std or non-finite inputs drop out silently). ``None``
    when ``stats_by_rank`` is ``None`` or empty.
    """
    if not stats_by_rank:
        return None
    out: dict[str, float] = {}
    for rkey, s in stats_by_rank.items():
        z = z_score(raw, s)
        if z is None:
            continue
        out[rkey] = sign * z
    return out or None


def format_z_profile(z_by_rank: Mapping[str, float] | None) -> str:
    """Render ``{rank_key: z}`` as ``+4.2σ @ 1x / +6.8σ @ 0.5x / +2.1σ @ 2x``.

    Rank labels are rendered as ``{multiplier}x`` (e.g. ``0.5x``) when
    they parse as ``rank_<float>``; anything else is passed through
    verbatim. ``None`` or empty input returns the empty string so
    callers can unconditionally append with ``f"{z} {profile}".rstrip()``.
    """
    if not z_by_rank:
        return ""
    parts: list[str] = []
    for rkey, z in z_by_rank.items():
        if rkey.startswith("rank_"):
            try:
                mult = float(rkey.removeprefix("rank_"))
                label = f"{mult:g}x"
            except ValueError:
                label = rkey
        else:
            label = rkey
        parts.append(f"{z:+.2f}σ @ {label}")
    return " / ".join(parts)
