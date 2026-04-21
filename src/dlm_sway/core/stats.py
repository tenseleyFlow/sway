"""Tiny statistics helpers for numeric probes (S14 / F9).

Bootstrap resampling is the shipping pattern. Every numeric probe
that aggregates over a handful of prompts (4–32 is the common range)
has real sampling noise on its point estimate — a single `mean =
0.083` number hides the fact that the underlying prompt set could
have given anywhere from 0.06 to 0.10 with a different prompt
selection.

:func:`bootstrap_ci` samples-with-replacement from the given array
``n_bootstrap`` times, takes the mean of each resample, and returns
the ``(lower, upper)`` percentile bounds for the requested
confidence level. Deterministic via the ``seed`` arg so two runs of
the same suite produce the same CI.

Intentionally *not* a full statistics module — just the one helper
the aggregating probes need. Bayesian CIs, studentized bootstraps,
BCa corrections are out of scope; bootstrap-percentile is the MVP
the audit's F9 entry called out.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def bootstrap_ci(
    samples: Sequence[float] | NDArray[np.float64],
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Percentile bootstrap confidence interval for the mean.

    Parameters
    ----------
    samples:
        Per-sample measurements (e.g. the list of per-prompt
        divergences inside ``delta_kl``). All values must be finite
        — non-finite inputs short-circuit to ``None`` so probes can
        fold the CI path without pre-validating.
    n_bootstrap:
        Resample count. 1000 is the sweet spot for percentile
        bootstraps: wider resamples (10k) give slightly tighter
        intervals at 10× the cost; fewer (100) start to show
        percentile discretization on small ``n``.
    confidence:
        Two-sided coverage. ``0.95`` → returns the 2.5th and 97.5th
        percentiles of the bootstrap distribution.
    seed:
        Governs the resample RNG. Threaded from ``ctx.seed`` so the
        CI is reproducible across runs of the same suite.

    Returns
    -------
    ``(lower, upper)`` floats. ``None`` when:

    - ``samples`` is empty
    - any sample is non-finite
    - ``confidence`` is outside ``(0, 1)``

    Notes
    -----
    Percentile bootstrap — doesn't correct for skew (BCa would) but
    is accurate enough for the 4–32-sample range probes operate in
    and has no knobs users could get wrong.
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    if not (0.0 < confidence < 1.0):
        return None

    # Degenerate (all-identical) input → zero-width CI at the common
    # value. Skip the resample work entirely — saves a few ms per
    # probe when the adapter happens to produce constant per-prompt
    # divergence, and avoids any RNG-dependent rounding noise on the
    # bounds.
    if float(arr.min()) == float(arr.max()):
        v = float(arr[0])
        return (v, v)

    rng = np.random.default_rng(seed)
    n = arr.size
    # Vectorized resample: one (n_bootstrap, n) index matrix; take
    # means along axis 1. Memory = 8 * n_bootstrap * n bytes —
    # ~128 KB at default settings with n=16. Fits in L2.
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    resampled_means = arr[idx].mean(axis=1)

    alpha = 1.0 - confidence
    lo_p = (alpha / 2.0) * 100.0
    hi_p = (1.0 - alpha / 2.0) * 100.0
    lo, hi = np.percentile(resampled_means, [lo_p, hi_p])
    lo_f, hi_f = float(lo), float(hi)
    if not (math.isfinite(lo_f) and math.isfinite(hi_f)):
        return None
    return (lo_f, hi_f)


__all__ = ["bootstrap_ci"]
