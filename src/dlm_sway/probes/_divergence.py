"""Shared math for divergence-based probes.

Extracted so :mod:`delta_kl`, :mod:`adapter_ablation`, and any future
probe operating on next-token distributions reuse the same aligned-
top-k KL / JS computation. Having one implementation keeps the numerical
treatment consistent across the report.

**Non-finite policy** (S01): every entry point in this module rejects
non-finite inputs *explicitly* by raising :class:`ProbeError`. The
historical bug — ``np.exp(nan) = nan`` flowing past a ``p > 0`` mask
that evaluates ``nan > 0`` as False — produced a mathematically
impossible JS divergence of 13.247 nats (bounded by ln 2 ≈ 0.693).
We refuse to compute on non-finite inputs rather than silently produce
garbage; the calling probe routes the resulting :class:`ProbeError` to
:attr:`Verdict.ERROR` via :func:`safe_finalize`.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from dlm_sway.core.errors import ProbeError
from dlm_sway.core.scoring import TokenDist

Divergence = Literal["kl", "js"]


def _check_finite_token_dist(name: str, dist: TokenDist) -> None:
    """Reject a TokenDist whose logprobs contain NaN/inf.

    Called at the entry to ``aligned_probs``. The error message names the
    side (base / ft) so a probe failure pinpoints the broken model.
    """
    if not np.all(np.isfinite(dist.logprobs)):
        n_bad = int(np.sum(~np.isfinite(dist.logprobs)))
        raise ProbeError(
            "divergence",
            f"{name} TokenDist contains {n_bad} non-finite logprob(s) — "
            f"refusing to compute divergence on a model that produces "
            f"NaN/inf logits",
        )


def aligned_probs(
    base: TokenDist, ft: TokenDist
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return aligned probability vectors over the union of top-k tokens.

    Two ``TokenDist`` objects may surface different top-k indices if
    the two models disagree about the hot tokens. We build a shared
    support — ``union(base.token_ids, ft.token_ids)`` — and slot the
    known probabilities in. Unknown entries fall back to the
    per-distribution tail mass divided across the missing tokens,
    which is the maximum-entropy completion under the truncation.

    Raises :class:`ProbeError` when either input contains non-finite
    logprobs.
    """
    _check_finite_token_dist("base", base)
    _check_finite_token_dist("ft", ft)

    union_ids = np.union1d(base.token_ids, ft.token_ids)
    k = int(union_ids.size)

    base_probs = _to_support(base, union_ids, k)
    ft_probs = _to_support(ft, union_ids, k)

    # Normalize in case of floating noise from the fill-in.
    base_total = float(base_probs.sum())
    ft_total = float(ft_probs.sum())
    if not (math.isfinite(base_total) and base_total > 0):
        raise ProbeError(
            "divergence", f"base distribution sums to {base_total} after support alignment"
        )
    if not (math.isfinite(ft_total) and ft_total > 0):
        raise ProbeError(
            "divergence", f"ft distribution sums to {ft_total} after support alignment"
        )
    base_probs /= base_total
    ft_probs /= ft_total
    return base_probs, ft_probs


def _to_support(dist: TokenDist, support: NDArray[np.int64], k: int) -> NDArray[np.float64]:
    probs = np.exp(dist.logprobs.astype(np.float64))
    out = np.zeros(k, dtype=np.float64)
    known_mass = float(probs.sum())
    tail_mass = max(0.0, 1.0 - known_mass)

    id_to_idx = {int(tok): idx for idx, tok in enumerate(support.tolist())}
    missing = 0
    for tok, p in zip(dist.token_ids.tolist(), probs.tolist(), strict=True):
        i = id_to_idx.get(int(tok))
        if i is None:
            # Shouldn't happen given union construction.
            missing += 1
            continue
        out[i] = float(p)

    # Spread the tail mass over the support entries that this dist
    # doesn't explicitly provide. Size of that set:
    n_unknown = int((out == 0.0).sum()) - missing
    if n_unknown > 0 and tail_mass > 0.0:
        per = tail_mass / n_unknown
        out[out == 0.0] = per

    return out


def _check_finite_array(name: str, arr: NDArray[np.float64]) -> None:
    """Reject an array containing NaN/inf."""
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.sum(~np.isfinite(arr)))
        raise ProbeError(
            "divergence",
            f"{name} contains {n_bad} non-finite entry/entries — refusing to compute divergence",
        )


def kl(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """KL(p || q) in nats. Robust to zeros in p (treated as 0·log0 = 0).

    Raises :class:`ProbeError` on non-finite inputs.
    """
    _check_finite_array("p", p)
    _check_finite_array("q", q)
    mask = p > 0.0
    safe_q = np.where(q > 0.0, q, 1e-12)
    result = float(np.sum(p[mask] * (np.log(p[mask]) - np.log(safe_q[mask]))))
    if not math.isfinite(result):
        raise ProbeError("divergence", f"kl computation produced non-finite result: {result}")
    return result


def js(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Jensen-Shannon divergence. Symmetric, bounded in [0, ln 2] (nats).

    The upper bound makes JS a nicer default for thresholding than raw
    KL — a user doesn't need to know their specific model's KL scale to
    pick a threshold.

    Raises :class:`ProbeError` on non-finite inputs or non-finite output.
    """
    _check_finite_array("p", p)
    _check_finite_array("q", q)
    m = 0.5 * (p + q)
    result = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    # Defense-in-depth: clamp into the theoretical bound. JS ∈ [0, ln 2].
    # Tiny negative or just-over-ln2 values are FP roundoff (especially
    # when p ≈ q); broader excursions indicate upstream numerical drift,
    # surface them as ProbeError rather than silently exceeding ln 2.
    tol = 1e-9
    if result < -tol or result > math.log(2.0) + tol:
        raise ProbeError(
            "divergence",
            f"js computed {result:.4f} nats, outside theoretical bound "
            f"[0, {math.log(2.0):.4f}] — likely numerical drift in upstream "
            f"distributions",
        )
    return max(0.0, result)


def divergence(base: TokenDist, ft: TokenDist, kind: Divergence = "js") -> float:
    """Compute KL or JS between two ``TokenDist`` on a shared support."""
    p, q = aligned_probs(base, ft)
    if kind == "js":
        return js(p, q)
    if kind == "kl":
        return kl(q, p)  # KL(ft || base) — "how much does ft diverge from base"
    raise ValueError(f"unknown divergence kind: {kind!r}")


def js_ln2() -> float:
    """Upper bound on JS in nats. Useful for normalization."""
    return math.log(2.0)
