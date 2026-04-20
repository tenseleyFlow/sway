"""Shared math for divergence-based probes.

Extracted so :mod:`delta_kl`, :mod:`adapter_ablation`, and any future
probe operating on next-token distributions reuse the same aligned-
top-k KL / JS computation. Having one implementation keeps the numerical
treatment consistent across the report.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from dlm_sway.core.scoring import TokenDist

Divergence = Literal["kl", "js"]


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
    """
    union_ids = np.union1d(base.token_ids, ft.token_ids)
    k = int(union_ids.size)

    base_probs = _to_support(base, union_ids, k)
    ft_probs = _to_support(ft, union_ids, k)

    # Normalize in case of floating noise from the fill-in.
    base_probs /= base_probs.sum()
    ft_probs /= ft_probs.sum()
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


def kl(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """KL(p || q) in nats. Robust to zeros in p (treated as 0·log0 = 0)."""
    mask = p > 0.0
    safe_q = np.where(q > 0.0, q, 1e-12)
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(safe_q[mask]))))


def js(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Jensen-Shannon divergence. Symmetric, bounded in [0, ln 2] (nats).

    The upper bound makes JS a nicer default for thresholding than raw
    KL — a user doesn't need to know their specific model's KL scale to
    pick a threshold.
    """
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


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
