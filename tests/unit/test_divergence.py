"""Tests for :mod:`dlm_sway.probes._divergence`."""

from __future__ import annotations

import math

import numpy as np

from dlm_sway.core.scoring import TokenDist
from dlm_sway.probes._divergence import aligned_probs, divergence, js, kl


def _dist(ids: list[int], probs: list[float], vocab: int = 100) -> TokenDist:
    return TokenDist(
        token_ids=np.asarray(ids, dtype=np.int64),
        logprobs=np.log(np.asarray(probs, dtype=np.float32)),
        vocab_size=vocab,
    )


class TestAligned:
    def test_identical_distributions(self) -> None:
        d = _dist([1, 2, 3], [0.5, 0.3, 0.2])
        p, q = aligned_probs(d, d)
        np.testing.assert_allclose(p, q)

    def test_union_support_fills_missing(self) -> None:
        base = _dist([1, 2, 3], [0.5, 0.3, 0.2])
        ft = _dist([2, 3, 4], [0.4, 0.4, 0.2])
        p, q = aligned_probs(base, ft)
        assert p.shape == (4,)
        assert abs(p.sum() - 1.0) < 1e-9
        assert abs(q.sum() - 1.0) < 1e-9


class TestKL:
    def test_zero_when_equal(self) -> None:
        p = np.array([0.5, 0.3, 0.2])
        assert kl(p, p) == 0.0

    def test_positive_when_different(self) -> None:
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.2, 0.3, 0.5])
        assert kl(p, q) > 0.0


class TestJS:
    def test_zero_when_equal(self) -> None:
        p = np.array([0.5, 0.3, 0.2])
        assert js(p, p) == 0.0

    def test_symmetric(self) -> None:
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.2, 0.3, 0.5])
        assert math.isclose(js(p, q), js(q, p), rel_tol=1e-9)

    def test_bounded_by_ln2(self) -> None:
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        # With zeros handled as 0·log0 = 0 this approaches ln(2).
        assert js(p, q) <= math.log(2.0) + 1e-9


class TestDivergenceDispatch:
    def test_default_is_js(self) -> None:
        d1 = _dist([1, 2], [0.6, 0.4])
        d2 = _dist([1, 2], [0.3, 0.7])
        assert divergence(d1, d2) == divergence(d1, d2, kind="js")

    def test_kl_available(self) -> None:
        d1 = _dist([1, 2], [0.6, 0.4])
        d2 = _dist([1, 2], [0.3, 0.7])
        assert divergence(d1, d2, kind="kl") >= 0.0
