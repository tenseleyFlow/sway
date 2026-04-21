"""Tests for :mod:`dlm_sway.probes._divergence`.

Includes property-based tests (`hypothesis`) on the divergence
invariants and explicit tests that non-finite inputs raise
``ProbeError`` rather than producing silent garbage. The latter pin the
S01 fix for the +11639σ bug.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from dlm_sway.core.errors import ProbeError
from dlm_sway.core.scoring import TokenDist
from dlm_sway.probes._divergence import aligned_probs, divergence, js, kl


def _dist(ids: list[int], probs: list[float], vocab: int = 100) -> TokenDist:
    return TokenDist(
        token_ids=np.asarray(ids, dtype=np.int64),
        logprobs=np.log(np.asarray(probs, dtype=np.float32)),
        vocab_size=vocab,
    )


def _normalized_simplex(size: int) -> st.SearchStrategy[np.ndarray]:
    """Hypothesis strategy: arrays of `size` non-negative floats summing to 1."""

    def _norm(a: np.ndarray) -> np.ndarray:
        a = np.abs(a) + 1e-6  # avoid all-zeros
        return a / a.sum()

    return arrays(
        dtype=np.float64,
        shape=size,
        elements=st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False),
    ).map(_norm)


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

    def test_disjoint_top_k_supports_produce_finite_divergence(self) -> None:
        """C12: base peaks on {1,2,3}, ft peaks on {7,8,9}. The aligned
        probabilities have 6 entries (union), with tail-mass redistributed
        to each side's missing tokens via the ``tail_logprob`` fallback.
        Divergence must be finite, positive, and close to ln(2) (the JS
        bound for distributions with disjoint support)."""
        base = TokenDist(
            token_ids=np.asarray([1, 2, 3], dtype=np.int64),
            logprobs=np.log(np.asarray([0.6, 0.25, 0.15], dtype=np.float32)),
            vocab_size=1000,
            # Top-3 covers all of base; tail is effectively zero.
            tail_logprob=float(math.log(1e-9)),
        )
        ft = TokenDist(
            token_ids=np.asarray([7, 8, 9], dtype=np.int64),
            logprobs=np.log(np.asarray([0.5, 0.3, 0.2], dtype=np.float32)),
            vocab_size=1000,
            tail_logprob=float(math.log(1e-9)),
        )
        p, q = aligned_probs(base, ft)
        # Union of {1,2,3} and {7,8,9} is 6 tokens.
        assert p.shape == (6,)
        assert q.shape == (6,)
        # Both distributions should be (approximately) normalized — any
        # tail redistribution leaves a small residual below 1e-3.
        assert abs(p.sum() - 1.0) < 1e-3
        assert abs(q.sum() - 1.0) < 1e-3

        d = divergence(base, ft, kind="js")
        assert math.isfinite(d)
        assert d > 0.0
        # Fully disjoint support → JS approaches its ln(2) upper bound.
        assert d < math.log(2.0) + 1e-6
        assert d > 0.5  # meaningfully above zero


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


class TestNonFiniteRejection:
    """Pins the S01 fix: NaN / inf inputs raise ProbeError, never silent garbage.

    The historical bug: ``np.exp(nan) = nan`` flowed past the ``p > 0``
    mask in ``kl()`` (because ``nan > 0`` is False), producing a
    ``js`` of 13.247 nats — algebraically impossible for JS (≤ ln 2).
    """

    def test_kl_rejects_nan_in_p(self) -> None:
        p = np.array([0.5, math.nan, 0.5])
        q = np.array([0.3, 0.4, 0.3])
        with pytest.raises(ProbeError, match="non-finite"):
            kl(p, q)

    def test_kl_rejects_inf_in_q(self) -> None:
        p = np.array([0.5, 0.5])
        q = np.array([0.5, math.inf])
        with pytest.raises(ProbeError, match="non-finite"):
            kl(p, q)

    def test_js_rejects_nan(self) -> None:
        p = np.array([math.nan, 1.0])
        q = np.array([0.5, 0.5])
        with pytest.raises(ProbeError, match="non-finite"):
            js(p, q)

    def test_aligned_probs_rejects_nan_logprobs(self) -> None:
        bad = TokenDist(
            token_ids=np.array([1, 2], dtype=np.int64),
            logprobs=np.array([-0.5, math.nan], dtype=np.float32),
            vocab_size=100,
        )
        good = _dist([1, 2], [0.5, 0.5])
        with pytest.raises(ProbeError, match="ft TokenDist contains 1 non-finite"):
            aligned_probs(good, bad)
        with pytest.raises(ProbeError, match="base TokenDist contains 1 non-finite"):
            aligned_probs(bad, good)

    def test_divergence_rejects_nan_token_dist(self) -> None:
        bad = TokenDist(
            token_ids=np.array([1, 2], dtype=np.int64),
            logprobs=np.array([math.nan, math.nan], dtype=np.float32),
            vocab_size=100,
        )
        good = _dist([1, 2], [0.5, 0.5])
        with pytest.raises(ProbeError):
            divergence(good, bad)

    def test_js_caps_at_ln2_bound(self) -> None:
        """Defense-in-depth: a hand-rolled p,q where naive computation
        could drift past ln(2) due to FP noise must still raise."""
        # This case is hard to construct intentionally; we instead poison
        # the KL math by constructing a p that's already pathological.
        p = np.array([1.0 - 1e-15, 1e-15])
        q = np.array([1e-15, 1.0 - 1e-15])
        # Real JS here is ≤ ln(2); the function should not raise on
        # well-formed near-extreme distributions.
        result = js(p, q)
        assert 0.0 <= result <= math.log(2.0) + 1e-9


class TestDegenerateUniformRejection:
    """Stronger-test #9 — reject a TokenDist whose top-k logprobs are
    identical. A real model never emits bit-uniform logits; getting
    one means lm_head broke or a fixture zeroed out logits. Silently
    computing ``divergence`` on such a dist returns a trivial constant
    across prompts that would contaminate ``delta_kl`` / ``cluster_kl``.
    """

    def test_perfectly_uniform_dist_is_rejected(self) -> None:
        k = 8
        uniform = TokenDist(
            token_ids=np.arange(k, dtype=np.int64),
            logprobs=np.full(k, -math.log(k), dtype=np.float32),
            vocab_size=1000,
        )
        good = _dist([1, 2], [0.9, 0.1])
        with pytest.raises(ProbeError, match="effectively-uniform"):
            aligned_probs(good, uniform)

    def test_near_uniform_real_model_shape_is_accepted(self) -> None:
        """A broad-but-not-literally-flat dist (the shape a real model
        with high entropy produces) must still compute a divergence."""
        k = 8
        lp = np.full(k, -math.log(k), dtype=np.float32)
        # Tiny monotonic perturbation — enough to clear the 1e-9
        # uniformity threshold without meaningfully changing the
        # entropy.
        lp += np.linspace(-1e-5, 1e-5, k, dtype=np.float32)
        broad = TokenDist(
            token_ids=np.arange(k, dtype=np.int64),
            logprobs=lp,
            vocab_size=1000,
        )
        sharp = TokenDist(
            token_ids=np.arange(k, dtype=np.int64),
            logprobs=np.array([-0.1] + [-5.0] * (k - 1), dtype=np.float32),
            vocab_size=1000,
        )
        # No exception — and KL/JS are finite and positive.
        result = js(*aligned_probs(sharp, broad))
        assert math.isfinite(result)
        assert result > 0.0

    def test_single_token_dist_not_rejected(self) -> None:
        """A distribution with only one token can't be "uniform" —
        there's no spread to compute. The guard must short-circuit."""
        one = TokenDist(
            token_ids=np.array([0], dtype=np.int64),
            logprobs=np.array([0.0], dtype=np.float32),
            vocab_size=1000,
        )
        # Must not raise (``aligned_probs`` handles single-token dists
        # fine; the degenerate check short-circuits at ``size < 2``).
        aligned_probs(one, one)


# ---- Hypothesis property tests ------------------------------------


@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(p=_normalized_simplex(5))
def test_kl_self_is_zero(p: np.ndarray) -> None:
    """KL(p || p) == 0 for any well-formed p."""
    assert kl(p, p) == pytest.approx(0.0, abs=1e-9)


@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(p=_normalized_simplex(5))
def test_js_self_is_zero(p: np.ndarray) -> None:
    """JS(p, p) == 0 for any well-formed p."""
    assert js(p, p) == pytest.approx(0.0, abs=1e-9)


@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(p=_normalized_simplex(5), q=_normalized_simplex(5))
def test_js_symmetric(p: np.ndarray, q: np.ndarray) -> None:
    """JS(p, q) == JS(q, p)."""
    assert js(p, q) == pytest.approx(js(q, p), abs=1e-9)


@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(p=_normalized_simplex(5), q=_normalized_simplex(5))
def test_js_bounded_by_ln2(p: np.ndarray, q: np.ndarray) -> None:
    """JS(p, q) ∈ [0, ln 2] for any pair of distributions."""
    v = js(p, q)
    assert 0.0 <= v <= math.log(2.0) + 1e-9


@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(p=_normalized_simplex(5), q=_normalized_simplex(5))
def test_kl_non_negative(p: np.ndarray, q: np.ndarray) -> None:
    """KL(p || q) ≥ 0 for any pair of distributions (Gibbs' inequality)."""
    assert kl(p, q) >= -1e-9
