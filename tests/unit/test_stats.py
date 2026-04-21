"""Tests for :mod:`dlm_sway.core.stats` (S14 / F9)."""

from __future__ import annotations

import math

import numpy as np

from dlm_sway.core.stats import bootstrap_ci


class TestBootstrapCi:
    def test_brackets_the_mean_on_gaussian(self) -> None:
        """With n=100 samples from N(0, 1), the 95% CI brackets the mean
        (0) the overwhelming majority of the time. We seed so the test
        is deterministic; one seed is enough for a regression lock.
        """
        rng = np.random.default_rng(0)
        samples = rng.normal(0.0, 1.0, size=100)
        ci = bootstrap_ci(samples, seed=0)
        assert ci is not None
        lo, hi = ci
        assert lo < samples.mean() < hi
        # Width should be small at n=100 under unit variance — SE of
        # the mean is ~0.1.
        assert hi - lo < 0.6

    def test_degenerate_constant_samples_zero_width(self) -> None:
        """All-identical samples → zero-width CI at the common value.
        The helper short-circuits the bootstrap to avoid RNG noise.
        """
        ci = bootstrap_ci([0.5, 0.5, 0.5, 0.5])
        assert ci == (0.5, 0.5)

    def test_nonfinite_samples_return_none(self) -> None:
        assert bootstrap_ci([1.0, float("nan"), 3.0]) is None
        assert bootstrap_ci([1.0, float("inf"), 3.0]) is None

    def test_empty_returns_none(self) -> None:
        assert bootstrap_ci([]) is None
        assert bootstrap_ci(np.array([], dtype=np.float64)) is None

    def test_confidence_outside_0_1_returns_none(self) -> None:
        assert bootstrap_ci([1.0, 2.0, 3.0], confidence=0.0) is None
        assert bootstrap_ci([1.0, 2.0, 3.0], confidence=1.0) is None
        assert bootstrap_ci([1.0, 2.0, 3.0], confidence=-0.5) is None

    def test_seed_reproducibility(self) -> None:
        samples = [1.2, 3.4, 5.6, 2.1, 4.5, 3.3, 2.8, 4.1]
        ci1 = bootstrap_ci(samples, seed=42)
        ci2 = bootstrap_ci(samples, seed=42)
        assert ci1 == ci2

    def test_seed_differs_produces_different_bounds(self) -> None:
        """Different seeds should give (tiny) bound differences on small n —
        not a correctness test, just a smoke check that the seed is
        actually plumbed into the RNG."""
        samples = [1.2, 3.4, 5.6, 2.1, 4.5, 3.3, 2.8, 4.1]
        ci1 = bootstrap_ci(samples, seed=1)
        ci2 = bootstrap_ci(samples, seed=2)
        # Bounds are close but not identical — each seed samples different indices.
        assert ci1 != ci2

    def test_wider_n_bootstrap_converges(self) -> None:
        """Increasing n_bootstrap tightens the percentile estimates'
        sampling noise (not the CI itself — that depends on sample
        size). Here we just confirm that more resamples don't blow
        up."""
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci_1k = bootstrap_ci(samples, n_bootstrap=1_000, seed=0)
        ci_10k = bootstrap_ci(samples, n_bootstrap=10_000, seed=0)
        assert ci_1k is not None
        assert ci_10k is not None
        # Same order of magnitude.
        assert abs((ci_1k[1] - ci_1k[0]) - (ci_10k[1] - ci_10k[0])) < 0.5

    def test_returns_bounds_are_finite(self) -> None:
        samples = [0.1, 0.2, 0.3, 0.25, 0.15]
        ci = bootstrap_ci(samples)
        assert ci is not None
        lo, hi = ci
        assert math.isfinite(lo)
        assert math.isfinite(hi)
        assert lo <= hi


class TestSafeFinalizeCi:
    """`safe_finalize` threads ci_95 but nulls it when raw gets nulled."""

    def test_ci_preserved_when_raw_finite(self) -> None:
        from dlm_sway.core.result import Verdict, safe_finalize

        result = safe_finalize(
            name="demo",
            kind="delta_kl",
            verdict=Verdict.PASS,
            raw=0.5,
            ci_95=(0.4, 0.6),
        )
        assert result.ci_95 == (0.4, 0.6)

    def test_ci_nulled_when_raw_is_non_finite(self) -> None:
        from dlm_sway.core.result import Verdict, safe_finalize

        result = safe_finalize(
            name="demo",
            kind="delta_kl",
            verdict=Verdict.PASS,
            raw=float("nan"),  # critical field non-finite
            ci_95=(0.4, 0.6),
        )
        assert result.ci_95 is None
        assert result.verdict == Verdict.ERROR  # critical-field guard fires

    def test_ci_none_default(self) -> None:
        from dlm_sway.core.result import Verdict, safe_finalize

        result = safe_finalize(
            name="demo",
            kind="delta_kl",
            verdict=Verdict.PASS,
            raw=0.5,
        )
        assert result.ci_95 is None


class TestProbeEmitsCi95:
    """Smoke: delta_kl on a dummy backend lands a ci_95 that brackets raw."""

    def test_delta_kl_ci_brackets_raw(self) -> None:
        from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
        from dlm_sway.probes.base import RunContext, build_probe

        backend = DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
        probe, spec = build_probe(
            {
                "name": "dk",
                "kind": "delta_kl",
                "prompts": ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"],
            }
        )
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        assert result.ci_95 is not None
        assert result.raw is not None
        lo, hi = result.ci_95
        assert lo <= result.raw <= hi
        # Evidence payload carries the same interval as a list.
        assert result.evidence["raw_ci_95"] == [lo, hi]
