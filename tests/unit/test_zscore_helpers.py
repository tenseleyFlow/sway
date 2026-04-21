"""Tests for :mod:`dlm_sway.probes._zscore`."""

from __future__ import annotations

import math

from dlm_sway.core.result import Verdict
from dlm_sway.probes._zscore import (
    MIN_STD,
    no_calibration_note,
    score_from_z,
    verdict_from_z,
    z_score,
)


class TestZScore:
    def test_healthy_path(self) -> None:
        stats = {"mean": 0.01, "std": 0.01, "n": 3.0}
        z = z_score(0.08, stats)
        assert z is not None
        assert math.isclose(z, 7.0)

    def test_raw_below_mean_gives_negative_z(self) -> None:
        stats = {"mean": 0.05, "std": 0.01}
        z = z_score(0.02, stats)
        assert z is not None
        assert z < 0

    def test_none_stats_returns_none(self) -> None:
        assert z_score(0.08, None) is None

    def test_std_below_min_returns_none(self) -> None:
        stats = {"mean": 0.01, "std": 1e-10}
        assert z_score(0.08, stats) is None

    def test_zero_std_returns_none(self) -> None:
        stats = {"mean": 0.01, "std": 0.0}
        assert z_score(0.08, stats) is None

    def test_negative_std_treated_as_invalid(self) -> None:
        stats = {"mean": 0.01, "std": -0.01}
        assert z_score(0.08, stats) is None

    def test_missing_mean_returns_none(self) -> None:
        stats = {"std": 0.01}
        assert z_score(0.08, stats) is None  # type: ignore[arg-type]

    def test_nan_raw_returns_none(self) -> None:
        stats = {"mean": 0.01, "std": 0.01}
        assert z_score(math.nan, stats) is None

    def test_inf_raw_returns_none(self) -> None:
        stats = {"mean": 0.01, "std": 0.01}
        assert z_score(math.inf, stats) is None

    def test_nan_mean_returns_none(self) -> None:
        stats = {"mean": math.nan, "std": 0.01}
        assert z_score(0.08, stats) is None

    def test_min_std_boundary_accepted(self) -> None:
        stats = {"mean": 0.0, "std": MIN_STD}
        z = z_score(1.0, stats)
        assert z is not None

    def test_degenerate_flag_rejects_even_valid_std(self) -> None:
        """F02 (Audit 03) — when null_adapter marks stats as degenerate
        (``runs: 1`` or coincidentally-identical seeds), ``z_score``
        refuses to divide even though the floored std passes MIN_STD.
        This is what prevents the observed ``+290,766σ`` output on a
        ``runs: 1`` leakage probe."""
        stats = {"mean": 0.01, "std": MIN_STD, "degenerate": 1.0}
        assert z_score(0.30, stats) is None

    def test_non_degenerate_flag_does_not_change_behavior(self) -> None:
        """A ``degenerate: 0.0`` marker on an otherwise-valid stats
        dict behaves identically to the no-marker path."""
        stats = {"mean": 0.01, "std": 0.01, "degenerate": 0.0}
        assert z_score(0.08, stats) is not None


class TestVerdictFromZ:
    def test_pass_at_threshold(self) -> None:
        assert verdict_from_z(3.0, threshold=3.0) == Verdict.PASS

    def test_fail_below_threshold(self) -> None:
        assert verdict_from_z(2.99, threshold=3.0) == Verdict.FAIL

    def test_high_z_passes(self) -> None:
        assert verdict_from_z(100.0, threshold=3.0) == Verdict.PASS

    def test_negative_z_fails(self) -> None:
        assert verdict_from_z(-1.0, threshold=3.0) == Verdict.FAIL

    def test_none_z_returns_none(self) -> None:
        assert verdict_from_z(None, threshold=3.0) is None


class TestScoreFromZ:
    def test_z_zero_gives_half(self) -> None:
        s = score_from_z(0.0)
        assert s is not None
        assert math.isclose(s, 0.5)

    def test_z_three_near_optimistic_band(self) -> None:
        s = score_from_z(3.0)
        assert s is not None
        assert 0.7 < s < 0.95

    def test_negative_z_near_zero(self) -> None:
        s = score_from_z(-10.0)
        assert s is not None
        assert s < 0.1

    def test_extreme_positive_clamped(self) -> None:
        """z=+1000 shouldn't overflow math.exp."""
        s = score_from_z(1000.0)
        assert s is not None
        assert 0.99 < s <= 1.0

    def test_extreme_negative_clamped(self) -> None:
        s = score_from_z(-1000.0)
        assert s is not None
        assert 0.0 <= s < 0.01

    def test_none_returns_none(self) -> None:
        assert score_from_z(None) is None


class TestNoCalibrationNote:
    def test_includes_probe_kind(self) -> None:
        note = no_calibration_note("delta_kl")
        assert "delta_kl" in note
        assert "no calibration" in note.lower()
