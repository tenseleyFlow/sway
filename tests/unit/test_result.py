"""Tests for :mod:`dlm_sway.core.result`."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from dlm_sway.core.result import (
    DEFAULT_COMPONENT_WEIGHTS,
    ProbeResult,
    SuiteResult,
    SwayScore,
    Verdict,
    utcnow,
)


class TestVerdict:
    def test_is_str_enum(self) -> None:
        assert Verdict.PASS.value == "pass"
        assert str(Verdict.WARN.value) == "warn"

    def test_all_expected_members(self) -> None:
        assert {v.value for v in Verdict} == {
            "pass",
            "fail",
            "warn",
            "skip",
            "error",
        }


class TestProbeResult:
    def test_minimum_construction(self) -> None:
        r = ProbeResult(name="t", kind="delta_kl", verdict=Verdict.PASS, score=0.82)
        assert r.raw is None
        assert r.evidence == {}
        assert r.message == ""
        assert r.duration_s == 0.0

    def test_frozen(self) -> None:
        r = ProbeResult(name="t", kind="t", verdict=Verdict.PASS, score=0.5)
        with pytest.raises(FrozenInstanceError):
            r.score = 0.6  # type: ignore[misc]


class TestSuiteResult:
    def test_wall_seconds(self) -> None:
        from datetime import timedelta

        started = utcnow()
        finished = started + timedelta(seconds=2, milliseconds=500)
        result = SuiteResult(
            spec_path="sway.yaml",
            started_at=started,
            finished_at=finished,
            base_model_id="b",
            adapter_id="a",
            sway_version="0.1.0.dev0",
        )
        assert result.wall_seconds == pytest.approx(2.5, abs=1e-6)


class TestSwayScore:
    def test_default_weights_sum_to_one(self) -> None:
        assert abs(sum(DEFAULT_COMPONENT_WEIGHTS.values()) - 1.0) < 1e-9

    def test_band_boundaries(self) -> None:
        assert SwayScore.band_for(0.0) == "noise"
        assert SwayScore.band_for(0.29) == "noise"
        assert SwayScore.band_for(0.30) == "partial"
        assert SwayScore.band_for(0.59) == "partial"
        assert SwayScore.band_for(0.60) == "healthy"
        assert SwayScore.band_for(0.85) == "healthy"
        assert SwayScore.band_for(0.851) == "suspicious"
        assert SwayScore.band_for(0.99) == "suspicious"


def test_utcnow_is_tz_aware() -> None:
    now = utcnow()
    assert now.tzinfo is not None
