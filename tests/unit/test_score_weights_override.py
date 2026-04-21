"""Tests for score_weights overrides via spec field + CLI flag."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dlm_sway.cli.commands import _parse_weights_flag
from dlm_sway.core.result import DEFAULT_COMPONENT_WEIGHTS
from dlm_sway.suite.spec import SuiteDefaults


class TestSuiteDefaultsScoreWeights:
    def test_none_means_use_defaults(self) -> None:
        d = SuiteDefaults()
        assert d.score_weights is None

    def test_partial_override_inherits_defaults(self) -> None:
        d = SuiteDefaults(score_weights={"adherence": 0.5})
        assert d.score_weights is not None
        # The partial override is merged with the defaults.
        assert d.score_weights["adherence"] == 0.5
        assert d.score_weights["attribution"] == DEFAULT_COMPONENT_WEIGHTS["attribution"]
        assert d.score_weights["calibration"] == DEFAULT_COMPONENT_WEIGHTS["calibration"]
        assert d.score_weights["ablation"] == DEFAULT_COMPONENT_WEIGHTS["ablation"]
        assert d.score_weights["baseline"] == DEFAULT_COMPONENT_WEIGHTS["baseline"]

    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown category keys"):
            SuiteDefaults(score_weights={"bogus": 0.5})

    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(ValidationError, match="non-negative"):
            SuiteDefaults(score_weights={"adherence": -0.1})

    def test_all_zero_weights_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least one positive weight"):
            SuiteDefaults(
                score_weights={
                    "adherence": 0.0,
                    "attribution": 0.0,
                    "calibration": 0.0,
                    "ablation": 0.0,
                    "baseline": 0.0,
                }
            )


class TestParseWeightsFlag:
    def test_none_input_returns_none(self) -> None:
        assert _parse_weights_flag(None) is None
        assert _parse_weights_flag("") is None

    def test_single_pair(self) -> None:
        assert _parse_weights_flag("adherence=0.4") == {"adherence": 0.4}

    def test_multiple_pairs(self) -> None:
        out = _parse_weights_flag("adherence=0.4,attribution=0.3,ablation=0.1")
        assert out == {"adherence": 0.4, "attribution": 0.3, "ablation": 0.1}

    def test_whitespace_tolerated(self) -> None:
        out = _parse_weights_flag(" adherence=0.4 , attribution=0.3 ")
        assert out == {"adherence": 0.4, "attribution": 0.3}

    def test_missing_equals_raises(self) -> None:
        import typer

        with pytest.raises(typer.BadParameter, match="key=value"):
            _parse_weights_flag("adherence0.4")

    def test_non_numeric_value_raises(self) -> None:
        import typer

        with pytest.raises(typer.BadParameter, match="not a number"):
            _parse_weights_flag("adherence=high")


class TestComputeWithOverride:
    def test_overridden_weights_change_composite(self) -> None:
        """A re-weighted compose should change the overall score."""
        from datetime import UTC, datetime

        from dlm_sway.core.result import ProbeResult, SuiteResult, Verdict
        from dlm_sway.suite.score import compute as compute_score

        now = datetime.now(UTC)
        # Two probes: one perfect adherence, one zero attribution.
        probes = (
            ProbeResult(
                name="dk",
                kind="delta_kl",
                verdict=Verdict.PASS,
                score=1.0,
                evidence={"weight": 1.0},
            ),
            ProbeResult(
                name="sis",
                kind="section_internalization",
                verdict=Verdict.FAIL,
                score=0.0,
                evidence={"weight": 1.0},
            ),
        )
        suite = SuiteResult(
            spec_path="<test>",
            started_at=now,
            finished_at=now,
            base_model_id="b",
            adapter_id="a",
            sway_version="0.0.0",
            probes=probes,
        )

        # Default weights skew toward attribution (0.35 vs 0.30).
        default_score = compute_score(suite)
        # Override to favor adherence.
        skewed_score = compute_score(
            suite,
            weights={
                "adherence": 0.9,
                "attribution": 0.05,
                "calibration": 0.025,
                "ablation": 0.025,
                "baseline": 0.0,
            },
        )
        assert skewed_score.overall > default_score.overall, (
            f"expected skewed_score > default; got skewed={skewed_score.overall:.3f}, "
            f"default={default_score.overall:.3f}"
        )
