"""Tests for :func:`dlm_sway.core.result.safe_finalize`.

This helper is the shared guardrail S01 installs against NaN-flows-through
bugs. It must:

- Route critical non-finite fields to :attr:`Verdict.ERROR` with score nulled
- Defensively null non-critical non-finite fields without changing the verdict
- Leave all-finite inputs untouched
- Preserve the original non-finite values in evidence for postmortem
"""

from __future__ import annotations

import math

from dlm_sway.core.result import ProbeResult, Verdict, safe_finalize


class TestAllFinite:
    def test_passthrough_preserves_all_fields(self) -> None:
        r = safe_finalize(
            name="p1",
            kind="delta_kl",
            verdict=Verdict.PASS,
            score=0.75,
            raw=0.08,
            z_score=3.2,
            base_value=0.0,
            ft_value=0.08,
            evidence={"num_prompts": 4},
            message="looks fine",
            duration_s=1.2,
        )
        assert r.verdict == Verdict.PASS
        assert r.score == 0.75
        assert r.raw == 0.08
        assert r.z_score == 3.2
        assert r.base_value == 0.0
        assert r.ft_value == 0.08
        assert r.message == "looks fine"
        assert r.duration_s == 1.2
        assert r.evidence == {"num_prompts": 4}

    def test_defaults(self) -> None:
        r = safe_finalize(name="p", kind="k", verdict=Verdict.PASS, score=1.0)
        assert r.raw is None
        assert r.z_score is None
        assert r.evidence == {}
        assert r.duration_s == 0.0


class TestCriticalNonFinite:
    def test_nan_raw_routes_to_error(self) -> None:
        r = safe_finalize(
            name="p",
            kind="delta_kl",
            verdict=Verdict.PASS,
            score=1.0,
            raw=math.nan,
            z_score=3.0,
        )
        assert r.verdict == Verdict.ERROR
        assert r.score is None
        assert r.raw is None
        assert r.z_score is None
        assert "non-finite critical" in r.message
        assert "raw" in r.message
        assert "raw" in r.evidence["non_finite_inputs"]
        assert math.isnan(r.evidence["non_finite_inputs"]["raw"])

    def test_inf_raw_routes_to_error(self) -> None:
        r = safe_finalize(
            name="p",
            kind="delta_kl",
            verdict=Verdict.PASS,
            score=1.0,
            raw=math.inf,
        )
        assert r.verdict == Verdict.ERROR
        assert r.evidence["non_finite_inputs"]["raw"] == math.inf

    def test_negative_inf_raw_routes_to_error(self) -> None:
        r = safe_finalize(
            name="p",
            kind="delta_kl",
            verdict=Verdict.PASS,
            score=1.0,
            raw=-math.inf,
        )
        assert r.verdict == Verdict.ERROR

    def test_error_capture_includes_all_non_finite_fields(self) -> None:
        """Even non-critical fields that are non-finite are recorded in evidence."""
        r = safe_finalize(
            name="p",
            kind="delta_kl",
            verdict=Verdict.PASS,
            score=1.0,
            raw=math.nan,
            z_score=math.inf,
            base_value=math.nan,
        )
        assert r.verdict == Verdict.ERROR
        captured = r.evidence["non_finite_inputs"]
        assert set(captured) == {"raw", "z_score", "base_value"}

    def test_error_preserves_caller_evidence_keys(self) -> None:
        r = safe_finalize(
            name="p",
            kind="delta_kl",
            verdict=Verdict.PASS,
            score=1.0,
            raw=math.nan,
            evidence={"per_prompt": [1, 2, 3], "num_prompts": 3},
        )
        assert r.verdict == Verdict.ERROR
        assert r.evidence["per_prompt"] == [1, 2, 3]
        assert r.evidence["num_prompts"] == 3
        assert "non_finite_inputs" in r.evidence


class TestNonCriticalNonFinite:
    def test_nan_z_score_is_nulled_silently(self) -> None:
        r = safe_finalize(
            name="p",
            kind="delta_kl",
            verdict=Verdict.PASS,
            score=0.7,
            raw=0.05,
            z_score=math.nan,
        )
        assert r.verdict == Verdict.PASS
        assert r.score == 0.7
        assert r.raw == 0.05
        assert r.z_score is None
        assert "z_score" in r.evidence["defensively_nulled"]

    def test_nan_base_and_ft_nulled_preserves_passing_score(self) -> None:
        r = safe_finalize(
            name="p",
            kind="delta_kl",
            verdict=Verdict.PASS,
            score=0.9,
            raw=0.1,
            base_value=math.nan,
            ft_value=math.inf,
        )
        assert r.verdict == Verdict.PASS
        assert r.base_value is None
        assert r.ft_value is None
        assert sorted(r.evidence["defensively_nulled"]) == ["base_value", "ft_value"]


class TestCriticalFieldsOverride:
    def test_z_score_critical_triggers_error_on_nan(self) -> None:
        r = safe_finalize(
            name="p",
            kind="adapter_ablation",
            verdict=Verdict.PASS,
            score=1.0,
            raw=0.9,
            z_score=math.nan,
            critical_fields=("raw", "z_score"),
        )
        assert r.verdict == Verdict.ERROR
        assert "z_score" in r.message

    def test_critical_fields_empty_allows_all_through(self) -> None:
        """When no field is critical, even NaN raw only gets defensively nulled."""
        r = safe_finalize(
            name="p",
            kind="delta_kl",
            verdict=Verdict.PASS,
            score=1.0,
            raw=math.nan,
            critical_fields=(),
        )
        assert r.verdict == Verdict.PASS
        assert r.raw is None
        assert "raw" in r.evidence["defensively_nulled"]


class TestBoolFieldsNotMistakenForFloat:
    """Pyantic sometimes wraps bools as ints; isinstance(True, int) is True.
    We don't want booleans to be treated as numeric checks.
    """

    def test_true_in_a_numeric_slot_is_not_non_finite(self) -> None:
        # This test pins behavior: even if a caller passes True, we don't
        # crash. We also don't treat True as non-finite.
        r = safe_finalize(
            name="p",
            kind="test",
            verdict=Verdict.PASS,
            score=1.0,
            raw=True,  # type: ignore[arg-type]
        )
        assert r.verdict == Verdict.PASS  # bool is finite


class TestResultTypeReturned:
    def test_returns_probe_result(self) -> None:
        r = safe_finalize(name="p", kind="k", verdict=Verdict.PASS, score=1.0)
        assert isinstance(r, ProbeResult)
