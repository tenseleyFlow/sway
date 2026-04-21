"""Unit tests for :mod:`dlm_sway.core.golden`.

Pins the comparator's tolerance math and the variable-field mask so
the cross-platform golden test (S18) has a reliable backbone. No HF
or torch dependency — the comparator is pure-Python and runs in the
fast lane.
"""

from __future__ import annotations

import math

from dlm_sway.core.golden import (
    DEFAULT_VARIABLE_FIELDS,
    Diff,
    compare_goldens,
    mask_variable_fields,
)


class TestMaskVariableFields:
    def test_strips_top_level_fields(self) -> None:
        payload = {
            "sway_version": "0.1.0",
            "wall_seconds": 1.23,
            "probes": [],
        }
        masked = mask_variable_fields(payload)
        assert "sway_version" not in masked
        assert "wall_seconds" not in masked
        assert "probes" in masked

    def test_strips_nested_duration_s(self) -> None:
        payload = {
            "probes": [
                {"name": "p1", "raw": 0.5, "duration_s": 0.01},
                {"name": "p2", "raw": 0.8, "duration_s": 0.02},
            ],
        }
        masked = mask_variable_fields(payload)
        for probe in masked["probes"]:
            assert "duration_s" not in probe
            assert "raw" in probe

    def test_strips_started_and_finished(self) -> None:
        payload = {"started_at": "2026-01-01T00:00:00Z", "finished_at": "2026-01-01T00:00:05Z"}
        masked = mask_variable_fields(payload)
        assert masked == {}

    def test_strips_backend_stats(self) -> None:
        payload = {
            "backend_stats": {"cache_hits": 42, "wall_ms": 1230.0},
            "overall": 0.8,
        }
        masked = mask_variable_fields(payload)
        assert "backend_stats" not in masked
        assert masked["overall"] == 0.8

    def test_preserves_scalars(self) -> None:
        assert mask_variable_fields(42) == 42
        assert mask_variable_fields("hello") == "hello"
        assert mask_variable_fields(None) is None

    def test_default_variable_fields_has_expected_members(self) -> None:
        """Lock the default mask set — accidentally dropping a field
        from the mask would make the golden test newly flaky."""
        expected_members = {
            "started_at",
            "finished_at",
            "wall_seconds",
            "duration_s",
            "sway_version",
            "backend_stats",
            # Platform-dependent path identifiers.
            "adapter_id",
            "base_model_id",
        }
        assert expected_members <= DEFAULT_VARIABLE_FIELDS


class TestCompareGoldensIdentical:
    def test_identical_payload_no_diffs(self) -> None:
        payload = {"overall": 0.85, "probes": [{"raw": 0.123, "score": 0.9}]}
        assert compare_goldens(payload, payload) == []

    def test_empty_payload_no_diffs(self) -> None:
        assert compare_goldens({}, {}) == []


class TestCompareGoldensTolerance:
    def test_floats_within_logprob_tol_pass(self) -> None:
        actual = {"probes": [{"raw": 0.12345}]}
        expected = {"probes": [{"raw": 0.12345 + 5e-7}]}  # well under 1e-6
        assert compare_goldens(actual, expected) == []

    def test_floats_just_above_logprob_tol_fail(self) -> None:
        actual = {"probes": [{"raw": 0.12345}]}
        expected = {"probes": [{"raw": 0.12345 + 2e-6}]}  # double the tol
        diffs = compare_goldens(actual, expected)
        assert len(diffs) == 1
        assert "raw" in diffs[0].path
        assert "Δ" in diffs[0].reason

    def test_scores_use_looser_tol(self) -> None:
        """Score fields get ``score_tol`` (1e-4), not ``logprob_tol``.
        A 5e-5 drift on a score field passes; the same drift on a
        non-score field would fail at default logprob tol."""
        actual = {"overall": 0.85}
        expected = {"overall": 0.85 + 5e-5}
        assert compare_goldens(actual, expected) == []

    def test_score_field_drift_above_score_tol_fails(self) -> None:
        actual = {"overall": 0.85}
        expected = {"overall": 0.85 + 2e-4}  # double the score tol
        diffs = compare_goldens(actual, expected)
        assert len(diffs) == 1
        assert diffs[0].path == "$.overall"

    def test_custom_tolerances_respected(self) -> None:
        """Callers can tighten or loosen both tolerances."""
        actual = {"probes": [{"raw": 0.1}]}
        expected = {"probes": [{"raw": 0.1 + 5e-4}]}
        # Default tol (1e-6) → fail.
        assert compare_goldens(actual, expected) != []
        # Loosened to 1e-3 → pass.
        assert compare_goldens(actual, expected, logprob_tol=1e-3) == []

    def test_nan_vs_nan_treated_equal(self) -> None:
        actual = {"z_score": float("nan")}
        expected = {"z_score": float("nan")}
        assert compare_goldens(actual, expected) == []

    def test_nan_vs_finite_is_drift(self) -> None:
        actual = {"z_score": float("nan")}
        expected = {"z_score": 3.0}
        diffs = compare_goldens(actual, expected)
        assert len(diffs) == 1
        assert diffs[0].path == "$.z_score"

    def test_inf_comparison(self) -> None:
        """Same-signed infinities compare equal; opposite signs drift."""
        actual = {"raw": float("inf")}
        expected = {"raw": float("inf")}
        assert compare_goldens(actual, expected) == []
        diffs = compare_goldens({"raw": float("inf")}, {"raw": float("-inf")})
        assert diffs
        # IEEE compares same-sign as equal but opposite as distinct;
        # the comparator bails on non-finite diffs without a tolerance.

    def test_int_vs_float_not_type_mismatch(self) -> None:
        """``raw: 0`` (int) vs ``raw: 0.0`` (float) is not drift."""
        assert compare_goldens({"raw": 0}, {"raw": 0.0}) == []


class TestCompareGoldensStructural:
    def test_missing_key_flagged(self) -> None:
        actual = {"overall": 0.8}
        expected = {"overall": 0.8, "band": "healthy"}
        diffs = compare_goldens(actual, expected)
        assert any(d.reason == "missing key in actual" for d in diffs)

    def test_extra_key_flagged(self) -> None:
        actual = {"overall": 0.8, "new_field": 42}
        expected = {"overall": 0.8}
        diffs = compare_goldens(actual, expected)
        assert any(d.reason == "unexpected key in actual" for d in diffs)

    def test_list_length_mismatch_flagged(self) -> None:
        actual = {"probes": [{"raw": 0.1}]}
        expected = {"probes": [{"raw": 0.1}, {"raw": 0.2}]}
        diffs = compare_goldens(actual, expected)
        assert len(diffs) == 1
        assert "list length mismatch" in diffs[0].reason

    def test_type_mismatch_flagged(self) -> None:
        actual = {"band": "healthy"}
        expected = {"band": {"name": "healthy", "level": 3}}
        diffs = compare_goldens(actual, expected)
        assert any(d.reason == "type mismatch" for d in diffs)

    def test_string_mismatch_flagged(self) -> None:
        actual = {"band": "noise"}
        expected = {"band": "healthy"}
        diffs = compare_goldens(actual, expected)
        assert len(diffs) == 1
        assert diffs[0].reason == "value mismatch"


class TestDiffRepr:
    def test_str_includes_path_and_reason(self) -> None:
        d = Diff(path="$.foo", actual=1.0, expected=2.0, reason="drift")
        s = str(d)
        assert "$.foo" in s
        assert "drift" in s
        assert "1.0" in s
        assert "2.0" in s


class TestRealisticPayload:
    def test_two_masked_payloads_match(self) -> None:
        """End-to-end sanity: mask timestamps + duration, compare the
        rest, drift-free."""
        actual = {
            "schema_version": 1,
            "sway_version": "0.1.0",
            "started_at": "2026-04-01T00:00:00Z",
            "finished_at": "2026-04-01T00:00:05Z",
            "wall_seconds": 5.123,
            "overall": 0.82,
            "probes": [
                {
                    "name": "dk",
                    "raw": 0.4561,
                    "score": 0.87,
                    "duration_s": 0.123,
                },
            ],
        }
        expected = {
            "schema_version": 1,
            "sway_version": "0.0.9",  # version bumped
            "started_at": "2026-03-15T12:00:00Z",
            "finished_at": "2026-03-15T12:00:03Z",
            "wall_seconds": 3.456,  # different wall
            "overall": 0.82 + 5e-5,  # within score_tol
            "probes": [
                {
                    "name": "dk",
                    "raw": 0.4561 + 5e-7,  # within logprob_tol
                    "score": 0.87,
                    "duration_s": 0.789,  # different duration
                },
            ],
        }
        masked_actual = mask_variable_fields(actual)
        masked_expected = mask_variable_fields(expected)
        assert compare_goldens(masked_actual, masked_expected) == []

    def test_simulated_silent_algorithm_change_is_caught(self) -> None:
        """Prove-the-value sanity: a 1e-3 drift on a probe's raw is
        flagged, even when every variable field differs."""
        expected = {"probes": [{"raw": 0.4561}]}
        # Simulate an algorithm change: someone edited delta_kl's
        # top_k default and raw shifted by 1e-3.
        actual = {"probes": [{"raw": 0.4571}]}
        diffs = compare_goldens(actual, expected)
        assert len(diffs) == 1
        assert "raw" in diffs[0].path
        assert math.isclose(
            abs(float(diffs[0].actual) - float(diffs[0].expected)), 1e-3, abs_tol=1e-9
        )
