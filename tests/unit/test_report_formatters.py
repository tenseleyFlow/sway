"""Tests for the unified number formatters in :mod:`dlm_sway.suite.report` (D10)."""

from __future__ import annotations

import math

from dlm_sway.suite import report


class TestFormatScore:
    def test_two_decimals(self) -> None:
        assert report.format_score(0.8765) == "0.88"

    def test_none_is_em_dash(self) -> None:
        assert report.format_score(None) == "—"

    def test_nan_is_em_dash(self) -> None:
        assert report.format_score(math.nan) == "—"

    def test_inf_is_em_dash(self) -> None:
        assert report.format_score(math.inf) == "—"

    def test_int_accepted(self) -> None:
        assert report.format_score(1) == "1.00"


class TestFormatRaw:
    def test_three_decimals(self) -> None:
        assert report.format_raw(0.123456) == "0.123"

    def test_thousands_separator(self) -> None:
        assert report.format_raw(1234.5678) == "1,234.568"

    def test_none_is_em_dash(self) -> None:
        assert report.format_raw(None) == "—"


class TestFormatZ:
    def test_signed_with_sigma(self) -> None:
        assert report.format_z(3.14) == "+3.14σ"

    def test_negative(self) -> None:
        assert report.format_z(-1.5) == "-1.50σ"

    def test_large_thousands_separator(self) -> None:
        assert report.format_z(1234.56) == "+1,234.56σ"

    def test_none_is_em_dash(self) -> None:
        assert report.format_z(None) == "—"


class TestFormatDuration:
    def test_sub_ten_seconds(self) -> None:
        assert report.format_duration_s(1.234) == "1.23s"

    def test_between_ten_and_hundred(self) -> None:
        assert report.format_duration_s(42.678) == "42.7s"

    def test_large_seconds_with_thousands(self) -> None:
        assert report.format_duration_s(12345.6) == "12,346s"

    def test_none_is_em_dash(self) -> None:
        assert report.format_duration_s(None) == "—"
