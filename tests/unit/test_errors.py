"""Tests for the exception hierarchy."""

from __future__ import annotations

import pytest

from dlm_sway.core.errors import (
    BackendNotAvailableError,
    ProbeError,
    SpecValidationError,
    SwayError,
)


class TestSwayError:
    def test_is_root_exception(self) -> None:
        assert issubclass(SpecValidationError, SwayError)
        assert issubclass(BackendNotAvailableError, SwayError)
        assert issubclass(ProbeError, SwayError)

    def test_raised_and_caught_as_sway_error(self) -> None:
        with pytest.raises(SwayError):
            raise ProbeError("delta_kl", "shape mismatch")


class TestSpecValidationError:
    def test_format_without_source(self) -> None:
        err = SpecValidationError("unknown key 'topp'")
        assert str(err) == "unknown key 'topp'"
        assert err.source is None

    def test_format_with_source(self) -> None:
        err = SpecValidationError("unknown key 'topp'", source="sway.yaml")
        assert str(err) == "sway.yaml: unknown key 'topp'"
        assert err.source == "sway.yaml"


class TestBackendNotAvailableError:
    def test_hint_rendered_in_message(self) -> None:
        err = BackendNotAvailableError("hf", extra="hf")
        assert "pip install 'dlm-sway[hf]'" in str(err)
        assert err.backend == "hf"
        assert err.extra == "hf"

    def test_appends_optional_hint(self) -> None:
        err = BackendNotAvailableError("mlx", extra="mlx", hint="Apple Silicon only.")
        assert "Apple Silicon only." in str(err)


class TestProbeError:
    def test_includes_probe_name(self) -> None:
        err = ProbeError("delta_kl", "NaN logits")
        assert "delta_kl" in str(err)
        assert "NaN logits" in str(err)
        assert err.probe == "delta_kl"
