"""Tests for :mod:`dlm_sway.suite.spec` + :mod:`dlm_sway.suite.loader`."""

from __future__ import annotations

from pathlib import Path

import pytest

from dlm_sway.core.errors import SpecValidationError
from dlm_sway.suite.loader import from_dict, load_spec
from dlm_sway.suite.spec import SwaySpec


def _minimum_valid() -> dict:
    return {
        "version": 1,
        "models": {
            "base": {"kind": "hf", "base": "HuggingFaceTB/SmolLM2-135M-Instruct"},
            "ft": {
                "kind": "hf",
                "base": "HuggingFaceTB/SmolLM2-135M-Instruct",
                "adapter": "/tmp/adapter",
            },
        },
        "suite": [],
    }


class TestSwaySpec:
    def test_minimum_valid(self) -> None:
        spec = from_dict(_minimum_valid())
        assert isinstance(spec, SwaySpec)
        assert spec.version == 1
        assert spec.defaults.seed == 0
        assert spec.defaults.differential is True
        assert spec.suite == []

    def test_rejects_unknown_top_level_keys(self) -> None:
        data = _minimum_valid()
        data["bogus"] = True
        with pytest.raises(SpecValidationError) as exc_info:
            from_dict(data)
        assert "bogus" in str(exc_info.value).lower()

    def test_rejects_future_version(self) -> None:
        data = _minimum_valid()
        data["version"] = 9
        with pytest.raises(SpecValidationError, match="unsupported sway spec version"):
            from_dict(data)

    def test_defaults_frozen(self) -> None:
        spec = from_dict(_minimum_valid())
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            spec.defaults.seed = 99  # type: ignore[misc]


class TestLoader:
    def test_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.yaml"
        with pytest.raises(SpecValidationError, match="not found"):
            load_spec(missing)

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        # An unmatched { triggers yaml.scanner; a structurally ambiguous
        # indent parses as a string value, which isn't a YAML error.
        bad.write_text("{ unmatched: [", encoding="utf-8")
        with pytest.raises(SpecValidationError, match="invalid YAML"):
            load_spec(bad)

    def test_non_mapping_top_level(self, tmp_path: Path) -> None:
        bad = tmp_path / "list.yaml"
        bad.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(SpecValidationError, match="must be a mapping"):
            load_spec(bad)

    def test_roundtrip_via_yaml(self, tmp_path: Path) -> None:
        import yaml

        path = tmp_path / "sway.yaml"
        path.write_text(yaml.safe_dump(_minimum_valid()), encoding="utf-8")
        spec = load_spec(path)
        assert spec.models.ft.adapter == Path("/tmp/adapter")
