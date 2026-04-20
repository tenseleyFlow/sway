"""Tests for :mod:`dlm_sway.core.model`."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dlm_sway.core.model import LoadedModel, Model, ModelSpec


class TestModelSpec:
    def test_defaults(self) -> None:
        spec = ModelSpec(base="HuggingFaceTB/SmolLM2-135M-Instruct")
        assert spec.kind == "hf"
        assert spec.adapter is None
        assert spec.dtype == "auto"
        assert spec.device == "auto"
        assert spec.trust_remote_code is False
        assert spec.entry_point is None

    def test_frozen(self) -> None:
        spec = ModelSpec(base="x")
        with pytest.raises(ValidationError):
            spec.base = "y"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ModelSpec(base="x", bogus="y")  # type: ignore[call-arg]
        assert "bogus" in str(exc_info.value).lower()

    def test_kind_enum(self) -> None:
        ModelSpec(base="x", kind="hf")
        ModelSpec(base="x", kind="mlx")
        ModelSpec(base="x", kind="dummy")
        ModelSpec(base="x", kind="custom", entry_point="pkg.mod:Backend")
        with pytest.raises(ValidationError):
            ModelSpec(base="x", kind="ollama")  # type: ignore[arg-type]

    def test_adapter_coerced_to_path(self) -> None:
        spec = ModelSpec(base="x", adapter="/tmp/adapter")  # type: ignore[arg-type]
        assert isinstance(spec.adapter, Path)


class TestLoadedModel:
    def test_frozen_dataclass(self) -> None:
        loaded = LoadedModel(
            id="base",
            spec=ModelSpec(base="x"),
            model=object(),
            tokenizer=object(),
            meta={"device": "cpu"},
        )
        assert loaded.id == "base"
        assert loaded.meta["device"] == "cpu"


class TestModelProtocol:
    def test_runtime_checkable(self) -> None:
        class FakeModel:
            id = "x"

            def generate(
                self,
                prompt: str,
                *,
                max_new_tokens: int,
                temperature: float = 0.0,
                top_p: float = 1.0,
                seed: int = 0,
            ) -> str:
                return f"{prompt}|{max_new_tokens}"

            def close(self) -> None:
                return None

        assert isinstance(FakeModel(), Model)
