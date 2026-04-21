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

    def test_adapter_tilde_expanded(self) -> None:
        """B22: ``~`` is expanded at spec-load time so backends see absolute paths."""
        spec = ModelSpec(base="x", adapter="~/some/adapter")  # type: ignore[arg-type]
        assert spec.adapter is not None
        assert "~" not in str(spec.adapter)
        assert spec.adapter.is_absolute()

    def test_adapter_relative_resolved(self) -> None:
        """B22: relative paths resolve against the current cwd."""
        spec = ModelSpec(base="x", adapter="adapter/v1")  # type: ignore[arg-type]
        assert spec.adapter is not None
        assert spec.adapter.is_absolute()

    def test_adapter_none_passthrough(self) -> None:
        """B22 normalizer doesn't blow up on the default ``None``."""
        spec = ModelSpec(base="x")
        assert spec.adapter is None

    def test_dtype_enum_accepts_known_values(self) -> None:
        """DC5 — every ``dtype`` branch parses cleanly. Backends rely
        on this validation, not their own — passing ``fp12`` here would
        otherwise crash deep inside the HF loader."""
        for dtype in ("auto", "fp16", "bf16", "fp32"):
            spec = ModelSpec(base="x", dtype=dtype)  # type: ignore[arg-type]
            assert spec.dtype == dtype

    def test_dtype_enum_rejects_unknown(self) -> None:
        with pytest.raises(ValidationError):
            ModelSpec(base="x", dtype="fp12")  # type: ignore[arg-type]

    def test_trust_remote_code_default_false(self) -> None:
        """DC5 — default is False (safe posture); user must opt in."""
        assert ModelSpec(base="x").trust_remote_code is False

    def test_trust_remote_code_accepts_true(self) -> None:
        assert ModelSpec(base="x", trust_remote_code=True).trust_remote_code is True

    def test_endpoint_default_none(self) -> None:
        """DC5 — ``endpoint`` is an ``api``-backend-only field. Default
        ``None`` when not specified."""
        assert ModelSpec(base="x").endpoint is None

    def test_endpoint_can_be_set(self) -> None:
        """DC5 — the ``api`` kind expects an endpoint URL."""
        spec = ModelSpec(
            base="gpt-3.5-turbo-instruct",
            kind="api",
            endpoint="http://localhost:11434",
        )
        assert spec.endpoint == "http://localhost:11434"
        assert spec.kind == "api"

    def test_entry_point_optional_without_custom(self) -> None:
        """DC5 — non-custom kinds don't need an entry_point."""
        spec = ModelSpec(base="x", kind="hf")
        assert spec.entry_point is None

    def test_custom_kind_accepts_entry_point(self) -> None:
        """DC5 — ``custom`` kind stores the entry_point verbatim (the
        runner imports it)."""
        spec = ModelSpec(base="x", kind="custom", entry_point="mypkg.backend:MyBackend")
        assert spec.entry_point == "mypkg.backend:MyBackend"

    def test_device_accepts_explicit_cpu(self) -> None:
        """DC5 — ``device`` is a free-form str; ``"auto"`` default
        resolves at backend-load time."""
        assert ModelSpec(base="x", device="cpu").device == "cpu"
        assert ModelSpec(base="x", device="cuda:0").device == "cuda:0"
        assert ModelSpec(base="x").device == "auto"


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
