"""Tests for the backend registry in ``dlm_sway.backends``.

The registry is the single place that maps a ModelSpec to a concrete
backend. These tests check the error paths — actually materializing an
HF backend requires model weights and is covered by the integration
suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlm_sway.backends import build
from dlm_sway.core.errors import BackendNotAvailableError, SpecValidationError
from dlm_sway.core.model import ModelSpec


class TestRegistry:
    def test_dummy_rejected_via_build(self) -> None:
        with pytest.raises(SpecValidationError, match="kind='dummy'"):
            build(ModelSpec(base="x", kind="dummy"))

    def test_hf_requires_adapter(self) -> None:
        with pytest.raises(SpecValidationError, match="adapter"):
            build(ModelSpec(base="x", kind="hf"))

    def test_mlx_not_yet_available(self) -> None:
        with pytest.raises(BackendNotAvailableError) as exc_info:
            build(ModelSpec(base="x", kind="mlx", adapter=Path("/tmp/a")))
        assert exc_info.value.backend == "mlx"

    def test_custom_not_yet_available(self) -> None:
        with pytest.raises(BackendNotAvailableError):
            build(
                ModelSpec(
                    base="x",
                    kind="custom",
                    entry_point="pkg:Backend",
                    adapter=Path("/tmp/a"),
                )
            )
