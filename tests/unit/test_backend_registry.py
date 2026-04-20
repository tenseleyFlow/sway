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

    def test_mlx_requires_adapter(self) -> None:
        with pytest.raises(SpecValidationError, match="adapter"):
            build(ModelSpec(base="x", kind="mlx"))

    def test_mlx_dispatch_raises_when_mlx_missing(self) -> None:
        # On non-Apple-Silicon (or Apple without mlx installed), constructing
        # the MLX backend raises BackendNotAvailableError with a pip hint.
        # We skip this assertion if mlx happens to be installed.
        import importlib.util

        if importlib.util.find_spec("mlx") is not None:
            pytest.skip("mlx is installed; error path not exercised")
        with pytest.raises(BackendNotAvailableError) as exc_info:
            build(ModelSpec(base="x", kind="mlx", adapter=Path("/tmp/a")))
        assert exc_info.value.backend == "mlx"

    def test_custom_requires_entry_point(self) -> None:
        with pytest.raises(SpecValidationError, match="entry_point"):
            build(ModelSpec(base="x", kind="custom", adapter=Path("/tmp/a")))

    def test_custom_validates_entry_point_shape(self) -> None:
        with pytest.raises(SpecValidationError, match="pkg.module:ClassName"):
            build(
                ModelSpec(
                    base="x",
                    kind="custom",
                    entry_point="not_a_valid_entry_point",
                    adapter=Path("/tmp/a"),
                )
            )

    def test_custom_rejects_unimportable_module(self) -> None:
        with pytest.raises(SpecValidationError, match="cannot import"):
            build(
                ModelSpec(
                    base="x",
                    kind="custom",
                    entry_point="nonexistent_pkg_xyz:Backend",
                    adapter=Path("/tmp/a"),
                )
            )

    def test_custom_rejects_missing_class(self) -> None:
        with pytest.raises(SpecValidationError, match="has no attribute"):
            build(
                ModelSpec(
                    base="x",
                    kind="custom",
                    entry_point="dlm_sway:NoSuchClass",
                    adapter=Path("/tmp/a"),
                )
            )

    def test_custom_rejects_non_differential_class(self) -> None:
        # A class that accepts the canonical constructor args but doesn't
        # implement the protocol.
        import sys
        import types

        class _Bad:
            def __init__(self, base_spec, adapter_path):  # type: ignore[no-untyped-def]
                del base_spec, adapter_path

        mod = types.ModuleType("_sway_bad_mod")
        mod.Bad = _Bad  # type: ignore[attr-defined]
        sys.modules["_sway_bad_mod"] = mod

        with pytest.raises(SpecValidationError, match="DifferentialBackend"):
            build(
                ModelSpec(
                    base="x",
                    kind="custom",
                    entry_point="_sway_bad_mod:Bad",
                    adapter=Path("/tmp/a"),
                )
            )

    def test_custom_dispatches_to_valid_backend(self) -> None:
        # Use the dummy backend via a custom entry point. The dummy class's
        # __init__ takes different args, so we write a thin adapter class.
        from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses

        class _AdapterBackend(DummyDifferentialBackend):
            def __init__(self, base_spec, adapter_path):  # type: ignore[no-untyped-def]
                super().__init__(base=DummyResponses(), ft=DummyResponses())

        # Register on a throwaway module we can find by name.
        import sys
        import types

        mod = types.ModuleType("_sway_custom_test_mod")
        mod.AdapterBackend = _AdapterBackend  # type: ignore[attr-defined]
        sys.modules["_sway_custom_test_mod"] = mod

        backend = build(
            ModelSpec(
                base="x",
                kind="custom",
                entry_point="_sway_custom_test_mod:AdapterBackend",
                adapter=Path("/tmp/a"),
            )
        )
        from dlm_sway.core.scoring import DifferentialBackend

        assert isinstance(backend, DifferentialBackend)
