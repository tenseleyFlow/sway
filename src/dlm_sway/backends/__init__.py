"""Scoring backends: HuggingFace (``hf``), MLX (``mlx``), dummy, custom.

Backends are constructed from a :class:`~dlm_sway.core.model.ModelSpec`
via :func:`build`. Heavy backends (HF, MLX) import their framework only
on construction so ``import dlm_sway`` stays cheap for users who only
touch the dummy backend or the spec loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dlm_sway.core.errors import BackendNotAvailableError, SpecValidationError
from dlm_sway.core.model import ModelSpec

if TYPE_CHECKING:
    from dlm_sway.core.scoring import DifferentialBackend


def build(base_spec: ModelSpec, *, adapter_path: Path | None = None) -> DifferentialBackend:
    """Materialize a differential backend from a model spec.

    The adapter path typically comes from ``ft.adapter`` in the spec —
    it's lifted to a keyword here so the same function can be used for
    "differential" (base + adapter on one loaded model) or future
    split-load paths.
    """
    effective_adapter = adapter_path if adapter_path is not None else base_spec.adapter

    if base_spec.kind == "dummy":
        # Dummy backend isn't really about the spec — it's for tests
        # that pre-populate responses. Surface a loud error if someone
        # tries to build it through the normal path.
        raise SpecValidationError(
            "kind='dummy' backends must be constructed directly via "
            "DummyDifferentialBackend(base=..., ft=...); they cannot be "
            "materialized from a ModelSpec."
        )

    if base_spec.kind == "hf":
        if effective_adapter is None:
            raise SpecValidationError(
                "hf backend requires an adapter path (set `adapter:` on the ft model)"
            )
        from dlm_sway.backends.hf import HuggingFaceDifferentialBackend

        return HuggingFaceDifferentialBackend(base_spec=base_spec, adapter_path=effective_adapter)

    if base_spec.kind == "mlx":
        raise BackendNotAvailableError(
            "mlx",
            extra="mlx",
            hint="MLX backend shipping in a later milestone.",
        )

    if base_spec.kind == "custom":
        raise BackendNotAvailableError(
            "custom",
            extra="hf",
            hint="Custom backend entry-point dispatch shipping in a later milestone.",
        )

    raise SpecValidationError(f"unknown backend kind: {base_spec.kind!r}")


__all__ = ["build"]
