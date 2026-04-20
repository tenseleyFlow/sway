"""Scoring backends: HuggingFace (``hf``), MLX (``mlx``), dummy, custom.

Backends are constructed from a :class:`~dlm_sway.core.model.ModelSpec`
via :func:`build`. Heavy backends (HF, MLX) import their framework only
on construction so ``import dlm_sway`` stays cheap for users who only
touch the dummy backend or the spec loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dlm_sway.core.errors import SpecValidationError
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
        if effective_adapter is None:
            raise SpecValidationError(
                "mlx backend requires an adapter path (set `adapter:` on the ft model; "
                "must be an MLX .npz adapter — use dlm's peft→mlx converter if needed)"
            )
        from dlm_sway.backends.mlx import MLXDifferentialBackend

        return MLXDifferentialBackend(base_spec=base_spec, adapter_path=effective_adapter)

    if base_spec.kind == "custom":
        return _load_custom(base_spec, effective_adapter)

    raise SpecValidationError(f"unknown backend kind: {base_spec.kind!r}")


def _load_custom(base_spec: ModelSpec, adapter: Path | None) -> DifferentialBackend:
    """Dispatch to a user-supplied backend via ``entry_point='pkg.mod:Name'``.

    The imported class is instantiated as ``Cls(base_spec=..., adapter_path=...)``
    — the same signature as :class:`dlm_sway.backends.hf.HuggingFaceDifferentialBackend`
    so authors can model their implementation on the built-in. The
    result is runtime-checked against :class:`DifferentialBackend` so
    protocol violations fail at construction, not deep inside a probe.
    """
    from dlm_sway.core.scoring import DifferentialBackend as DiffBackend

    entry = base_spec.entry_point
    if not entry:
        raise SpecValidationError(
            "kind='custom' requires an entry_point of the form 'pkg.module:ClassName'"
        )
    if ":" not in entry:
        raise SpecValidationError(f"entry_point must be 'pkg.module:ClassName', got {entry!r}")
    module_path, _, class_name = entry.partition(":")
    if not module_path or not class_name:
        raise SpecValidationError(f"entry_point must be 'pkg.module:ClassName', got {entry!r}")

    import importlib

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise SpecValidationError(
            f"custom backend: cannot import module {module_path!r}: {exc}"
        ) from exc
    cls = getattr(module, class_name, None)
    if cls is None:
        raise SpecValidationError(
            f"custom backend: module {module_path!r} has no attribute {class_name!r}"
        )

    try:
        instance = cls(base_spec=base_spec, adapter_path=adapter)
    except TypeError as exc:
        raise SpecValidationError(
            f"custom backend {entry!r} constructor signature mismatch: {exc}. "
            "Expected Cls(base_spec: ModelSpec, adapter_path: Path | None)"
        ) from exc

    if not isinstance(instance, DiffBackend):
        raise SpecValidationError(
            f"custom backend {entry!r} does not satisfy DifferentialBackend "
            "(needs as_base() and as_finetuned() context managers)"
        )
    return instance


__all__ = ["build"]
