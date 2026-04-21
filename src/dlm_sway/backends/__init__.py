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

    if base_spec.kind == "api":
        # An API backend represents ONE endpoint — it has no local
        # model toggle. The idiomatic path is
        # ``defaults.differential: false`` in sway.yaml, which routes
        # through :func:`build_two_separate`; that calls this dispatch
        # twice (once per side) and wraps the results in
        # :class:`TwoModelDifferential`. The ``adapter:`` field is
        # ignored — "adapter" for an API backend is a *different
        # model name*, not a local file path.
        if base_spec.endpoint is None:
            raise SpecValidationError(
                "api backend requires `endpoint:` on the ModelSpec "
                "(the base URL of the /v1/completions server)"
            )
        from dlm_sway.backends.api import ApiScoringBackend

        return ApiScoringBackend(
            base_url=base_spec.endpoint,
            model_name=base_spec.base,
        )

    if base_spec.kind == "custom":
        return _load_custom(base_spec, effective_adapter)

    raise SpecValidationError(f"unknown backend kind: {base_spec.kind!r}")


def _load_custom(base_spec: ModelSpec, adapter: Path | None) -> DifferentialBackend:
    """Dispatch to a user-supplied backend via ``entry_point='pkg.mod:Name'``.

    The imported class is instantiated as ``Cls(base_spec=..., adapter_path=...)``
    — the same signature as :class:`dlm_sway.backends.hf.HuggingFaceDifferentialBackend`
    so authors can model their implementation on the built-in. The
    result is runtime-checked against :class:`DifferentialBackend` and
    the optional :class:`NullCalibratedBackend` /
    :class:`ScalableDifferentialBackend` so protocol violations fail
    at construction, not deep inside a probe (B20). The set of
    satisfied protocols is recorded on the instance as
    ``__sway_protocols__: tuple[str, ...]`` for the report's
    backend-info section.
    """
    from dlm_sway.core.scoring import (
        DifferentialBackend as DiffBackend,
    )
    from dlm_sway.core.scoring import (
        NullCalibratedBackend,
        ScalableDifferentialBackend,
    )

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

    # B20: probe optional protocols and record them so the runner /
    # report can show which downstream features are available without
    # repeated isinstance checks.
    satisfied: list[str] = ["DifferentialBackend"]
    if isinstance(instance, NullCalibratedBackend):
        satisfied.append("NullCalibratedBackend")
    if isinstance(instance, ScalableDifferentialBackend):
        satisfied.append("ScalableDifferentialBackend")
    instance.__sway_protocols__ = tuple(satisfied)  # type: ignore[attr-defined]
    return instance


from dlm_sway.backends.two_model import TwoModelDifferential, build_two_separate  # noqa: E402

__all__ = ["TwoModelDifferential", "build", "build_two_separate"]
