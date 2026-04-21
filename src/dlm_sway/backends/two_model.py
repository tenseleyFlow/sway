"""Two-model differential wrapper.

Used when ``sway.yaml`` sets ``defaults.differential: false``. Instead of
toggling one loaded model via PEFT, the runner loads two independent
backends and routes ``as_base()`` through one and ``as_finetuned()``
through the other.

This doubles memory (two full model copies in RAM) and halves throughput
— the flag exists for custom backends that can't do in-place adapter
toggling, not as a production setting. The differential toggle path
remains the recommended configuration.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dlm_sway.core.scoring import DifferentialBackend


class TwoModelDifferential:
    """Wrap two independent backends as a single ``DifferentialBackend``.

    ``as_base()`` delegates to ``base.as_base()``; ``as_finetuned()``
    delegates to ``ft.as_finetuned()``. The two inner backends are held
    for their lifetime — callers close them explicitly (or via
    :meth:`close`) when finished.

    The wrapper does **not** enforce view exclusion across the two
    backends (each inner backend is responsible for its own exclusion,
    but nothing stops a caller holding simultaneous base+ft contexts
    that come from different backends). That's fine — the exclusion
    invariant in :class:`DifferentialBackend` exists to protect the
    toggle state of a single-model backend; two independent models
    have no toggle to corrupt.
    """

    def __init__(self, base: DifferentialBackend, ft: DifferentialBackend) -> None:
        self._base = base
        self._ft = ft
        # Compose the concurrency flag from both inner backends. The
        # wrapper itself holds no shared mutable state — the toggle
        # invariant :class:`DifferentialBackend` protects doesn't apply
        # here — so the composed value is exactly as strong as the
        # weaker of the two inner backends.
        self.safe_for_concurrent_views: bool = bool(
            getattr(self._base, "safe_for_concurrent_views", False)
            and getattr(self._ft, "safe_for_concurrent_views", False)
        )

    @contextmanager
    def as_base(self) -> Iterator[Any]:
        with self._base.as_base() as view:
            yield view

    @contextmanager
    def as_finetuned(self) -> Iterator[Any]:
        with self._ft.as_finetuned() as view:
            yield view

    def close(self) -> None:
        """Close both inner backends if they expose a ``close()`` method."""
        for backend in (self._base, self._ft):
            close = getattr(backend, "close", None)
            if callable(close):
                close()

    def preflight_finite_check(self) -> tuple[bool, str]:
        """Delegate preflight to the ft-side backend if it supports it.

        The failure mode the preflight catches is "adapter weights are
        NaN" — which lives on the ft side only. A base-only check
        would validate the wrong thing.
        """
        fn = getattr(self._ft, "preflight_finite_check", None)
        if not callable(fn):
            return True, "ft backend does not support preflight"
        ok, reason = fn()
        return bool(ok), str(reason)


def _build_via_backends(spec_models: Any) -> tuple[DifferentialBackend, DifferentialBackend]:
    """Materialize ``(base_backend, ft_backend)`` from a ``SuiteModels`` spec.

    Uses :func:`dlm_sway.backends.build` for each side. The base-side
    build fails if the user didn't supply an adapter path on the base
    spec — in which case we raise a clear error pointing them at
    ``differential: true``.
    """
    from dlm_sway.backends import build
    from dlm_sway.core.errors import SpecValidationError

    base_spec = spec_models.base
    ft_spec = spec_models.ft

    if base_spec.kind == "hf" and base_spec.adapter is None:
        raise SpecValidationError(
            "defaults.differential=false with an HF base requires the base "
            "ModelSpec to carry an adapter path too (the HF backend loads "
            "via PEFT). Either set `models.base.adapter` explicitly, or "
            "switch to `defaults.differential: true` to use the single-load "
            "toggle path."
        )

    base_backend = build(base_spec)
    ft_backend = build(ft_spec)
    return base_backend, ft_backend


def build_two_separate(spec_models: Any) -> TwoModelDifferential:
    """Build a :class:`TwoModelDifferential` from a ``SuiteModels`` spec.

    Front door used by the suite runner when
    ``spec.defaults.differential`` is ``False``. Re-exported from
    :mod:`dlm_sway.backends` for symmetry with :func:`backends.build`.
    """
    base_backend, ft_backend = _build_via_backends(spec_models)
    return TwoModelDifferential(base=base_backend, ft=ft_backend)


__all__ = ["TwoModelDifferential", "build_two_separate"]
