"""Backend proxy used by ``NullAdapterProbe`` to drive per-kind calibration.

The proxy wraps a :class:`~dlm_sway.core.scoring.NullCalibratedBackend`
so that ``as_finetuned()`` actually yields a null-adapter view at a
fixed seed. From the proxied probe's perspective it sees a regular
``DifferentialBackend``: ``as_base()`` is the real base and
``as_finetuned()`` is a structural-noise (random-init LoRA) view.

Running a probe with this proxy substituted into the
:class:`~dlm_sway.probes.base.RunContext` produces "what does this
probe report when the fine-tune is just noise?" — exactly the
denominator each numeric probe needs to z-score itself.

The proxy only implements :class:`DifferentialBackend`; it does *not*
forward :class:`ScalableDifferentialBackend` (because adapter ablation
makes no sense on a null adapter) or :class:`NullCalibratedBackend`
(probes shouldn't recursively null-calibrate themselves). This is a
deliberate narrowing.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from dlm_sway.core.scoring import (
    NullCalibratedBackend,
    ScoringModel,
)


class NullCalibrationBackendProxy:
    """Wraps a backend so ``as_finetuned()`` yields the null view.

    Constructed once per calibration seed by
    :class:`~dlm_sway.probes.null_adapter.NullAdapterProbe`. Each
    seed produces an independent draw from the null distribution.

    Attributes
    ----------
    inner:
        The real backend whose null views we substitute.
    seed:
        Seed handed to ``inner.as_null_adapter(seed)`` on every
        ``as_finetuned()`` entry. Reusing the proxy across multiple
        ``as_finetuned()`` blocks therefore yields the *same* null
        view (deterministic).
    init_scale:
        Forwarded to ``as_null_adapter(init_scale=…)``.
    rank_scale:
        Forwarded to ``as_null_adapter(rank_scale=…)``. Defaults to
        1.0 for parity with pre-S10 behavior. ``NullAdapterProbe``
        builds one proxy per ``rank_multipliers`` entry when calibrating
        across ranks.
    """

    def __init__(
        self,
        inner: NullCalibratedBackend,
        *,
        seed: int,
        init_scale: float = 0.02,
        rank_scale: float = 1.0,
    ) -> None:
        self._inner = inner
        self._seed = seed
        self._init_scale = init_scale
        self._rank_scale = rank_scale

    @contextmanager
    def as_base(self) -> Iterator[ScoringModel]:
        """Forwarded straight to the inner backend's base view."""
        with self._inner.as_base() as view:
            yield view

    @contextmanager
    def as_finetuned(self) -> Iterator[ScoringModel]:
        """The substitution: yield a null-adapter view, not the real ft.

        A probe that calls ``as_finetuned()`` against the proxy is
        *actually* asking "what does this metric look like when the
        adapter is structural noise" — which is the calibration
        question.
        """
        with self._inner.as_null_adapter(
            self._seed, init_scale=self._init_scale, rank_scale=self._rank_scale
        ) as view:
            yield view


__all__ = ["NullCalibrationBackendProxy"]
