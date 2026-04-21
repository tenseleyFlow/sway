"""Top-level ``sway.yaml`` spec models.

Per-probe specs live next to their implementations in
:mod:`dlm_sway.probes`. This module owns the *outer* envelope —
``version``, ``models``, ``defaults``, ``suite`` — plus the runtime
bind between raw probe dicts and registered probe classes.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dlm_sway.core.model import ModelSpec
from dlm_sway.core.result import DEFAULT_COMPONENT_WEIGHTS

SUPPORTED_VERSION = 1


class SuiteModels(BaseModel):
    """Named model handles the suite references — ``base`` + ``ft``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    base: ModelSpec
    ft: ModelSpec


class SuiteDefaults(BaseModel):
    """Shared defaults for the whole suite. Probes may override per-entry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = 0
    top_k: int = 256
    differential: bool = True
    """If ``False``, the runner loads base + ft as two separate models
    instead of toggling on one. More memory-heavy; only useful when a
    backend can't do in-place toggling."""
    coverage_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.6
    """Minimum composite score for ``sway gate`` to pass."""
    concurrent_probes: Annotated[int, Field(ge=1, le=32)] = 1
    """Maximum number of independent probes to dispatch concurrently.

    **Default is 1 (sequential).** Values > 1 are respected only when
    the backend declares
    :attr:`~dlm_sway.core.scoring.DifferentialBackend.safe_for_concurrent_views`
    as ``True`` — which no shipped backend does in v0.1 (see
    ``.docs/design/backend-concurrency.md`` / B19). The flag exists so
    custom backends that *are* already concurrency-safe (e.g. a
    stateless hosted-API backend) can opt in without waiting for the
    HF backend fix; shipped backends treat it as a no-op."""
    score_weights: dict[str, float] | None = None
    """Per-category weight overrides for the composite score. ``None``
    uses :data:`dlm_sway.core.result.DEFAULT_COMPONENT_WEIGHTS`. Keys
    must be a subset of the known categories (``adherence``,
    ``attribution``, ``calibration``, ``ablation``, ``baseline``);
    unknown keys are rejected. Missing keys inherit the default weight
    so a user who only wants to re-weight one category doesn't have to
    respecify all of them. All values must be non-negative and at
    least one must be positive."""

    @field_validator("score_weights")
    @classmethod
    def _validate_weights(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        if v is None:
            return v
        known = set(DEFAULT_COMPONENT_WEIGHTS)
        unknown = sorted(set(v) - known)
        if unknown:
            raise ValueError(
                f"score_weights contains unknown category keys: {unknown}. "
                f"Known categories: {sorted(known)}"
            )
        if any(w < 0.0 for w in v.values()):
            raise ValueError("score_weights values must be non-negative")
        # Merge with defaults so partial overrides are ergonomic.
        merged = dict(DEFAULT_COMPONENT_WEIGHTS)
        merged.update(v)
        if sum(merged.values()) <= 0.0:
            raise ValueError("score_weights must have at least one positive weight")
        return merged


class SwaySpec(BaseModel):
    """Root of ``sway.yaml``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = 1
    models: SuiteModels
    defaults: SuiteDefaults = SuiteDefaults()
    suite: list[dict[str, Any]] = Field(default_factory=list)
    """Raw probe entries. Validated one-at-a-time by the probe registry
    via :func:`dlm_sway.probes.base.build_probe` so that the set of
    allowed probe kinds is an open registry rather than a closed
    discriminated union."""
    dlm_source: str | None = None
    """Optional path to a ``.dlm`` file. When present, the runner asks
    :mod:`dlm_sway.integrations.dlm.resolver` for typed sections and
    hands them to probes via :attr:`RunContext.sections`. Auto-populated
    by ``sway autogen``."""

    def check_version(self) -> None:
        """Raise ``ValueError`` if the spec version is unsupported.

        Called explicitly by the loader after validation so the error
        surfaces with a loader-source tag rather than a pydantic stack.
        """
        if self.version != SUPPORTED_VERSION:
            raise ValueError(
                f"unsupported sway spec version: {self.version} (this build supports {SUPPORTED_VERSION})"
            )
