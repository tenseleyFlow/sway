"""Top-level ``sway.yaml`` spec models.

Per-probe specs live next to their implementations in
:mod:`dlm_sway.probes`. This module owns the *outer* envelope —
``version``, ``models``, ``defaults``, ``suite`` — plus the runtime
bind between raw probe dicts and registered probe classes.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from dlm_sway.core.model import ModelSpec

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
    """Minimum composite score for ``dlm-sway gate`` to pass."""


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

    def check_version(self) -> None:
        """Raise ``ValueError`` if the spec version is unsupported.

        Called explicitly by the loader after validation so the error
        surfaces with a loader-source tag rather than a pydantic stack.
        """
        if self.version != SUPPORTED_VERSION:
            raise ValueError(
                f"unsupported sway spec version: {self.version} (this build supports {SUPPORTED_VERSION})"
            )
