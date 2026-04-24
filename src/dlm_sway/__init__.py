"""sway — differential testing for fine-tuned causal language models.

Published on PyPI as ``dlm-sway`` (the short name is taken); the CLI
entry point and source repo are ``sway``.
"""

from __future__ import annotations

from dlm_sway.core.errors import (
    BackendNotAvailableError,
    ProbeError,
    SpecValidationError,
    SwayError,
)
from dlm_sway.core.model import LoadedModel, Model, ModelSpec
from dlm_sway.core.result import ProbeResult, SuiteResult, SwayScore, Verdict, safe_finalize
from dlm_sway.core.scoring import (
    DifferentialBackend,
    NullCalibratedBackend,
    PreflightCheckable,
    RollingLogprob,
    ScalableDifferentialBackend,
    ScoringBackend,
    TokenDist,
)

__all__ = [
    "BackendNotAvailableError",
    "DifferentialBackend",
    "LoadedModel",
    "Model",
    "ModelSpec",
    "NullCalibratedBackend",
    "PreflightCheckable",
    "ProbeError",
    "ProbeResult",
    "RollingLogprob",
    "ScalableDifferentialBackend",
    "ScoringBackend",
    "SpecValidationError",
    "SuiteResult",
    "SwayError",
    "SwayScore",
    "TokenDist",
    "Verdict",
    "safe_finalize",
]

__version__ = "0.1.0"
