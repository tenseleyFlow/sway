"""dlm-sway — differential testing for fine-tuned causal language models."""

from __future__ import annotations

from dlm_sway.core.errors import (
    BackendNotAvailableError,
    ProbeError,
    SpecValidationError,
    SwayError,
)
from dlm_sway.core.model import LoadedModel, Model, ModelSpec
from dlm_sway.core.result import ProbeResult, SuiteResult, SwayScore, Verdict
from dlm_sway.core.scoring import (
    DifferentialBackend,
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
]

__version__ = "0.1.0.dev0"
