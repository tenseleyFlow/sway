"""Probe abstract base + per-kind registry.

The registry is the extension point. Adding a new probe means:

1. Subclass :class:`ProbeSpec` with a unique ``kind`` field (Literal).
2. Subclass :class:`Probe` setting ``kind`` and ``spec_cls``.
3. Importing the probe module at least once (its subclass hook registers
   itself).

The runner uses :func:`build_probe` to map each raw spec dict to a
``(Probe, ProbeSpec)`` pair. Validation errors are turned into
:class:`~dlm_sway.core.errors.SpecValidationError` with the probe name
as the source so error messages localize to the offending entry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, ValidationError

from dlm_sway.core.errors import SpecValidationError
from dlm_sway.core.result import ProbeResult
from dlm_sway.core.scoring import DifferentialBackend


class ProbeSpec(BaseModel):
    """Common fields for every probe's spec entry in ``sway.yaml``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    """Unique within a suite; surfaces in the report."""
    kind: str
    """Discriminator — must match a registered :class:`Probe` subclass."""
    enabled: bool = True
    """If ``False`` the runner records a :class:`~dlm_sway.core.result.Verdict.SKIP`."""
    weight: float = 1.0
    """Weight inside the probe's component (adherence / attribution / …)."""


@dataclass(frozen=True, slots=True)
class RunContext:
    """What a probe can read beyond its own spec.

    Probes should receive exactly what they need and nothing more; fat
    contexts encourage coupling between unrelated probes.

    Attributes
    ----------
    backend:
        The differential backend holding base + fine-tuned views.
    seed:
        Seed for deterministic probe RNGs (paraphrase sampling, etc).
    top_k:
        Default truncation for next-token distributions.
    sections:
        Optional list of typed sections (populated by the .dlm bridge;
        ``None`` when sway is invoked against bare HF+PEFT).
    doc_text:
        Raw document text, if available.
    null_stats:
        Null-adapter baseline stats for z-score calibration, keyed by
        probe *kind*. Populated by the runner after it's executed the
        ``null_adapter`` probe (if configured).
    """

    backend: DifferentialBackend
    seed: int = 0
    top_k: int = 256
    sections: tuple[Any, ...] | None = None
    doc_text: str | None = None
    null_stats: dict[str, dict[str, float]] = field(default_factory=dict)


_REGISTRY: dict[str, type[Probe]] = {}


class Probe(ABC):
    """Concrete probe. One instance per probe spec in the suite."""

    kind: ClassVar[str]
    """The string used in ``sway.yaml``'s ``kind`` field."""
    spec_cls: ClassVar[type[ProbeSpec]]
    """The pydantic model class that validates this probe's spec."""
    category: ClassVar[str] = "adherence"
    """One of: ``adherence``, ``attribution``, ``calibration``,
    ``ablation``, ``baseline``. Drives composite scoring."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # The abstract class itself has no `kind`; skip registration.
        if "kind" not in cls.__dict__:
            return
        kind = cls.kind
        if kind in _REGISTRY:
            raise ValueError(f"duplicate probe kind {kind!r}: {_REGISTRY[kind]!r} vs {cls!r}")
        _REGISTRY[kind] = cls

    @abstractmethod
    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult: ...


def registry() -> dict[str, type[Probe]]:
    """Read-only view of registered probes."""
    return dict(_REGISTRY)


def build_probe(raw: dict[str, Any]) -> tuple[Probe, ProbeSpec]:
    """Validate a raw YAML probe entry and return (Probe instance, spec)."""
    kind = raw.get("kind")
    if not isinstance(kind, str):
        raise SpecValidationError(
            "probe entry missing string 'kind' field",
            source=str(raw.get("name", "<unknown>")),
        )
    if kind not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise SpecValidationError(
            f"unknown probe kind {kind!r} (registered: {known})",
            source=str(raw.get("name", "<unknown>")),
        )
    probe_cls = _REGISTRY[kind]
    try:
        spec = probe_cls.spec_cls.model_validate(raw)
    except ValidationError as exc:
        raise SpecValidationError(str(exc), source=str(raw.get("name", "<unknown>"))) from exc
    return probe_cls(), spec
