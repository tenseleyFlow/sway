"""The :class:`Model` abstraction and :class:`ModelSpec` user-facing config.

Probes operate on objects that satisfy :class:`Model` (for generation)
and :class:`~dlm_sway.core.scoring.ScoringBackend` (for logit-level
access). Backends return concrete instances of both — they are
deliberately separate Protocols because not every backend exposes logits
(e.g. an Ollama HTTP backend would implement ``Model`` but not
``ScoringBackend``).

The user-facing surface is :class:`ModelSpec`, a pydantic model that
describes how to materialize a base + adapter pair. No ``.dlm``
concepts live at this layer — those belong in
:mod:`dlm_sway.integrations.dlm`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

BackendKind = Literal["hf", "mlx", "api", "dummy", "custom"]
"""Registered scoring-backend kinds.

``api`` targets an OpenAI-compatible HTTP endpoint (OpenAI, vLLM
serve, Ollama). ``custom`` is an escape hatch — the runner looks up
an entry point when it sees ``custom`` in a spec.
"""


class ModelSpec(BaseModel):
    """How to materialize one model (base or fine-tuned)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: BackendKind = "hf"
    base: str
    """HuggingFace repo id (``HuggingFaceTB/SmolLM2-135M-Instruct``) or
    a local path to a model directory."""

    adapter: Path | None = None
    """Path to a PEFT adapter directory (containing ``adapter_config.json``
    and ``adapter_model.safetensors``). ``None`` → base-only model."""

    dtype: Literal["auto", "fp16", "bf16", "fp32"] = "auto"
    device: str = "auto"
    """``"auto"`` chooses CUDA → MPS → CPU in that order."""

    trust_remote_code: bool = False
    """HuggingFace ``trust_remote_code`` passthrough. Off by default —
    the user must opt in explicitly, matching sway's no-surprises
    posture."""

    entry_point: str | None = Field(default=None)
    """Required when ``kind='custom'``. Import path like
    ``mypkg.mybackend:MyBackend``."""

    endpoint: str | None = Field(default=None)
    """Required when ``kind='api'``. Base URL of an OpenAI-compatible
    completions server (``https://api.openai.com``,
    ``http://localhost:11434`` for Ollama, or wherever ``vllm serve``
    listens). The ``/v1/completions`` path is appended by the backend.

    The API key comes from the environment (``SWAY_API_KEY``, falling
    back to ``OPENAI_API_KEY``) so secrets don't live in the YAML spec."""

    @field_validator("adapter")
    @classmethod
    def _normalize_adapter_path(cls, v: Path | None) -> Path | None:
        """Expand ``~`` and resolve relative segments at spec-load time.

        Before B22 every backend re-did this work in its constructor;
        normalizing once at the spec boundary means the cache key in
        :func:`dlm_sway.probes._null_cache.compute_key` (which encodes
        the adapter path) is stable regardless of how the user spelled
        it in YAML or on the CLI.
        """
        if v is None:
            return None
        return Path(v).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A materialized model plus the tokenizer that produced it.

    Returned by backend ``load()`` methods. Probes usually don't touch
    this directly — they go through the :class:`Model` /
    :class:`~dlm_sway.core.scoring.ScoringBackend` Protocols.
    """

    id: str
    """Stable handle: ``"base"`` or ``"ft"`` typically."""
    spec: ModelSpec
    model: Any
    """Framework-native handle (torch ``nn.Module``, MLX array module …).

    Typed as ``Any`` because the frameworks themselves ship unstubbed.
    Backend implementations narrow this at their boundary."""
    tokenizer: Any
    meta: dict[str, Any]
    """Backend-captured metadata: device, dtype, adapter version, bytes
    on disk, num trainable params. Surfaced in the suite report."""


@runtime_checkable
class Model(Protocol):
    """Minimum interface for text generation.

    Implemented by backend-wrapped model objects. Probes that need logits
    also require :class:`~dlm_sway.core.scoring.ScoringBackend`.
    """

    id: str

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> str:
        """Generate a completion.

        Defaults (``temperature=0``, ``top_p=1``) are greedy-decode for
        reproducibility. Callers wanting sampled output must pass
        non-defaults *and* a seed.
        """
        ...

    def close(self) -> None:
        """Release any resources held by this model."""
        ...
