"""MLX backend for Apple Silicon (darwin-arm64).

Partial implementation covering the common case: a PEFT adapter that's
already been converted to MLX's ``.npz`` format. Unlike the HF backend,
MLX has no runtime ``disable_adapter`` context — adapters get fused into
the linear layers at load time — so this backend keeps **both** a base
model and an adapted model in memory. Fine for the small (<3B) models
MLX is typically used with on Apple Silicon; document the cost clearly.

If users point this backend at raw PEFT safetensors, ``mlx_lm.load``
will refuse them with its own error. A future milestone can wire a
PEFT-→-MLX converter; for now the contract is "bring your own .npz".
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from dlm_sway.core.errors import BackendNotAvailableError, ProbeError
from dlm_sway.core.model import ModelSpec
from dlm_sway.core.scoring import RollingLogprob, TokenDist

if TYPE_CHECKING:
    pass


def _require_mlx() -> tuple[Any, Any]:
    try:
        import mlx.core as mx
        import mlx_lm
    except ImportError as exc:
        raise BackendNotAvailableError(
            "mlx",
            extra="mlx",
            hint="MLX backend needs mlx + mlx-lm on darwin-arm64.",
        ) from exc
    return mx, mlx_lm


@dataclass(slots=True)
class _MLXView:
    """One side (base or ft) of the MLX backend.

    Both sides carry the same tokenizer (MLX stores it alongside the
    converted model files, so sharing avoids double-loading).
    """

    id: str
    _model: Any
    _tokenizer: Any

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> str:
        del seed  # mlx_lm.generate seeds via its own global state
        _, mlx_lm = _require_mlx()
        kwargs: dict[str, Any] = {"max_tokens": max_new_tokens, "verbose": False}
        if temperature > 0.0:
            kwargs["temp"] = temperature
            kwargs["top_p"] = top_p
        out = mlx_lm.generate(self._model, self._tokenizer, prompt=prompt, **kwargs)
        return str(out)

    def close(self) -> None:
        return None

    # -- ScoringBackend ------------------------------------------------

    def _forward_logits(self, prompt: str) -> np.ndarray:
        """Run the model once and return ``(seq_len, vocab)`` logits."""
        mx, _ = _require_mlx()
        input_ids = self._tokenizer.encode(prompt)
        tokens = mx.array(input_ids)[None, :]  # (1, T)
        out = self._model(tokens)
        # mlx_lm models return an mx.array; convert to numpy for downstream math.
        return np.asarray(out[0])

    def logprob_of(self, prompt: str, completion: str) -> float:
        input_ids = self._tokenizer.encode(prompt)
        full_ids = self._tokenizer.encode(prompt + completion)
        if len(full_ids) <= len(input_ids):
            raise ProbeError(
                "logprob_of",
                f"completion tokenized to zero tokens (prompt={prompt!r}, completion={completion!r})",
            )
        logits = self._forward_logits(prompt + completion)  # (T, V)
        # Position t predicts token t+1 — slice off the last row and the prompt span.
        shift = logits[len(input_ids) - 1 : -1, :]
        target_ids = np.asarray(full_ids[len(input_ids) :], dtype=np.int64)
        log_probs = _log_softmax(shift.astype(np.float64), axis=-1)
        gathered = log_probs[np.arange(len(target_ids)), target_ids]
        return float(gathered.sum())

    def rolling_logprob(self, text: str) -> RollingLogprob:
        ids = self._tokenizer.encode(text)
        if len(ids) < 2:
            return RollingLogprob(
                token_ids=np.asarray(ids, dtype=np.int64),
                logprobs=np.array([], dtype=np.float32),
                num_tokens=len(ids),
                total_logprob=0.0,
            )
        logits = self._forward_logits(text)
        log_probs = _log_softmax(logits[:-1].astype(np.float64), axis=-1)
        ids_arr = np.asarray(ids, dtype=np.int64)
        gathered = log_probs[np.arange(len(ids) - 1), ids_arr[1:]]
        return RollingLogprob(
            token_ids=ids_arr,
            logprobs=gathered.astype(np.float32),
            num_tokens=len(ids),
            total_logprob=float(gathered.sum()),
        )

    def next_token_dist(self, prompt: str, *, top_k: int = 256) -> TokenDist:
        logits = self._forward_logits(prompt)
        last_logits = logits[-1].astype(np.float64)
        log_probs = _log_softmax(last_logits, axis=-1)
        k = min(top_k, log_probs.shape[0])
        # np.argpartition for top-k then sort the partition.
        part = np.argpartition(log_probs, -k)[-k:]
        top_ids = part[np.argsort(log_probs[part])[::-1]]
        top_lp = log_probs[top_ids]
        tail_mass = float(1.0 - np.exp(top_lp).sum())
        tail_logprob = float(np.log(max(tail_mass, 1e-12))) if tail_mass > 1e-12 else 0.0
        return TokenDist(
            token_ids=top_ids.astype(np.int64),
            logprobs=top_lp.astype(np.float32),
            vocab_size=int(log_probs.shape[0]),
            tail_logprob=tail_logprob,
        )


class MLXDifferentialBackend:
    """A :class:`~dlm_sway.core.scoring.DifferentialBackend` for MLX models.

    Loads two copies of the same base model — one bare, one with the
    adapter fused — because MLX has no runtime toggle. Memory cost: 2×
    base weights. On typical Apple Silicon workloads with ≤3B models
    this is acceptable.
    """

    def __init__(self, *, base_spec: ModelSpec, adapter_path: Path) -> None:
        mx, mlx_lm = _require_mlx()
        self._mx = mx
        self._spec = base_spec
        self._adapter_path = Path(adapter_path).expanduser().resolve()

        # Load bare base (no adapter).
        self._base_model, self._tokenizer = mlx_lm.load(base_spec.base)
        # Load ft with adapter attached. ``adapter_path`` is mlx_lm's kwarg.
        self._ft_model, _ = mlx_lm.load(base_spec.base, adapter_path=str(self._adapter_path))
        self._active: str | None = None

    @contextmanager
    def as_base(self) -> Iterator[_MLXView]:
        self._enter("base")
        try:
            yield _MLXView(id="base", _model=self._base_model, _tokenizer=self._tokenizer)
        finally:
            self._exit()

    @contextmanager
    def as_finetuned(self) -> Iterator[_MLXView]:
        self._enter("ft")
        try:
            yield _MLXView(id="ft", _model=self._ft_model, _tokenizer=self._tokenizer)
        finally:
            self._exit()

    def close(self) -> None:
        """MLX reclaims memory when references drop; nothing to do here."""
        return

    def _enter(self, mode: str) -> None:
        if self._active is not None:
            raise RuntimeError(
                f"MLXDifferentialBackend view {self._active!r} already active; "
                f"exit it before entering {mode!r}."
            )
        self._active = mode

    def _exit(self) -> None:
        self._active = None


def _log_softmax(x: np.ndarray, *, axis: int) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    y = x - x_max
    log_sum = np.log(np.sum(np.exp(y), axis=axis, keepdims=True))
    return np.asarray(y - log_sum, dtype=np.float64)


__all__ = ["MLXDifferentialBackend"]
