"""HuggingFace + PEFT differential backend.

Loads the base once, attaches the LoRA adapter once, and toggles between
"base" and "fine-tuned" views on the same module via PEFT's
:meth:`~peft.PeftModel.disable_adapter` / :meth:`~peft.PeftModel.set_adapter`.

This is the single most important backend in sway. Every numeric probe
benefits from the shared-weights toggle — memory is halved compared to
loading two copies, and KV-cache layouts stay aligned so pairwise KL math
is straight-forward.

Heavy imports (``torch``, ``transformers``, ``peft``) are deferred until
``HuggingFaceDifferentialBackend`` is actually instantiated so
``import dlm_sway`` stays light for users of the dummy backend or spec
validation.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from dlm_sway.core.errors import BackendNotAvailableError, ProbeError
from dlm_sway.core.model import ModelSpec
from dlm_sway.core.scoring import RollingLogprob, TokenDist

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


Device = Literal["cuda", "mps", "cpu"]


def _detect_device() -> Device:
    try:
        import torch
    except ImportError as exc:
        raise BackendNotAvailableError("hf", extra="hf") from exc
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(requested: str, device: Device) -> Any:
    """Map the user's ``dtype`` preference to a torch dtype."""
    import torch  # noqa: PLC0415 — lazy

    if requested == "fp16":
        return torch.float16
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp32":
        return torch.float32
    # auto: bf16 on CUDA (Ampere+) / MPS; fp32 on CPU for numerical stability.
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def _require_hf() -> tuple[Any, Any, Any]:
    """Import torch + transformers + peft, raising a friendly error if missing."""
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise BackendNotAvailableError("hf", extra="hf") from exc
    try:
        import peft
    except ImportError as exc:
        raise BackendNotAvailableError(
            "hf", extra="hf", hint="peft is required for the adapter toggle."
        ) from exc
    return torch, transformers, peft


# --- the view object ------------------------------------------------------


@dataclass(slots=True)
class _HFView:
    """One side (base or ft) of a :class:`HuggingFaceDifferentialBackend`.

    Both sides reuse the same underlying module; the difference is
    whether the adapter is active.
    """

    id: str
    _model: Any
    _tokenizer: Any
    _device: str
    _pad_token_id: int

    # -- Model ---------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> str:
        import torch

        torch.manual_seed(seed)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        do_sample = temperature > 0.0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        with torch.inference_mode():
            out_ids = self._model.generate(**inputs, **gen_kwargs)
        new_tokens = out_ids[0, inputs["input_ids"].shape[1] :]
        return str(self._tokenizer.decode(new_tokens, skip_special_tokens=True))

    def close(self) -> None:
        return None

    # -- ScoringBackend ------------------------------------------------
    def logprob_of(self, prompt: str, completion: str) -> float:
        import torch
        import torch.nn.functional as F

        prompt_ids = self._tokenizer(prompt, return_tensors="pt").input_ids.to(self._device)
        full_ids = self._tokenizer(prompt + completion, return_tensors="pt").input_ids.to(
            self._device
        )
        if full_ids.shape[1] <= prompt_ids.shape[1]:
            raise ProbeError(
                "logprob_of",
                f"completion tokenized to zero tokens (prompt={prompt!r}, completion={completion!r})",
            )
        target_ids = full_ids[:, prompt_ids.shape[1] :]
        with torch.inference_mode():
            logits = self._model(full_ids).logits  # (1, T, V)
        # Align: logit at position t predicts token at t+1. We want
        # predictions for the completion slice.
        shift_logits = logits[:, prompt_ids.shape[1] - 1 : -1, :]  # (1, C, V)
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        gathered = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        return float(gathered.sum().item())

    def rolling_logprob(self, text: str) -> RollingLogprob:
        import torch
        import torch.nn.functional as F

        ids = self._tokenizer(text, return_tensors="pt").input_ids.to(self._device)
        if ids.shape[1] < 2:
            return RollingLogprob(
                token_ids=ids[0].cpu().numpy().astype(np.int64),
                logprobs=np.array([], dtype=np.float32),
                num_tokens=int(ids.shape[1]),
                total_logprob=0.0,
            )
        with torch.inference_mode():
            logits = self._model(ids).logits  # (1, T, V)
        log_probs = F.log_softmax(logits[:, :-1].float(), dim=-1)  # predicts tokens 1..T
        gathered = log_probs.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).squeeze(0)
        return RollingLogprob(
            token_ids=ids[0].cpu().numpy().astype(np.int64),
            logprobs=gathered.cpu().numpy().astype(np.float32),
            num_tokens=int(ids.shape[1]),
            total_logprob=float(gathered.sum().item()),
        )

    def next_token_dist(self, prompt: str, *, top_k: int = 256) -> TokenDist:
        import torch
        import torch.nn.functional as F

        ids = self._tokenizer(prompt, return_tensors="pt").input_ids.to(self._device)
        with torch.inference_mode():
            logits = self._model(ids).logits[:, -1, :]  # (1, V)
        log_probs = F.log_softmax(logits.float(), dim=-1).squeeze(0)
        k = min(top_k, int(log_probs.shape[0]))
        top = torch.topk(log_probs, k=k)
        tail_mass = float(1.0 - torch.exp(top.values).sum().item())
        tail_logprob = float(np.log(max(tail_mass, 1e-12))) if tail_mass > 1e-12 else 0.0
        return TokenDist(
            token_ids=top.indices.cpu().numpy().astype(np.int64),
            logprobs=top.values.cpu().numpy().astype(np.float32),
            vocab_size=int(log_probs.shape[0]),
            tail_logprob=tail_logprob,
        )


# --- the backend -----------------------------------------------------------


class HuggingFaceDifferentialBackend:
    """A :class:`~dlm_sway.core.scoring.DifferentialBackend` for HF+PEFT.

    The adapter toggle relies on
    :meth:`peft.PeftModel.disable_adapter` producing a context where the
    forward pass skips the LoRA deltas, and
    :meth:`peft.PeftModel.set_adapter` (or just exiting the disable
    context) re-enabling them. A dedicated sanity test asserts that
    these actually change logits on a fixture.
    """

    def __init__(self, *, base_spec: ModelSpec, adapter_path: Path) -> None:
        torch, transformers, peft = _require_hf()
        self._torch = torch
        self._spec = base_spec
        self._adapter_path = Path(adapter_path).expanduser().resolve()

        device_str: Device = (
            _detect_device() if base_spec.device == "auto" else base_spec.device  # type: ignore[assignment]
        )
        self._device: str = device_str
        dtype = _resolve_dtype(base_spec.dtype, device_str)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(self._adapter_path)
            if (self._adapter_path / "tokenizer_config.json").exists()
            else base_spec.base,
            trust_remote_code=base_spec.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            base_spec.base,
            torch_dtype=dtype,
            trust_remote_code=base_spec.trust_remote_code,
        )
        base_model.to(self._device)
        peft_model = peft.PeftModel.from_pretrained(
            base_model,
            str(self._adapter_path),
            is_trainable=False,
        )
        peft_model.eval()

        self._tokenizer: PreTrainedTokenizerBase = tokenizer
        self._peft_model: PreTrainedModel = peft_model
        self._pad_token_id: int = int(tokenizer.pad_token_id)
        self._active: str | None = None

    # -- DifferentialBackend -------------------------------------------

    @contextmanager
    def as_base(self) -> Iterator[_HFView]:
        self._enter("base")
        try:
            with self._peft_model.disable_adapter():
                yield self._make_view("base")
        finally:
            self._exit()

    @contextmanager
    def as_finetuned(self) -> Iterator[_HFView]:
        self._enter("ft")
        try:
            yield self._make_view("ft")
        finally:
            self._exit()

    @contextmanager
    def as_scaled_adapter(self, lam: float) -> Iterator[_HFView]:
        """Temporarily multiply every LoRA layer's scaling factor by ``lam``.

        Works by walking the PEFT module tree and mutating each
        ``LoraLayer.scaling[adapter_name]`` in place. The original
        scalings are restored when the context exits — or when an
        exception propagates, to keep the model in a sane state.
        """
        self._enter(f"scaled({lam})")
        saved: list[tuple[object, str, float]] = []
        try:
            import peft  # noqa: PLC0415 — already a hard dep of this backend

            lora_cls = getattr(peft.tuners.lora, "LoraLayer", None)
            if lora_cls is None:
                raise RuntimeError("peft.tuners.lora.LoraLayer not found; check peft>=0.13 pin")
            for module in self._peft_model.modules():
                if not isinstance(module, lora_cls):
                    continue
                scaling = getattr(module, "scaling", None)
                if not isinstance(scaling, dict):
                    continue
                for key, original in scaling.items():
                    saved.append((module, key, float(original)))
                    scaling[key] = float(original) * lam
            yield self._make_view(f"scaled_{lam:.2f}")
        finally:
            for module, key, original in saved:
                module.scaling[key] = original  # type: ignore[attr-defined]
            self._exit()

    def close(self) -> None:
        """Release GPU memory. Safe to call more than once."""
        if getattr(self, "_peft_model", None) is not None:
            del self._peft_model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    # -- internals -----------------------------------------------------

    def _make_view(self, mode: str) -> _HFView:
        return _HFView(
            id=mode,
            _model=self._peft_model,
            _tokenizer=self._tokenizer,
            _device=self._device,
            _pad_token_id=self._pad_token_id,
        )

    def _enter(self, mode: str) -> None:
        if self._active is not None:
            raise RuntimeError(
                f"HuggingFaceDifferentialBackend view {self._active!r} already active; "
                f"exit it before entering {mode!r}."
            )
        self._active = mode

    def _exit(self) -> None:
        self._active = None


__all__ = ["HuggingFaceDifferentialBackend"]
