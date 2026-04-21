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

from dlm_sway.backends._instrumentation import BackendInstrumentation
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
    whether the adapter is active. Scoring calls route through
    ``_inst`` (the backend's shared cache + tracer + stats) so repeated
    forward passes on the same ``(view_id, prompt, top_k)`` are served
    from the LRU instead of re-executed — the Sprint 07 performance win.
    """

    id: str
    _model: Any
    _tokenizer: Any
    _device: str
    _pad_token_id: int
    _inst: BackendInstrumentation

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
        # Generation is intentionally *not* cached — probes call it
        # through (prompt, max_new_tokens, temperature, seed) tuples
        # that rarely collide across a suite, and cache hits on sampled
        # output would hide seed bugs behind stale strings.
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
        # Fold (prompt, completion) into one cache-key string so a repeat
        # (q, a) pair hits the cache without the completion args needing
        # their own slot in ``ForwardCache``.
        key_prompt = f"{prompt}\x00{completion}"
        return self._inst.cached(
            "logprob_of",
            self.id,
            key_prompt,
            0,
            lambda: self._compute_logprob_of(prompt, completion),
        )

    def _compute_logprob_of(self, prompt: str, completion: str) -> float:
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
        return self._inst.cached(
            "rolling_logprob",
            self.id,
            text,
            0,
            lambda: self._compute_rolling_logprob(text),
        )

    def _compute_rolling_logprob(self, text: str) -> RollingLogprob:
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
        return self._inst.cached(
            "next_token_dist",
            self.id,
            prompt,
            top_k,
            lambda: self._compute_next_token_dist(prompt, top_k=top_k),
        )

    def _compute_next_token_dist(self, prompt: str, *, top_k: int = 256) -> TokenDist:
        import torch
        import torch.nn.functional as F

        ids = self._tokenizer(prompt, return_tensors="pt").input_ids.to(self._device)
        with torch.inference_mode():
            logits = self._model(ids).logits[:, -1, :]  # (1, V)
        log_probs = F.log_softmax(logits.float(), dim=-1).squeeze(0)
        vocab = int(log_probs.shape[0])
        k = min(top_k, vocab)
        top = torch.topk(log_probs, k=k)
        # B6: distinguish "no tail" (k covers vocab) from "measurable tail"
        # from "underflowed-to-zero tail." See TokenDist.tail_logprob docs.
        if k == vocab:
            tail_logprob: float | None = None
        else:
            tail_mass = float(1.0 - torch.exp(top.values).sum().item())
            tail_logprob = float(np.log(tail_mass)) if tail_mass > 1e-12 else 0.0
        return TokenDist(
            token_ids=top.indices.cpu().numpy().astype(np.int64),
            logprobs=top.values.cpu().numpy().astype(np.float32),
            vocab_size=vocab,
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

    #: B19 — the shared-weights toggle is not thread-safe. The runner
    #: treats ``spec.defaults.concurrent_probes > 1`` as a no-op when
    #: this attribute is ``False``. See ``.docs/design/backend-concurrency.md``.
    safe_for_concurrent_views: bool = False

    def __init__(self, *, base_spec: ModelSpec, adapter_path: Path) -> None:
        torch, transformers, peft = _require_hf()
        self._torch = torch
        self._spec = base_spec
        # Path normalization lives in ``ModelSpec.adapter`` (B22). When
        # the backend is constructed via ``backends.build``, the value
        # is already absolute. Direct constructions (some tests) may
        # pass a relative path, so ``Path(...).resolve()`` stays as a
        # cheap idempotent fallback.
        self._adapter_path = Path(adapter_path).resolve()

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
        # Shared cache + trace + stats (Sprint 07). The instrumentation
        # instance outlives individual view entries — entering/exiting
        # ``as_base()`` doesn't reset the cache, because each view id
        # (``"base"`` / ``"ft"`` / ``"scaled_0.50"`` / ``"null_42"``)
        # is part of the cache key. A future toggle bug that fails to
        # flip the view would surface as a cache hit returning the
        # wrong side's value — integration tests cover this.
        self._inst = BackendInstrumentation()

    # -- DifferentialBackend -------------------------------------------

    @contextmanager
    def as_base(self) -> Iterator[_HFView]:
        self._enter("base")
        try:
            # peft.PeftModel.disable_adapter is a context manager; newer
            # transformers builds ship stubs that mis-type it as a Tensor,
            # so we suppress the operator check on the `with` line. The
            # ``unused-ignore`` tag makes the suppression itself a no-op
            # when [hf] isn't installed (mypy can't see the conflicting
            # stub and would otherwise flag the ignore as redundant).
            with self._peft_model.disable_adapter():  # type: ignore[operator,unused-ignore]
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
        # ``module`` is dynamic (peft LoraLayer subclass) — Any avoids
        # mypy treating its ``.scaling`` as a Tensor when peft is loaded.
        saved: list[tuple[Any, str, float]] = []
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
                module.scaling[key] = original
            self._exit()

    @contextmanager
    def as_null_adapter(self, seed: int, *, init_scale: float = 0.02) -> Iterator[_HFView]:
        """Temporarily replace every LoRA ``A``/``B`` tensor with random noise.

        Same rank, alpha, and target modules as the real adapter — only
        the weights differ. This is the denominator in every z-score
        path: "how much signal does structural noise produce?"

        Implementation walks the PEFT module tree for ``lora_A``/``lora_B``
        parameters, saves a clone of each current value, overwrites in
        place with a zero-mean Gaussian at ``init_scale``, and restores
        on exit (including on exception).
        """
        import torch

        self._enter(f"null({seed})")
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        saved: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        try:
            for pname, param in self._peft_model.named_parameters():
                if not any(key in pname for key in ("lora_A", "lora_B")):
                    continue
                saved.append((param, param.detach().clone()))
                with torch.no_grad():
                    noise = torch.randn(
                        *param.shape,
                        generator=gen,
                        dtype=torch.float32,
                    ).to(dtype=param.dtype, device=param.device)
                    param.copy_(noise * init_scale)
            yield self._make_view(f"null_{seed}")
        finally:
            with torch.no_grad():
                for param, original in saved:
                    param.copy_(original)
            self._exit()

    def close(self) -> None:
        """Release GPU memory + flush the trace writer. Safe to call more than once."""
        inst = getattr(self, "_inst", None)
        if inst is not None:
            inst.close()
        if getattr(self, "_peft_model", None) is not None:
            del self._peft_model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    # -- PreflightCheckable -------------------------------------------

    _PREFLIGHT_PROMPT = "hello"
    _PREFLIGHT_TOP_K = 8

    def cache_identity(self) -> str:
        """Stable string identifying this backend for on-disk caching.

        The base model id + the adapter's resolved absolute path is
        enough to key a null-calibration cache: swapping either
        invalidates the previously-computed stats.
        """
        return f"hf:{self._spec.base}:{self._adapter_path}"

    def preflight_finite_check(self) -> tuple[bool, str]:
        """One forward pass per view; assert both produce finite logits.

        Catches the +11639σ class of bug at suite-load time: a NaN-weighted
        adapter would produce non-finite logprobs here, the runner sees
        ``ok=False``, and the suite aborts with a single synthetic ERROR
        probe — never reaching a probe that would pass on garbage.
        """
        import math

        try:
            with self.as_base() as base_view:
                base_dist = base_view.next_token_dist(
                    self._PREFLIGHT_PROMPT, top_k=self._PREFLIGHT_TOP_K
                )
            with self.as_finetuned() as ft_view:
                ft_dist = ft_view.next_token_dist(
                    self._PREFLIGHT_PROMPT, top_k=self._PREFLIGHT_TOP_K
                )
        except Exception as exc:  # noqa: BLE001 — backend may raise anything
            return False, f"preflight forward pass raised {type(exc).__name__}: {exc}"

        for label, dist in (("base", base_dist), ("ft", ft_dist)):
            n_bad = int((~np.isfinite(dist.logprobs)).sum())
            if n_bad > 0:
                return (
                    False,
                    f"{label} view produced {n_bad}/{dist.logprobs.size} non-finite "
                    f"logprob(s) on prompt {self._PREFLIGHT_PROMPT!r} — adapter is "
                    f"likely broken (NaN/inf weights). sway refuses to score a model "
                    f"producing non-finite outputs.",
                )
            tail = dist.tail_logprob
            # B6: ``None`` is a sentinel for "k covered the whole vocab,"
            # not a numeric value to range-check.
            if tail is not None and not math.isfinite(tail):
                return (
                    False,
                    f"{label} view produced non-finite tail_logprob = {tail}",
                )

        return True, ""

    # -- internals -----------------------------------------------------

    def _make_view(self, mode: str) -> _HFView:
        return _HFView(
            id=mode,
            _model=self._peft_model,
            _tokenizer=self._tokenizer,
            _device=self._device,
            _pad_token_id=self._pad_token_id,
            _inst=self._inst,
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
