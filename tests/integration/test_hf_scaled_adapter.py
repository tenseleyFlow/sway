"""Integration test: ``HF.as_scaled_adapter`` and the response-curve invariants.

The adapter-ablation probe (the sway signature primitive) leans on
``as_scaled_adapter(lam)`` to walk a λ sweep. Two things must hold:

1. **Monotonicity of the *signal* across λ**: divergence at λ=1.25
   should be strictly larger than divergence at λ=0 (which is base).
   We don't claim a smooth curve here — the unit tests on the probe
   itself cover curve shape — only that the scaling actually scales.
2. **State restoration on exit**: every ``LoraLayer.scaling[adapter_name]``
   value the context manager touched must be back to its original
   number after the ``with`` block. Anything else corrupts subsequent
   probes.

Marked ``slow+online`` to share the tiny-model fixture with the rest
of the integration suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dlm_sway.backends.hf import HuggingFaceDifferentialBackend
from dlm_sway.core.model import ModelSpec
from dlm_sway.probes._divergence import divergence

pytestmark = [pytest.mark.slow, pytest.mark.online]


def _build_random_lora_adapter(base_dir: Path, out_dir: Path) -> None:
    """Same shape as the toggle-test adapter — a small but non-zero LoRA."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(str(base_dir), torch_dtype=torch.float32)
    cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base, cfg)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn_like(param) * 0.05)
    peft_model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))


@pytest.fixture(scope="module")
def random_adapter(tiny_model_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    adapter_dir = tmp_path_factory.mktemp("scaled-random-adapter")
    _build_random_lora_adapter(tiny_model_dir, adapter_dir)
    return adapter_dir


@pytest.fixture(scope="module")
def hf_backend(tiny_model_dir: Path, random_adapter: Path) -> HuggingFaceDifferentialBackend:
    backend = HuggingFaceDifferentialBackend(
        base_spec=ModelSpec(base=str(tiny_model_dir), kind="hf", dtype="fp32", device="cpu"),
        adapter_path=random_adapter,
    )
    yield backend
    backend.close()


def _captured_scalings(backend: HuggingFaceDifferentialBackend) -> dict[tuple[int, str], float]:
    """Snapshot every ``LoraLayer.scaling[key]`` keyed by (id, key)."""
    import peft

    lora_cls: Any = peft.tuners.lora.LoraLayer
    out: dict[tuple[int, str], float] = {}
    for module in backend._peft_model.modules():  # type: ignore[attr-defined]
        if not isinstance(module, lora_cls):
            continue
        scaling = getattr(module, "scaling", None)
        if not isinstance(scaling, dict):
            continue
        for key, value in scaling.items():
            out[(id(module), key)] = float(value)
    return out


def test_lambda_sweep_monotonic_in_signal(hf_backend: HuggingFaceDifferentialBackend) -> None:
    """Divergence(@λ=0, @λ=1.25) > divergence(@λ=0, @λ=0) (which is 0)."""
    prompt = "The quick brown fox"
    with hf_backend.as_scaled_adapter(0.0) as v0:
        d0 = v0.next_token_dist(prompt, top_k=64)
    with hf_backend.as_scaled_adapter(1.25) as v_over:
        d_over = v_over.next_token_dist(prompt, top_k=64)

    div_at_zero = divergence(d0, d0, kind="js")
    div_at_overshoot = divergence(d0, d_over, kind="js")
    assert div_at_zero == pytest.approx(0.0, abs=1e-9), (
        f"self-divergence at λ=0 should be ~0; got {div_at_zero}"
    )
    assert div_at_overshoot > 1e-6, f"λ=1.25 should drift far from λ=0; got {div_at_overshoot}"


def test_lambda_one_matches_finetuned_within_tolerance(
    hf_backend: HuggingFaceDifferentialBackend,
) -> None:
    """``as_scaled_adapter(1.0)`` should be functionally identical to ``as_finetuned()``."""
    prompt = "hello"
    with hf_backend.as_finetuned() as ft:
        d_ft = ft.next_token_dist(prompt, top_k=32)
    with hf_backend.as_scaled_adapter(1.0) as v1:
        d1 = v1.next_token_dist(prompt, top_k=32)
    np.testing.assert_allclose(d_ft.logprobs, d1.logprobs, rtol=1e-5, atol=1e-6)


def test_scaling_restored_on_clean_exit(hf_backend: HuggingFaceDifferentialBackend) -> None:
    """Every LoraLayer.scaling[key] is back to its original value after exit."""
    before = _captured_scalings(hf_backend)
    with hf_backend.as_scaled_adapter(0.42) as v:
        v.next_token_dist("anything", top_k=8)
    after = _captured_scalings(hf_backend)
    assert before == after, "scaling table not restored after as_scaled_adapter context"


def test_scaling_restored_on_exception(hf_backend: HuggingFaceDifferentialBackend) -> None:
    """Same restoration invariant when the body raises."""
    before = _captured_scalings(hf_backend)
    with pytest.raises(RuntimeError, match="boom"):
        with hf_backend.as_scaled_adapter(0.7):
            raise RuntimeError("boom")
    after = _captured_scalings(hf_backend)
    assert before == after, "scaling table not restored after exception inside the context"
