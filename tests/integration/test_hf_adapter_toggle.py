"""Integration test: PEFT ``disable_adapter`` actually changes logits.

This is the load-bearing sanity check for the whole differential design.
If a future ``peft`` release subtly breaks the disable-context semantics,
sway's KL / SIS / ablation probes would all silently report zero signal.
We catch that here, before the rest of the test battery runs.

The test builds a random-init LoRA adapter on a tiny model so no network
dependency beyond the base model snapshot itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlm_sway.backends.hf import HuggingFaceDifferentialBackend
from dlm_sway.core.model import ModelSpec

pytestmark = [pytest.mark.slow, pytest.mark.online]


def _build_random_lora_adapter(base_dir: Path, out_dir: Path) -> None:
    """Construct a LoRA adapter with random-init weights on ``base_dir``.

    The weights are kept small so the toggle-delta is clear but the
    adapter is structurally valid (correct ``adapter_config.json``,
    tokenizer files, safetensors layout).
    """
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

    # Explicitly scale lora_B out of its PEFT-default zero-init so the
    # adapter actually changes outputs. Real training does this via
    # gradients; we do it with a scaled normal.
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn_like(param) * 0.05)

    peft_model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))


@pytest.fixture(scope="module")
def random_adapter(tiny_model_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    adapter_dir = tmp_path_factory.mktemp("random-adapter")
    _build_random_lora_adapter(tiny_model_dir, adapter_dir)
    return adapter_dir


def test_disable_adapter_changes_logits(
    tiny_model_dir: Path, random_adapter: Path
) -> None:
    """The keystone invariant: base view ≠ ft view on the same prompt."""
    import numpy as np

    backend = HuggingFaceDifferentialBackend(
        base_spec=ModelSpec(base=str(tiny_model_dir), kind="hf", dtype="fp32", device="cpu"),
        adapter_path=random_adapter,
    )
    try:
        prompt = "The quick brown fox"
        with backend.as_base() as b:
            base_dist = b.next_token_dist(prompt, top_k=32)
        with backend.as_finetuned() as f:
            ft_dist = f.next_token_dist(prompt, top_k=32)

        # Top-k indices may shift under the adapter; take a safe shared
        # subset instead of asserting identical ordering.
        assert not np.array_equal(base_dist.token_ids, ft_dist.token_ids) or not np.allclose(
            base_dist.logprobs, ft_dist.logprobs, atol=1e-5
        ), "adapter toggle did not change next-token distribution"
    finally:
        backend.close()


def test_roundtrip_toggle_restores_base(
    tiny_model_dir: Path, random_adapter: Path
) -> None:
    """as_base → as_finetuned → as_base yields a stable base view."""
    import numpy as np

    backend = HuggingFaceDifferentialBackend(
        base_spec=ModelSpec(base=str(tiny_model_dir), kind="hf", dtype="fp32", device="cpu"),
        adapter_path=random_adapter,
    )
    try:
        prompt = "hello"
        with backend.as_base() as b:
            first = b.next_token_dist(prompt, top_k=16).logprobs
        with backend.as_finetuned() as f:
            f.next_token_dist(prompt, top_k=16)  # toggle
        with backend.as_base() as b:
            second = b.next_token_dist(prompt, top_k=16).logprobs
        np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-6)
    finally:
        backend.close()
