"""Integration test: ``HF.as_null_adapter`` determinism + restoration.

Two contracts the calibration matrix depends on:

1. **Same seed → same null weights.** ``as_null_adapter(seed=0)`` called
   twice in a row must produce bit-identical lora_A / lora_B tensors
   inside the context. If a future PEFT release randomized something
   we missed (e.g. dropout sampling), the cached null stats would be
   silently inconsistent across runs.
2. **Original adapter restored on exit.** After the context manager
   returns, every ``lora_A`` / ``lora_B`` parameter must equal its
   pre-context value. Otherwise the next probe runs against a
   randomly-poisoned ft view.

Marked ``slow+online`` to share the tiny-model fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dlm_sway.backends.hf import HuggingFaceDifferentialBackend
from dlm_sway.core.model import ModelSpec

pytestmark = [pytest.mark.slow, pytest.mark.online]


def _build_random_lora_adapter(base_dir: Path, out_dir: Path) -> None:
    """Same shape as the toggle-test adapter."""
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
    adapter_dir = tmp_path_factory.mktemp("null-random-adapter")
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


def _snapshot_lora_params(backend: HuggingFaceDifferentialBackend) -> dict[str, np.ndarray]:
    """Capture every lora_A / lora_B parameter as a numpy copy."""
    out: dict[str, np.ndarray] = {}
    for pname, param in backend._peft_model.named_parameters():  # type: ignore[attr-defined]
        if any(key in pname for key in ("lora_A", "lora_B")):
            out[pname] = param.detach().cpu().numpy().copy()
    return out


def test_same_seed_produces_identical_null_weights(
    hf_backend: HuggingFaceDifferentialBackend,
) -> None:
    with hf_backend.as_null_adapter(seed=0):
        first = _snapshot_lora_params(hf_backend)
    with hf_backend.as_null_adapter(seed=0):
        second = _snapshot_lora_params(hf_backend)
    assert set(first) == set(second)
    for key in first:
        np.testing.assert_array_equal(
            first[key],
            second[key],
            err_msg=f"as_null_adapter(seed=0) was not deterministic for {key!r}",
        )


def test_different_seeds_produce_different_null_weights(
    hf_backend: HuggingFaceDifferentialBackend,
) -> None:
    with hf_backend.as_null_adapter(seed=0):
        a = _snapshot_lora_params(hf_backend)
    with hf_backend.as_null_adapter(seed=1):
        b = _snapshot_lora_params(hf_backend)
    different = any(not np.array_equal(a[k], b[k]) for k in a)
    assert different, "null adapters at seed=0 and seed=1 produced identical weights"


def test_original_adapter_restored_on_exit(
    hf_backend: HuggingFaceDifferentialBackend,
) -> None:
    before = _snapshot_lora_params(hf_backend)
    with hf_backend.as_null_adapter(seed=42):
        # confirm the inner state is *not* the original
        inner = _snapshot_lora_params(hf_backend)
        assert any(not np.array_equal(before[k], inner[k]) for k in before)
    after = _snapshot_lora_params(hf_backend)
    for key in before:
        np.testing.assert_array_equal(
            before[key],
            after[key],
            err_msg=f"original adapter not restored for {key!r}",
        )


def test_original_adapter_restored_on_exception(
    hf_backend: HuggingFaceDifferentialBackend,
) -> None:
    before = _snapshot_lora_params(hf_backend)
    with pytest.raises(RuntimeError, match="boom"):
        with hf_backend.as_null_adapter(seed=99):
            raise RuntimeError("boom")
    after = _snapshot_lora_params(hf_backend)
    for key in before:
        np.testing.assert_array_equal(
            before[key],
            after[key],
            err_msg=f"original adapter not restored after exception for {key!r}",
        )
