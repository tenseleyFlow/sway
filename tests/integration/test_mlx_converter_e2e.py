"""S24 — end-to-end PEFT → MLX adapter conversion (darwin-arm64-only).

Closes the F01 audit gap: ``dlm train`` writes a PEFT-shaped adapter,
``MLXDifferentialBackend`` is pointed at it, the backend auto-converts
into the user's cache, ``mlx_lm.load`` consumes the result, scoring
returns finite logprobs.

This is the **prove-the-value** test the sprint file calls out — every
other layer of testing (synthetic-input unit tests, CLI smoke tests)
is upstream of this. If this passes locally on darwin-arm64, the
headline ``.dlm → MLX`` flow works.

Skips cleanly on:
- non-darwin (mlx is Apple Silicon only)
- non-arm64
- ``mlx_lm`` not installed (the ``[mlx]`` extra is optional)
- ``peft`` / ``transformers`` not installed (the ``[hf]`` extra needed
  to *build* the source PEFT adapter)
"""

from __future__ import annotations

import math
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.online]


# Default to the unquantized MLX repo because the 4-bit variant has
# slipped into a gated/auth state on HF Hub. Either repo's adapter
# slot works for the converter — the test only cares that mlx-lm
# loads our converted ``adapters.safetensors``.
_MODEL_ID = "mlx-community/SmolLM2-135M-Instruct"


def _platform_supports_mlx() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _build_random_peft_lora(base_dir: Path, out_dir: Path) -> None:
    """Same deterministic LoRA the HF integration tests use, shipped
    here because we don't want to import from another test file."""
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
def peft_adapter(tiny_model_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not _platform_supports_mlx():
        pytest.skip("MLX requires darwin-arm64")
    pytest.importorskip("peft", reason="needs the [hf] extra to build a PEFT adapter")
    out = tmp_path_factory.mktemp("peft-for-mlx-convert")
    _build_random_peft_lora(tiny_model_dir, out)
    return out


@pytest.fixture(scope="module")
def mlx_backend(peft_adapter: Path, tmp_path_factory: pytest.TempPathFactory):
    """Point the MLX backend at a PEFT-shaped adapter dir; the backend
    auto-converts into a tmp cache (XDG_CACHE_HOME redirected so we
    don't pollute the user's real cache)."""
    pytest.importorskip("mlx_lm", reason="install the [mlx] extra to run MLX tests")

    # Redirect the cache so this test doesn't write to the user's
    # ~/.cache/dlm-sway/. Each fixture invocation gets a fresh dir.
    import os

    cache_root = tmp_path_factory.mktemp("mlx-convert-cache")
    prev = os.environ.get("XDG_CACHE_HOME")
    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    try:
        from dlm_sway.backends.mlx import MLXDifferentialBackend
        from dlm_sway.core.model import ModelSpec

        backend = MLXDifferentialBackend(
            base_spec=ModelSpec(base=_MODEL_ID, kind="mlx"),
            adapter_path=peft_adapter,
        )
        yield backend, cache_root
        backend.close()
    finally:
        if prev is None:
            os.environ.pop("XDG_CACHE_HOME", None)
        else:
            os.environ["XDG_CACHE_HOME"] = prev


def test_auto_conversion_writes_to_xdg_cache(mlx_backend) -> None:
    """The backend's __init__ must have populated the cache dir with
    an MLX-format adapter — proves the auto-convert path fired."""
    _backend, cache_root = mlx_backend
    converted = list((cache_root / "dlm-sway" / "mlx-converted").glob("*"))
    assert len(converted) == 1, f"expected exactly one cached MLX adapter dir, got {converted}"
    cache_dir = converted[0]
    assert (cache_dir / "adapters.safetensors").exists()
    assert (cache_dir / "adapter_config.json").exists()


def test_next_token_dist_returns_finite_topk_via_converted_adapter(mlx_backend) -> None:
    """The converted adapter, loaded via mlx_lm + scored via the MLX
    backend, must produce finite, well-ordered top-k logprobs."""
    backend, _ = mlx_backend
    with backend.as_finetuned() as ft:
        d = ft.next_token_dist("The capital of France is", top_k=32)
    assert d.token_ids.shape == (32,)
    assert d.logprobs.shape == (32,)
    assert np.all(np.isfinite(d.logprobs))
    assert np.all(np.diff(d.logprobs) <= 1e-7)  # descending


def test_logprob_of_finite_via_converted_adapter(mlx_backend) -> None:
    backend, _ = mlx_backend
    with backend.as_finetuned() as ft:
        lp = ft.logprob_of("The capital of France is", " Paris")
    assert math.isfinite(lp)
    assert lp < 0.0


def test_repeat_load_skips_reconvert(
    peft_adapter: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Second backend instance against the same PEFT adapter must
    short-circuit on the cache and NOT rewrite the converted file."""
    pytest.importorskip("mlx_lm", reason="install the [mlx] extra to run MLX tests")

    import os

    cache_root = tmp_path_factory.mktemp("mlx-convert-cache-2")
    prev = os.environ.get("XDG_CACHE_HOME")
    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    try:
        from dlm_sway.backends.mlx import MLXDifferentialBackend
        from dlm_sway.core.model import ModelSpec

        b1 = MLXDifferentialBackend(
            base_spec=ModelSpec(base=_MODEL_ID, kind="mlx"),
            adapter_path=peft_adapter,
        )
        cache_dir = next((cache_root / "dlm-sway" / "mlx-converted").glob("*"))
        first_mtime = (cache_dir / "adapters.safetensors").stat().st_mtime_ns
        b1.close()

        b2 = MLXDifferentialBackend(
            base_spec=ModelSpec(base=_MODEL_ID, kind="mlx"),
            adapter_path=peft_adapter,
        )
        second_mtime = (cache_dir / "adapters.safetensors").stat().st_mtime_ns
        b2.close()

        assert second_mtime == first_mtime, (
            "second backend init re-wrote the cached MLX adapter — cache short-circuit is broken"
        )
    finally:
        if prev is None:
            os.environ.pop("XDG_CACHE_HOME", None)
        else:
            os.environ["XDG_CACHE_HOME"] = prev
