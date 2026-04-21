"""Integration test: ``external_perplexity`` end-to-end on a real tiny model.

Runs the probe against SmolLM2-135M with a small random LoRA so both
sides produce real rolling-logprob values. The test asserts three
contracts:

1. The probe terminates in a non-ERROR verdict (the real backend's
   ``rolling_logprob`` returns finite logprobs on natural English prose).
2. The per-chunk delta array has the requested length and no NaNs.
3. The null-calibration path lights up the ``z_score`` field in a
   two-probe suite (``null_adapter`` first, then ``external_perplexity``).

Marked ``slow+online``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from dlm_sway.backends.hf import HuggingFaceDifferentialBackend
from dlm_sway.core.model import ModelSpec
from dlm_sway.core.result import Verdict
from dlm_sway.probes.base import RunContext, build_probe
from dlm_sway.suite.runner import run as run_suite
from dlm_sway.suite.spec import SwaySpec

pytestmark = [pytest.mark.slow, pytest.mark.online]


def _build_random_lora_adapter(base_dir: Path, out_dir: Path) -> None:
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
    adapter_dir = tmp_path_factory.mktemp("ext-ppl-random-adapter")
    _build_random_lora_adapter(tiny_model_dir, adapter_dir)
    return adapter_dir


@pytest.fixture(scope="module")
def hf_backend(
    tiny_model_dir: Path, random_adapter: Path
) -> Iterator[HuggingFaceDifferentialBackend]:
    backend = HuggingFaceDifferentialBackend(
        base_spec=ModelSpec(base=str(tiny_model_dir), kind="hf", dtype="fp32", device="cpu"),
        adapter_path=random_adapter,
    )
    yield backend
    backend.close()


def test_probe_runs_on_real_backend(hf_backend: HuggingFaceDifferentialBackend) -> None:
    probe, spec = build_probe(
        {
            "name": "ext_ppl",
            "kind": "external_perplexity",
            "max_chunks": 2,
            "chunk_chars": 512,
        }
    )
    ctx = RunContext(backend=hf_backend)
    result = probe.run(spec, ctx)
    assert result.verdict != Verdict.ERROR, f"probe errored: {result.message}"
    assert result.raw is not None
    assert math.isfinite(result.raw)
    per_chunk = result.evidence["per_chunk_delta"]
    assert len(per_chunk) == 2
    assert all(math.isfinite(d) for d in per_chunk)
    assert np.all(np.isfinite(np.asarray(per_chunk, dtype=np.float64)))


def test_null_calibration_lights_up_zscore(hf_backend: HuggingFaceDifferentialBackend) -> None:
    """null_adapter → external_perplexity produces a z_score end-to-end."""
    raw_spec = SwaySpec.model_validate(
        {
            "version": 1,
            "models": {
                "base": {"base": "placeholder"},
                "ft": {"base": "placeholder", "adapter": "/tmp/placeholder"},
            },
            "suite": [
                # Two null seeds keep runtime bounded; std just has to be
                # non-zero for the z-score path to engage.
                {"name": "null", "kind": "null_adapter", "runs": 2, "cache": False},
                {
                    "name": "ext",
                    "kind": "external_perplexity",
                    "max_chunks": 2,
                    "chunk_chars": 512,
                    "assert_z_gte": -100.0,  # permissive — sign/magnitude is adapter-specific
                },
            ],
        }
    )
    result = run_suite(raw_spec, hf_backend)
    assert len(result.probes) == 2
    null_result = result.probes[0]
    ext_result = result.probes[1]
    assert null_result.verdict == Verdict.PASS
    assert ext_result.verdict != Verdict.ERROR
    assert ext_result.z_score is not None, (
        f"external_perplexity should have z-scored against null baseline; "
        f"evidence={ext_result.evidence}, message={ext_result.message}"
    )
    assert math.isfinite(ext_result.z_score)
