"""Integration test: ``cluster_kl`` end-to-end on a real tiny model.

Mirrors ``test_external_perplexity_e2e`` — the sprint file for S16
explicitly lists this as a DoD item but the fixture was never shipped.

The test:

1. Builds a small random LoRA on SmolLM2-135M (same template as
   ``test_external_perplexity_e2e``).
2. Runs ``cluster_kl`` with a 16-prompt two-topic set (animals +
   programming) — split the ft signal across topics so the specificity
   ratio has a chance to be meaningfully non-0.5.
3. Asserts the probe terminates in a non-ERROR verdict, the specificity
   is finite and in ``[0, 1]``, and when preceded by ``null_adapter`` in
   a suite the z-score field is populated.

Needs the ``[semsim]`` extra at runtime (sentence-transformers +
scikit-learn). We assume integration runners install those; skip
gracefully when they don't.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import pytest

from dlm_sway.backends.hf import HuggingFaceDifferentialBackend
from dlm_sway.core.model import ModelSpec
from dlm_sway.core.result import Verdict
from dlm_sway.probes.base import RunContext, build_probe
from dlm_sway.suite.runner import run as run_suite
from dlm_sway.suite.spec import SwaySpec

pytestmark = [pytest.mark.slow, pytest.mark.online]


# 16 prompts split 8/8 across two obvious topics.
_PROMPTS = [
    # Animals (topic A)
    "The cat chased the mouse around the house.",
    "Dogs wag their tails when they are happy.",
    "Elephants never forget a face they have seen.",
    "Lions hunt in packs called prides.",
    "Horses gallop across open fields.",
    "Sharks have rows of sharp teeth.",
    "Bees pollinate flowers as they gather nectar.",
    "Owls hunt small rodents at night.",
    # Programming (topic B)
    "Write a Python decorator that logs every call.",
    "Implement binary search in Rust.",
    "Debug a segmentation fault in C++ pointer arithmetic.",
    "Explain ownership semantics in Rust.",
    "Refactor this JavaScript callback hell into promises.",
    "Optimize the SQL query by adding an index.",
    "Profile the memory usage of a Go program.",
    "Write unit tests for a REST API endpoint.",
]


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
    adapter_dir = tmp_path_factory.mktemp("cluster-kl-random-adapter")
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
    pytest.importorskip("sklearn")
    pytest.importorskip("sentence_transformers")

    probe, spec = build_probe(
        {
            "name": "ck",
            "kind": "cluster_kl",
            "prompts": _PROMPTS,
            "num_clusters": 2,
            "min_prompts": 16,
        }
    )
    ctx = RunContext(backend=hf_backend)
    result = probe.run(spec, ctx)

    assert result.verdict != Verdict.ERROR, f"probe errored: {result.message}"
    # Under a small random LoRA we don't know the specificity sign;
    # just pin that it's finite and in [0, 1].
    assert result.raw is not None
    assert math.isfinite(result.raw)
    assert 0.0 <= result.raw <= 1.0
    assert result.evidence["num_clusters"] == 2
    assert result.evidence["num_prompts"] == 16
    per_cluster = result.evidence["per_cluster_mean_kl"]
    assert len(per_cluster) == 2


def test_null_calibration_lights_up_zscore(
    hf_backend: HuggingFaceDifferentialBackend,
) -> None:
    """null_adapter → cluster_kl produces a z_score end-to-end."""
    pytest.importorskip("sklearn")
    pytest.importorskip("sentence_transformers")

    raw_spec = SwaySpec.model_validate(
        {
            "version": 1,
            "models": {
                "base": {"base": "placeholder"},
                "ft": {"base": "placeholder", "adapter": "/tmp/placeholder"},
            },
            "suite": [
                {"name": "null", "kind": "null_adapter", "runs": 2, "cache": False},
                {
                    "name": "ck",
                    "kind": "cluster_kl",
                    "prompts": _PROMPTS,
                    "num_clusters": 2,
                    "min_prompts": 16,
                    "assert_z_gte": -100.0,  # permissive — just want z populated
                },
            ],
        }
    )
    result = run_suite(raw_spec, hf_backend)
    assert len(result.probes) == 2
    null_result = result.probes[0]
    ck_result = result.probes[1]
    assert null_result.verdict == Verdict.PASS
    assert ck_result.verdict != Verdict.ERROR
    assert ck_result.z_score is not None, (
        f"cluster_kl should have z-scored against null baseline; "
        f"evidence={ck_result.evidence}, message={ck_result.message}"
    )
    assert math.isfinite(ck_result.z_score)
