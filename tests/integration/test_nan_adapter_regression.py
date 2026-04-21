"""Integration regression for Audit 01's +11639σ bug.

This test pins the S01 invariant end-to-end: a PEFT adapter whose
weights are NaN on disk must produce ``Verdict.ERROR`` from the suite,
not a PASS verdict at a mathematically-impossible z-score.

We build a real LoRA adapter on the tiny-model fixture, then poison
every ``lora_A`` / ``lora_B`` safetensors shard with NaN before
constructing :class:`HuggingFaceDifferentialBackend` and running a
real ``sway run`` against it. The full chain — preflight check,
``_divergence`` guards, ``safe_finalize`` — is exercised.

Marked ``slow+online`` so the default fast test run skips it; the
audit-response CI lane runs ``pytest -m slow`` to execute it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlm_sway.backends.hf import HuggingFaceDifferentialBackend
from dlm_sway.core.model import ModelSpec
from dlm_sway.core.result import Verdict
from dlm_sway.suite.runner import run as run_suite
from dlm_sway.suite.spec import SwaySpec

pytestmark = [pytest.mark.slow, pytest.mark.online]


def _build_nan_lora_adapter(base_dir: Path, out_dir: Path) -> None:
    """Build a PEFT adapter then overwrite every lora_A/lora_B with NaN.

    Reproduces the exact pathology the audit observed: structurally
    valid adapter config + tokenizer + safetensors shard layout, but
    the numeric tensors are populated with NaN. This is what
    ``dlm train`` used to produce on MPS with tiny datasets (fixed
    upstream but still the canonical "broken adapter" regression case).
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

    # Poison: fill every LoRA parameter with NaN.
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.fill_(float("nan"))

    peft_model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))


@pytest.fixture(scope="module")
def nan_adapter(tiny_model_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    adapter_dir = tmp_path_factory.mktemp("nan-adapter")
    _build_nan_lora_adapter(tiny_model_dir, adapter_dir)
    return adapter_dir


def test_nan_adapter_on_disk_is_reproducibly_nan(nan_adapter: Path) -> None:
    """Sanity: the poisoned adapter's persisted weights are actually NaN."""
    import torch
    from safetensors.torch import load_file

    weights = load_file(str(nan_adapter / "adapter_model.safetensors"))
    assert weights, "no tensors in adapter_model.safetensors"
    at_least_one_nan = False
    for name, t in weights.items():
        if "lora_A" in name or "lora_B" in name:
            assert torch.isnan(t).all(), f"{name} is not fully NaN — regression fixture broken"
            at_least_one_nan = True
    assert at_least_one_nan, "no lora_A/lora_B tensors found — adapter structure unexpected"


def test_hf_backend_preflight_rejects_nan_adapter(
    tiny_model_dir: Path, nan_adapter: Path
) -> None:
    """The HF backend's preflight catches the NaN adapter at construction time.

    Before S01 this ran to completion and produced JS = 13.247 nats.
    Now: preflight returns ``(False, ...)`` and the suite aborts.
    """
    backend = HuggingFaceDifferentialBackend(
        base_spec=ModelSpec(base=str(tiny_model_dir), kind="hf", dtype="fp32", device="cpu"),
        adapter_path=nan_adapter,
    )
    try:
        ok, reason = backend.preflight_finite_check()
        assert ok is False
        assert "non-finite" in reason.lower() or "nan" in reason.lower()
    finally:
        backend.close()


def test_full_suite_run_emits_error_not_pass_on_nan_adapter(
    tiny_model_dir: Path, nan_adapter: Path
) -> None:
    """End-to-end: ``sway run`` against a NaN adapter returns ERROR banner.

    The regression this pins is the +11639σ headline the audit caught.
    """
    spec = SwaySpec.model_validate(
        {
            "version": 1,
            "models": {
                "base": {
                    "kind": "hf",
                    "base": str(tiny_model_dir),
                    "dtype": "fp32",
                    "device": "cpu",
                },
                "ft": {
                    "kind": "hf",
                    "base": str(tiny_model_dir),
                    "dtype": "fp32",
                    "device": "cpu",
                    "adapter": str(nan_adapter),
                },
            },
            "suite": [
                {"name": "doc_kl", "kind": "delta_kl", "prompts": ["hello world"]},
            ],
        }
    )
    backend = HuggingFaceDifferentialBackend(
        base_spec=spec.models.ft,
        adapter_path=nan_adapter,
    )
    try:
        result = run_suite(spec, backend, spec_path="<nan-regression>")
    finally:
        backend.close()

    # Preflight should short-circuit: exactly one synthetic ERROR probe;
    # the configured delta_kl probe never runs.
    assert len(result.probes) == 1
    preflight = result.probes[0]
    assert preflight.kind == "preflight"
    assert preflight.verdict == Verdict.ERROR
    assert "preflight failed" in preflight.message.lower()
    # Absolutely no PASS verdict anywhere in the suite result.
    assert not any(r.verdict == Verdict.PASS for r in result.probes)
    # Sanity: the delta_kl probe configured in the spec did not run.
    assert not any(r.kind == "delta_kl" for r in result.probes)
