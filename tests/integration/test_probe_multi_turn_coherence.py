"""Integration test: multi_turn_coherence_decay end-to-end on a tiny LoRA.

Builds a tiny random LoRA on SmolLM2-135M-Instruct (which has a real
chat_template) and runs the probe through 4 turns of synthetic
dialogue. The intent isn't to assert specific KL values — they
depend on the random adapter — but to exercise the full code path
on a real backend so a regression in the chat-template wiring,
turn-loop, or curve-fit plumbing surfaces in slow CI.

Marked ``slow + online``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.online]


def _build_random_lora_adapter(base_dir: Path, out_dir: Path) -> None:
    """Same shape as the other slow-lane backend tests."""
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
                # Tiny perturbation — base != ft, but generations stay sane
                # enough to thread through 4 dialogue turns without
                # collapsing to junk.
                param.copy_(torch.randn_like(param) * 0.02)
    peft_model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))


@pytest.fixture(scope="module")
def random_adapter(tiny_model_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("multi-turn-coherence-adapter")
    _build_random_lora_adapter(tiny_model_dir, out)
    return out


def test_probe_runs_end_to_end_on_real_adapter(tiny_model_dir: Path, random_adapter: Path) -> None:
    """Smoke: HF backend + chat_template-equipped tokenizer + real
    multi-turn dialogue produces a finalized result with the
    documented evidence keys + finite per-turn KLs."""
    from dlm_sway.backends.hf import HuggingFaceDifferentialBackend
    from dlm_sway.core.model import ModelSpec
    from dlm_sway.core.result import Verdict
    from dlm_sway.probes.base import RunContext, build_probe

    backend = HuggingFaceDifferentialBackend(
        base_spec=ModelSpec(base=str(tiny_model_dir), kind="hf", dtype="fp32", device="cpu"),
        adapter_path=random_adapter,
    )
    try:
        probe, spec = build_probe(
            {
                "name": "mtc_smoke",
                "kind": "multi_turn_coherence_decay",
                "prompts": [
                    "What's the difference between TCP and UDP?",
                    "Explain how a neural network learns.",
                ],
                "max_turns": 3,
                "max_new_tokens": 32,  # keep CPU runtime under control
            }
        )
        ctx = RunContext(backend=backend, seed=0, top_k=64)
        result = probe.run(spec, ctx)
    finally:
        backend.close()

    # Shape: any verdict that isn't ERROR is fine. We don't pin
    # PASS/FAIL because the random adapter's actual decay shape isn't
    # under our control.
    assert result.verdict in {
        Verdict.PASS,
        Verdict.FAIL,
        Verdict.WARN,
    }, result.message
    assert result.evidence["max_turns"] == 3
    assert result.evidence["num_prompts"] == 2
    per_turn = result.evidence["per_turn_kls"]
    assert len(per_turn) == 2  # max_turns - 1
    for kl in per_turn:
        assert isinstance(kl, float)
        assert kl >= 0.0  # KL is non-negative
    assert result.evidence["fit_status"] in {"ok", "stable", "non_monotonic", "degenerate"}
    sparkline = result.evidence["sparkline"]
    assert isinstance(sparkline, str)
    assert len(sparkline) == 2
