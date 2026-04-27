"""Integration test: tool_use_fidelity end-to-end on a real tiny adapter.

Builds a tiny random LoRA on SmolLM2-135M-Instruct and runs the probe
against a single tool-use case. The intent isn't to assert that the
135M base produces useful tool calls — it almost certainly won't —
but to exercise the full code path on a real backend so a regression
in the JSON-extraction / schema-check / hallucination plumbing
surfaces in slow CI rather than only at user-fix time.

Marked ``slow + online``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.online]


def _build_random_lora_adapter(base_dir: Path, out_dir: Path) -> None:
    """A trivially-random LoRA over q_proj/v_proj — same shape used by
    the other slow-lane backend tests."""
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
                # Tiny perturbation — enough that base.generate ≠ ft.generate
                # but small enough to keep generations finite + sensible.
                param.copy_(torch.randn_like(param) * 0.01)
    peft_model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))


@pytest.fixture(scope="module")
def random_adapter(tiny_model_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("tool-use-fidelity-adapter")
    _build_random_lora_adapter(tiny_model_dir, out)
    return out


def test_probe_runs_end_to_end_on_real_adapter(tiny_model_dir: Path, random_adapter: Path) -> None:
    """Smoke: HF backend + real adapter + probe execution returns a finalized
    result with all evidence keys populated. SmolLM2-135M can't reliably
    emit OpenAI-shape calls, so the probe will likely FAIL the validity
    floor — what we assert here is that it produced a structured verdict
    + finite metrics, not a particular pass/fail outcome."""
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
                "name": "tuf_smoke",
                "kind": "tool_use_fidelity",
                "cases": [
                    {
                        "prompt": (
                            "You are a tool-using assistant. The user asks: "
                            "search the web for cats. Reply with ONLY a JSON "
                            'object of the form {"name": ..., "arguments": {...}}.'
                        ),
                        "tool_spec": {
                            "name": "search_web",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "max_results": {"type": "integer"},
                                },
                                "required": ["query"],
                            },
                        },
                        "gold_tool_name": "search_web",
                        "max_new_tokens": 64,
                    }
                ],
                "allowed_tools": ["search_web"],
            }
        )
        ctx = RunContext(backend=backend, seed=0)
        result = probe.run(spec, ctx)
    finally:
        backend.close()

    # End-to-end shape: a verdict came back, evidence carries every
    # documented key, and the rates are in [0, 1].
    assert result.verdict in {Verdict.PASS, Verdict.FAIL, Verdict.WARN}, result.message
    assert result.evidence["num_cases"] == 1
    for key in (
        "json_valid_rate_base",
        "json_valid_rate_ft",
        "validity_delta",
        "mean_arg_disagreement",
        "hallucination_rate",
    ):
        assert key in result.evidence, f"missing evidence key {key}"
    assert 0.0 <= result.evidence["json_valid_rate_ft"] <= 1.0
    assert 0.0 <= result.evidence["json_valid_rate_base"] <= 1.0
    assert -1.0 <= result.evidence["validity_delta"] <= 1.0
    assert 0.0 <= result.evidence["hallucination_rate"] <= 1.0
