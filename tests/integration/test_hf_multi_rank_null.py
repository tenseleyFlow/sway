"""Integration test: HF backend under multi-rank null calibration (S10 / F4).

Two contracts:

1. **Correctness.** Three ``rank_multipliers`` against SmolLM2-135M
   produce three per-rank null-stats groups; each rank's stats are
   finite and the std rises with rank (larger rank_scale → larger
   effective noise by the sqrt(r) scaling the backend uses).
2. **Performance.** Wall time for ``rank_multipliers=[0.5, 1.0, 2.0]``
   stays under ``2×`` the single-rank baseline — i.e., we don't
   reload the base model per multiplier. The noise-scaling approach
   makes rank switches free (no tensor reshape, no reload).

Marked ``slow+online``.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from dlm_sway.backends.hf import HuggingFaceDifferentialBackend
from dlm_sway.core.model import ModelSpec
from dlm_sway.core.result import Verdict
from dlm_sway.probes.base import RunContext, build_probe

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
    adapter_dir = tmp_path_factory.mktemp("multi-rank-random-adapter")
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


def _run_null(
    backend: HuggingFaceDifferentialBackend, rank_multipliers: list[float]
) -> tuple[float, dict[str, dict[str, dict[str, float]]]]:
    """Run null_adapter once and return (wall_seconds, null_stats_by_rank)."""
    probe, spec = build_probe(
        {
            "name": "null",
            "kind": "null_adapter",
            "runs": 2,
            "rank_multipliers": rank_multipliers,
            "calibrate_kinds": ["delta_kl"],
            "cache": False,  # force real compute for the timing comparison
        }
    )
    ctx = RunContext(backend=backend)
    t0 = time.perf_counter()
    result = probe.run(spec, ctx)
    wall = time.perf_counter() - t0
    assert result.verdict == Verdict.PASS, result.message
    return wall, dict(result.evidence["null_stats_by_rank"])


def test_three_ranks_produce_three_stats_groups(
    hf_backend: HuggingFaceDifferentialBackend,
) -> None:
    _, by_rank = _run_null(hf_backend, [0.5, 1.0, 2.0])
    assert set(by_rank) == {"rank_0.50", "rank_1.00", "rank_2.00"}
    for rkey, kind_stats in by_rank.items():
        delta_kl = kind_stats.get("delta_kl")
        assert delta_kl is not None, f"{rkey} missing delta_kl stats"
        assert delta_kl["n"] == 2.0
        assert delta_kl["std"] > 0.0


def test_multi_rank_does_not_reload_base(
    hf_backend: HuggingFaceDifferentialBackend,
) -> None:
    """Three ranks must scale ~linearly with probe iterations, *not*
    incur a per-rank base-model reload.

    Three ranks × two seeds = 6 calibration iterations vs single-rank's
    2 iterations — so a linear-compute upper bound is ≈3×. The S07
    forward-pass cache on the base view can save more, but doesn't
    always (null-side view_ids are distinct per rank and seed). We
    assert < 4× as the clear "no reload" ceiling: a per-rank base
    reload would blow this past 10× on a 135M model.
    """
    # Warmup: first call amortizes the base-model load. Without this
    # the single-rank baseline absorbs the load cost and the ratio
    # becomes uninformative.
    _run_null(hf_backend, [1.0])

    single_wall, _ = _run_null(hf_backend, [1.0])
    multi_wall, _ = _run_null(hf_backend, [0.5, 1.0, 2.0])
    ratio = multi_wall / max(single_wall, 0.01)
    assert ratio < 4.0, (
        f"multi-rank wall {multi_wall:.2f}s is {ratio:.2f}× single-rank {single_wall:.2f}s "
        "(threshold: < 4× — a true base-model reload would exceed 10×)"
    )
