"""S23 follow-up — HF-backend batched next_token_dist end-to-end.

Proves the S23 batched-forward path works against a real
HuggingFace + PEFT stack (not just the dummy backend's loop
fallback). The unit tests in ``tests/unit/test_batched_backend_s23``
pin the Protocol and probe-level contracts on the dummy backend;
this one rides the same SmolLM2-135M fixture every other slow+online
test uses and exercises the left-padded ``model.forward`` batched
code path in ``_HFView.next_token_dist_batch``.

What this test locks down:

1. Batched output is numerically equivalent to the serial per-prompt
   output on the same prompts (within fp32 batch-reorder tolerance).
2. The instrumentation counters (``batches_sent``, ``batched_prompts``,
   ``max_batch_size``) reflect at least one real batched forward.
3. The cache short-circuits per-prompt when the same prompt re-enters
   a batch, so the batched counter doesn't inflate on repeat runs.

What this test *doesn't* do (deferred):

- The fortran-spec wall-time benchmark (≤ 60s vs 155s baseline) —
  needs a 1.5B adapter and real GPU; this fixture is 135M on CPU.
- Real MLX batched forward — MLX backend still loops; real
  ``mx.array`` padded forward is a separate follow-up.

Marked ``slow + online``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dlm_sway.backends.hf import HuggingFaceDifferentialBackend
from dlm_sway.core.model import ModelSpec

pytestmark = [pytest.mark.slow, pytest.mark.online]


# Same deterministic-LoRA build the other integration tests use.
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
def batched_adapter(tiny_model_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    adapter_dir = tmp_path_factory.mktemp("batched-s23-adapter")
    _build_random_lora_adapter(tiny_model_dir, adapter_dir)
    return adapter_dir


@pytest.fixture(scope="module")
def hf_backend(tiny_model_dir: Path, batched_adapter: Path) -> HuggingFaceDifferentialBackend:
    backend = HuggingFaceDifferentialBackend(
        base_spec=ModelSpec(base=str(tiny_model_dir), kind="hf", dtype="fp32", device="cpu"),
        adapter_path=batched_adapter,
    )
    yield backend
    backend.close()


# Varied-length prompts so the left-padding path genuinely matters
# (a single length would let a misimplementation slip through).
_PROMPTS = [
    "The capital of France is",
    "Two plus two equals",
    "The quick brown fox jumps over the",
    "Paris",
]


def test_batched_output_matches_serial_on_real_model(
    hf_backend: HuggingFaceDifferentialBackend,
) -> None:
    """The batched forward's top-k logprobs must match the per-prompt
    serial forward's on the same prompts, within a tight fp32
    reorder tolerance.

    Rationale: left-padded batches reorder the underlying attention
    accumulations vs a single-prompt forward. We accept ~1e-4
    divergence — same bar S18's determinism golden uses on CPU.
    """
    # Fresh views to avoid the cache serving identical results from
    # the first call and hiding any real divergence.
    with hf_backend.as_base() as base_view:
        batched_base = base_view.next_token_dist_batch(_PROMPTS, top_k=32)

    # Clear the cache so the serial calls actually re-forward.
    hf_backend._inst.cache.clear()  # noqa: SLF001

    with hf_backend.as_base() as base_view:
        serial_base = [base_view.next_token_dist(p, top_k=32) for p in _PROMPTS]

    for i, (b, s) in enumerate(zip(batched_base, serial_base, strict=True)):
        # Token-id sets should be identical in the top-k slice
        # (ordering can swap on exact-tie logprobs, compare as sets).
        assert set(b.token_ids.tolist()) == set(s.token_ids.tolist()), (
            f"prompt[{i}]={_PROMPTS[i]!r}: top-k token sets differ "
            f"(batched {b.token_ids.tolist()}, serial {s.token_ids.tolist()})"
        )
        # Top-1 logprob should match within the fp32 reorder tol.
        np.testing.assert_allclose(
            sorted(b.logprobs.tolist(), reverse=True)[:5],
            sorted(s.logprobs.tolist(), reverse=True)[:5],
            atol=1e-4,
            rtol=1e-3,
            err_msg=f"prompt[{i}]={_PROMPTS[i]!r}: top-5 logprobs diverged",
        )


def test_batched_forward_fires_instrumentation(hf_backend: HuggingFaceDifferentialBackend) -> None:
    """A batched call on the HF backend must increment
    ``batches_sent`` + ``batched_prompts`` + ``max_batch_size``. This
    is how the report footer knows to print the ``batches: N (avg=K)``
    segment."""
    hf_backend._inst.cache.clear()  # noqa: SLF001
    stats = hf_backend._inst.stats  # noqa: SLF001
    before = (stats.batches_sent, stats.batched_prompts, stats.max_batch_size)

    with hf_backend.as_base() as base_view:
        out = base_view.next_token_dist_batch(_PROMPTS, top_k=16)

    assert len(out) == len(_PROMPTS)
    after = (stats.batches_sent, stats.batched_prompts, stats.max_batch_size)
    assert after[0] == before[0] + 1, f"expected one new batch, got {after[0] - before[0]}"
    assert after[1] == before[1] + len(_PROMPTS), (
        f"expected +{len(_PROMPTS)} batched prompts, got {after[1] - before[1]}"
    )
    assert after[2] >= len(_PROMPTS)


def test_batched_cache_short_circuits_repeat_prompts(
    hf_backend: HuggingFaceDifferentialBackend,
) -> None:
    """Second batched call with identical prompts hits the cache
    per-prompt. ``batches_sent`` must NOT increment a second time
    because no prompts missed."""
    hf_backend._inst.cache.clear()  # noqa: SLF001
    with hf_backend.as_base() as base_view:
        base_view.next_token_dist_batch(_PROMPTS, top_k=16)
    before_batches = hf_backend._inst.stats.batches_sent  # noqa: SLF001
    before_hits = hf_backend._inst.stats.cache_hits  # noqa: SLF001

    with hf_backend.as_base() as base_view:
        base_view.next_token_dist_batch(_PROMPTS, top_k=16)

    after_batches = hf_backend._inst.stats.batches_sent  # noqa: SLF001
    after_hits = hf_backend._inst.stats.cache_hits  # noqa: SLF001
    # No new batch — everything came from the cache.
    assert after_batches == before_batches, (
        f"second all-cache-hit call spuriously fired a batch ({before_batches} → {after_batches})"
    )
    assert after_hits - before_hits == len(_PROMPTS), (
        f"expected {len(_PROMPTS)} fresh cache hits, got {after_hits - before_hits}"
    )
