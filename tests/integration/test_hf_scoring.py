"""Integration test: HF backend scoring methods on a real tiny model.

Covers ``logprob_of`` / ``rolling_logprob`` / ``next_token_dist`` for
both base and ft views — the surface area sway probes hammer hardest
and the area Audit 01 flagged as 21% covered (C2).

The zero-token-completion path of ``logprob_of`` (which raises
``ProbeError``) is exercised here too, since the alternative is the
full CLI integration test catching it for the wrong reason.

Marked ``slow+online``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from dlm_sway.backends.hf import HuggingFaceDifferentialBackend
from dlm_sway.core.errors import ProbeError
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
    adapter_dir = tmp_path_factory.mktemp("scoring-random-adapter")
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


_PROMPTS_AND_COMPLETIONS = [
    ("The capital of France is", " Paris"),
    ("Two plus two equals", " four"),
    ("The quick brown fox jumps over the", " lazy dog"),
]


class TestLogprobOf:
    @pytest.mark.parametrize(("prompt", "completion"), _PROMPTS_AND_COMPLETIONS)
    def test_finite_negative_for_real_completions(
        self,
        hf_backend: HuggingFaceDifferentialBackend,
        prompt: str,
        completion: str,
    ) -> None:
        with hf_backend.as_base() as b:
            lp_base = b.logprob_of(prompt, completion)
        with hf_backend.as_finetuned() as f:
            lp_ft = f.logprob_of(prompt, completion)
        assert math.isfinite(lp_base)
        assert lp_base < 0.0
        assert math.isfinite(lp_ft)
        assert lp_ft < 0.0

    def test_zero_token_completion_raises_probe_error(
        self, hf_backend: HuggingFaceDifferentialBackend
    ) -> None:
        """Empty completion tokenizes to zero new tokens — the entry
        point must reject it loudly so a probe can route to ERROR."""
        with hf_backend.as_base() as b:
            with pytest.raises(ProbeError, match="completion tokenized to zero"):
                b.logprob_of("hello", "")

    def test_longer_completion_is_more_negative(
        self, hf_backend: HuggingFaceDifferentialBackend
    ) -> None:
        """Sanity: extending a completion can only add negative logprob."""
        with hf_backend.as_base() as b:
            short = b.logprob_of("the prefix is", " short")
            longer = b.logprob_of("the prefix is", " short and gets longer here")
        assert longer < short, f"longer={longer}, short={short}"


class TestRollingLogprob:
    def test_returns_per_position_logprobs_and_finite_summary(
        self, hf_backend: HuggingFaceDifferentialBackend
    ) -> None:
        with hf_backend.as_base() as b:
            r = b.rolling_logprob("Hello world. This is a sentence.")
        assert r.num_tokens >= 2
        assert r.logprobs.size == r.num_tokens - 1
        assert math.isfinite(r.total_logprob)
        assert math.isfinite(r.mean_logprob)
        assert math.isfinite(r.perplexity)
        assert r.perplexity > 1.0  # any text past one token has PPL > 1

    def test_short_text_under_two_tokens_returns_empty(
        self, hf_backend: HuggingFaceDifferentialBackend
    ) -> None:
        """Single-token text has no per-position predictions to gather."""
        with hf_backend.as_base() as b:
            r = b.rolling_logprob("a")
        assert r.logprobs.size == 0
        assert r.total_logprob == 0.0


class TestGenerate:
    def test_greedy_generation_returns_string(
        self, hf_backend: HuggingFaceDifferentialBackend
    ) -> None:
        with hf_backend.as_base() as b:
            out = b.generate("Hello", max_new_tokens=8, seed=0)
        assert isinstance(out, str)
        assert len(out) > 0

    def test_sampled_generation_obeys_seed(
        self, hf_backend: HuggingFaceDifferentialBackend
    ) -> None:
        """``temperature > 0`` engages the sampling path (do_sample=True)."""
        with hf_backend.as_base() as b:
            a = b.generate("The future of AI is", max_new_tokens=8, temperature=0.7, seed=7)
            b1 = b.generate("The future of AI is", max_new_tokens=8, temperature=0.7, seed=7)
        assert a == b1, f"sampled generation not deterministic at seed=7: {a!r} vs {b1!r}"


class TestNextTokenDist:
    def test_top_k_dist_finite_and_sorted(self, hf_backend: HuggingFaceDifferentialBackend) -> None:
        with hf_backend.as_base() as b:
            d = b.next_token_dist("The capital of France is", top_k=64)
        assert d.token_ids.shape == (64,)
        assert d.logprobs.shape == (64,)
        assert np.all(np.isfinite(d.logprobs))
        # Top-k must arrive in descending probability order.
        assert np.all(np.diff(d.logprobs) <= 1e-7)
        assert d.vocab_size > 64
        assert math.isfinite(d.tail_logprob) or d.tail_logprob == 0.0

    def test_dist_changes_under_adapter(self, hf_backend: HuggingFaceDifferentialBackend) -> None:
        prompt = "the adapter influences"
        with hf_backend.as_base() as b:
            base_dist = b.next_token_dist(prompt, top_k=32)
        with hf_backend.as_finetuned() as f:
            ft_dist = f.next_token_dist(prompt, top_k=32)
        # Either the top-32 token IDs reordered, or at least one logprob
        # moved by more than fp32 noise.
        same_ids = np.array_equal(base_dist.token_ids, ft_dist.token_ids)
        if same_ids:
            assert not np.allclose(base_dist.logprobs, ft_dist.logprobs, atol=1e-5)
