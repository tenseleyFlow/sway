"""S14 F9 prove-the-value: bootstrap CI width shrinks as N grows.

The audit's F9 pitch is "every numeric probe should publish a CI so
downstream claims are honest about sampling noise." The narrow-vs-
wide behavior is the concrete evidence that the CI is informative
rather than a fixed-width decoration.

Test construction: build a dummy backend that produces *per-prompt
varying* divergences by seeding each prompt's ft token distribution
with its hash. Run ``delta_kl`` at N=4 and N=32 on that backend,
assert the N=32 CI is strictly narrower than the N=4 CI — the F9
claim in concrete form.

The stock `DummyDifferentialBackend.as_finetuned()` returns the same
synthesized distribution for every prompt, which produces identical
per-prompt divergences and a zero-width CI at any N. This test's
fixture subclasses the dummy to inject per-prompt variation so the
bootstrap has actual dispersion to measure.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses, _DummyView
from dlm_sway.core.scoring import TokenDist
from dlm_sway.probes.base import RunContext, build_probe


class _VariableFtView(_DummyView):
    """A dummy view whose ``next_token_dist`` varies by prompt.

    Each prompt gets a deterministically-seeded small perturbation of
    the default ft distribution — enough to produce per-prompt JS
    differences in the 0.001–0.05 range, which is where the bootstrap
    CI narrowing is visible.
    """

    def next_token_dist(self, prompt: str, *, top_k: int = 256) -> TokenDist:
        base_dist = super().next_token_dist(prompt, top_k=top_k)
        # Use a stable hash (hashlib) instead of Python's built-in
        # ``hash()``, which salts per-process via PYTHONHASHSEED and
        # would make per-prompt dispersion vary across pytest runs.
        import hashlib

        seed = int(hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 0.5, size=base_dist.logprobs.shape).astype(np.float32)
        perturbed = base_dist.logprobs + noise
        # Renormalize (within the top-k slice).
        max_lp = perturbed.max()
        probs = np.exp(perturbed - max_lp)
        probs /= probs.sum()
        return TokenDist(
            token_ids=base_dist.token_ids,
            logprobs=np.log(probs).astype(np.float32),
            vocab_size=base_dist.vocab_size,
            tail_logprob=base_dist.tail_logprob,
        )


class _VariableFtBackend(DummyDifferentialBackend):
    """Dummy backend whose ft view perturbs per-prompt."""

    @contextmanager
    def as_finetuned(self) -> Iterator[_DummyView]:
        self._enter("ft")
        try:
            view = _VariableFtView("ft", self._ft_r, inst=self._inst)
            yield view
        finally:
            self._exit()


def _run_delta_kl(n_prompts: int) -> tuple[float, float, float]:
    """Run delta_kl with ``n_prompts`` synthesized prompts. Returns
    ``(raw, ci_lo, ci_hi)``.
    """
    backend = _VariableFtBackend(base=DummyResponses(), ft=DummyResponses())
    prompts = [f"prompt-{i:03d}" for i in range(n_prompts)]
    probe, spec = build_probe({"name": f"dk_{n_prompts}", "kind": "delta_kl", "prompts": prompts})
    ctx = RunContext(backend=backend)
    result = probe.run(spec, ctx)
    assert result.raw is not None, "dummy backend delta_kl should produce a raw value"
    assert result.ci_95 is not None, "bootstrap_ci should land on delta_kl output"
    lo, hi = result.ci_95
    return result.raw, lo, hi


def test_ci_width_shrinks_with_more_prompts() -> None:
    """The F9 claim: `delta_kl = 0.05 [0.01, 0.11]` at N=4 narrows to
    something tighter at N=32."""
    raw_4, lo_4, hi_4 = _run_delta_kl(n_prompts=4)
    raw_32, lo_32, hi_32 = _run_delta_kl(n_prompts=32)

    width_4 = hi_4 - lo_4
    width_32 = hi_32 - lo_32

    # Both raws are positive divergences, live in the same order of
    # magnitude, and bracket their own raw value.
    assert raw_4 > 0
    assert raw_32 > 0
    assert lo_4 <= raw_4 <= hi_4
    assert lo_32 <= raw_32 <= hi_32

    # The N=32 CI is strictly tighter than N=4. Theory predicts the
    # CI half-width scales as 1/sqrt(N), so N=32 should be roughly
    # sqrt(8) ≈ 2.8× narrower than N=4.
    assert width_32 < width_4, (
        f"expected width_32 < width_4; got {width_32:.4f} >= {width_4:.4f} "
        f"(CIs: N=4 {[lo_4, hi_4]}, N=32 {[lo_32, hi_32]})"
    )
    # Loose additional check: the narrowing factor is meaningfully
    # bigger than 1.0 — the CI isn't just slightly tighter from
    # RNG noise.
    assert width_4 / max(width_32, 1e-9) > 1.5, (
        f"expected N=4 width at least 1.5× N=32 width; got ratio "
        f"{width_4 / max(width_32, 1e-9):.2f}"
    )
    # Sanity on magnitudes — widths are positive and finite.
    for w in (width_4, width_32):
        assert w > 0
        assert math.isfinite(w)
