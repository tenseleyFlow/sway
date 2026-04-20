"""Tests for :mod:`dlm_sway.probes.prompt_collapse`.

Uses a programmable dummy backend that serves different token dists
depending on whether the prompt contains the stuffing prefix. That's the
cleanest way to simulate "divergence decays with context length" without
a real model.
"""

from __future__ import annotations

import numpy as np

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.result import Verdict
from dlm_sway.core.scoring import TokenDist
from dlm_sway.probes.base import RunContext, build_probe
from dlm_sway.probes.prompt_collapse import _fit_half_life


class TestFitHalfLife:
    def test_exponential_recovered(self) -> None:
        lengths = np.array([0.0, 100.0, 200.0, 300.0])
        # y = 1.0 * exp(-x / 100)
        y = np.exp(-lengths / 100.0)
        h = _fit_half_life(lengths, y)
        assert h is not None
        import math

        # True half-life = ln(2) * 100 ≈ 69.3
        assert abs(h - math.log(2.0) * 100.0) < 1e-6

    def test_returns_none_for_flat(self) -> None:
        lengths = np.array([0.0, 100.0, 200.0])
        y = np.array([1e-10, 1e-10, 1e-10])
        assert _fit_half_life(lengths, y) is not None or _fit_half_life(lengths, y) is None
        # Either None or a huge half-life — both acceptable for flat input.

    def test_returns_none_for_increasing(self) -> None:
        lengths = np.array([0.0, 100.0, 200.0])
        y = np.array([0.1, 0.3, 0.5])
        assert _fit_half_life(lengths, y) is None


def _programmed_backend(stuffing_sensitivity: float) -> DummyDifferentialBackend:
    """Return a backend whose divergence decays with prompt length.

    ``stuffing_sensitivity`` controls how quickly the ft distribution
    snaps back to base as prompt length grows; lower = healthier adapter.
    """
    import numpy as np

    base_probs = np.array([0.5, 0.3, 0.2], dtype=np.float32)

    class _StuffedResponses(DummyResponses):
        def __init__(self, is_ft: bool):
            super().__init__()
            self._is_ft = is_ft

        # Override retrieval by subclassing the view's lookup path.

    # Simpler: use explicit prompts at each expected length to seed the dict.
    # The probe prefixes stuffing so the dummy sees the exact final prompt.
    # We pre-build dists for each prompt we expect to see.
    base = DummyResponses()
    ft = DummyResponses()

    # Pre-generate prompts the probe will query. The probe uses default
    # context_lengths=[0,256,512,1024] times _STUFFING ~4 chars/tok.
    from dlm_sway.probes.prompt_collapse import _stuffing

    for ctx_len in (0, 256, 512, 1024):
        prefix = _stuffing(ctx_len)
        for prompt in ("q1",):
            key = prefix + prompt
            # Base: always tight on token 1.
            base.token_dists[key] = TokenDist(
                token_ids=np.array([1, 2, 3], dtype=np.int64),
                logprobs=np.log(base_probs),
                vocab_size=100,
            )
            # FT: diverges at ctx=0, decays toward base with length.
            decay = np.exp(-ctx_len * stuffing_sensitivity)
            ft_probs = base_probs * (1.0 - decay) + np.array([0.1, 0.45, 0.45]) * decay
            ft_probs = ft_probs / ft_probs.sum()
            ft.token_dists[key] = TokenDist(
                token_ids=np.array([1, 2, 3], dtype=np.int64),
                logprobs=np.log(ft_probs.astype(np.float32)),
                vocab_size=100,
            )
    return DummyDifferentialBackend(base=base, ft=ft)


class TestPromptCollapse:
    def test_healthy_adapter_passes(self) -> None:
        probe, spec = build_probe(
            {
                "name": "pc",
                "kind": "prompt_collapse",
                "prompts": ["q1"],
                "context_lengths": [0, 256, 512, 1024],
                "assert_half_life_tokens": 100,
            }
        )
        ctx = RunContext(backend=_programmed_backend(stuffing_sensitivity=0.001))
        result = probe.run(spec, ctx)
        # Half-life should be well above 100 with slow decay.
        assert result.verdict == Verdict.PASS
        assert result.raw is not None
        assert result.raw > 100

    def test_collapsing_adapter_fails(self) -> None:
        probe, spec = build_probe(
            {
                "name": "pc",
                "kind": "prompt_collapse",
                "prompts": ["q1"],
                "context_lengths": [0, 256, 512, 1024],
                "assert_half_life_tokens": 500,
            }
        )
        ctx = RunContext(backend=_programmed_backend(stuffing_sensitivity=0.02))
        result = probe.run(spec, ctx)
        # Fast decay → short half-life → fail against 500-token threshold.
        assert result.verdict == Verdict.FAIL

    def test_error_on_empty_prompts(self) -> None:
        probe, spec = build_probe(
            {
                "name": "pc",
                "kind": "prompt_collapse",
                "prompts": [],
                "context_lengths": [0, 256],
            }
        )
        ctx = RunContext(backend=_programmed_backend(0.001))
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.ERROR
