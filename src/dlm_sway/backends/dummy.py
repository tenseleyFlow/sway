"""In-memory backend for unit tests.

Deterministic, torchless, and trivially fast. Tests pass canned responses
and canned score tables keyed by ``(mode, prompt, completion)``. The same
backend instance serves as both ``as_base`` and ``as_finetuned`` — it
switches an internal mode flag.

Use it to drive every probe's unit test without loading a real model.
For integration tests against a real PEFT adapter, see
:class:`~dlm_sway.backends.hf.HuggingFaceDifferentialBackend`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from dlm_sway.core.scoring import RollingLogprob, TokenDist

Mode = Literal["base", "ft"]


@dataclass(slots=True)
class DummyResponses:
    """Canned data for one mode (base or ft).

    Callers populate one of these per mode and hand both to
    :class:`DummyDifferentialBackend`.
    """

    generations: dict[str, str] = field(default_factory=dict)
    """Prompt → canned completion. Lookup is exact-match."""
    logprobs: dict[tuple[str, str], float] = field(default_factory=dict)
    """``(prompt, completion) → sum logprob``. Default ``-10.0`` if missing."""
    rolling: dict[str, RollingLogprob] = field(default_factory=dict)
    """Text → canned :class:`RollingLogprob`."""
    token_dists: dict[str, TokenDist] = field(default_factory=dict)
    """Prompt → canned :class:`TokenDist`."""


class _DummyView:
    """The per-mode view yielded by ``as_base`` / ``as_finetuned``.

    Implements :class:`~dlm_sway.core.model.Model` *and*
    :class:`~dlm_sway.core.scoring.ScoringBackend` — i.e. the
    ``ScoringModel`` intersection.
    """

    def __init__(self, mode: Mode, responses: DummyResponses) -> None:
        self.id = mode
        self._mode: Mode = mode
        self._r = responses

    # -- Model ---------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> str:
        del max_new_tokens, temperature, top_p, seed  # canned; decoding is trivial.
        try:
            return self._r.generations[prompt]
        except KeyError as exc:
            raise KeyError(
                f"dummy backend ({self._mode}): no canned generation for prompt {prompt!r}"
            ) from exc

    def close(self) -> None:
        return None

    # -- ScoringBackend ------------------------------------------------
    def logprob_of(self, prompt: str, completion: str) -> float:
        return self._r.logprobs.get((prompt, completion), -10.0)

    def rolling_logprob(self, text: str) -> RollingLogprob:
        if text in self._r.rolling:
            return self._r.rolling[text]
        # Synthesize a plausible rolling logprob so probes that just
        # want a non-trivial value work without per-text configuration.
        tokens = text.split()
        n = max(len(tokens), 1)
        per_tok = -2.0 if self._mode == "base" else -1.5
        return RollingLogprob(
            token_ids=np.arange(n, dtype=np.int64),
            logprobs=np.full(max(n - 1, 0), per_tok, dtype=np.float32),
            num_tokens=n,
            total_logprob=per_tok * max(n - 1, 0),
        )

    def next_token_dist(self, prompt: str, *, top_k: int = 256) -> TokenDist:
        del top_k
        if prompt in self._r.token_dists:
            return self._r.token_dists[prompt]
        # Synthesize a sharp base / broad ft distribution so divergence
        # probes see a non-zero signal without hand-rolled data.
        vocab = 1000
        k = 8
        if self._mode == "base":
            lp = np.array([-0.1] + [-5.0] * (k - 1), dtype=np.float32)
        else:
            # More uniform mass across the top-k tokens.
            lp = np.full(k, -math.log(k), dtype=np.float32)
        return TokenDist(
            token_ids=np.arange(k, dtype=np.int64),
            logprobs=lp,
            vocab_size=vocab,
            tail_logprob=math.log1p(-float(np.exp(lp).sum())) if np.exp(lp).sum() < 1 else 0.0,
        )


class _NullView(_DummyView):
    """A dummy view that perturbs the base distribution with seeded noise.

    Used by :meth:`DummyDifferentialBackend.as_null_adapter`. The
    perturbation is small (matches an ``init_scale=0.02`` adapter) so
    the null-vs-base divergence stays well below real-adapter territory
    in probe tests.
    """

    def __init__(self, base_responses: DummyResponses, seed: int, init_scale: float) -> None:
        super().__init__("base", base_responses)
        self._seed = seed
        self._init_scale = init_scale

    def next_token_dist(self, prompt: str, *, top_k: int = 256) -> TokenDist:
        base_dist = super().next_token_dist(prompt, top_k=top_k)
        rng = np.random.default_rng(self._seed + hash(prompt) % 1_000_003)
        noise = rng.normal(0.0, self._init_scale, size=base_dist.logprobs.shape).astype(np.float32)
        new_lp = base_dist.logprobs + noise
        # Re-normalize (within the top-k slice) so a valid distribution comes back.
        max_lp = new_lp.max()
        new_probs = np.exp(new_lp - max_lp)
        new_probs /= new_probs.sum()
        return TokenDist(
            token_ids=base_dist.token_ids,
            logprobs=np.log(new_probs).astype(np.float32),
            vocab_size=base_dist.vocab_size,
            tail_logprob=base_dist.tail_logprob,
        )


class _InterpolatedView(_DummyView):
    """A dummy view where logits/dists are a lam-blend of base and ft.

    Used by :meth:`DummyDifferentialBackend.as_scaled_adapter`.
    Generation falls back to the ft view at lam>=0.5, base otherwise —
    rounded because the dummy backend's generations are canned strings
    with no notion of "how much".
    """

    def __init__(
        self,
        base_responses: DummyResponses,
        ft_responses: DummyResponses,
        lam: float,
    ) -> None:
        super().__init__(
            "ft" if lam >= 0.5 else "base", ft_responses if lam >= 0.5 else base_responses
        )
        self._base_r = base_responses
        self._ft_r = ft_responses
        self._lam = lam

    def logprob_of(self, prompt: str, completion: str) -> float:
        base_v = self._base_r.logprobs.get((prompt, completion), -10.0)
        ft_v = self._ft_r.logprobs.get((prompt, completion), -10.0)
        return (1 - self._lam) * base_v + self._lam * ft_v

    def next_token_dist(self, prompt: str, *, top_k: int = 256):  # type: ignore[no-untyped-def]
        base_dist = _DummyView("base", self._base_r).next_token_dist(prompt, top_k=top_k)
        ft_dist = _DummyView("ft", self._ft_r).next_token_dist(prompt, top_k=top_k)
        # Both dists are on the same synthetic support when unseeded; blend
        # their logprobs via log-space linear interpolation, which is a
        # log-linear "tempered" mix and keeps normalization close enough.
        lam = self._lam
        blended_lp = (1 - lam) * base_dist.logprobs + lam * ft_dist.logprobs
        return type(base_dist)(
            token_ids=base_dist.token_ids,
            logprobs=blended_lp,
            vocab_size=base_dist.vocab_size,
            tail_logprob=base_dist.tail_logprob,
        )


class DummyDifferentialBackend:
    """Dummy implementation of
    :class:`~dlm_sway.core.scoring.DifferentialBackend`.

    Construction takes one :class:`DummyResponses` per mode. The two
    modes are mutually exclusive — the backend enforces that callers
    exit one view before entering the other, catching bugs in probes
    that hold a stale view across a toggle.

    Also implements
    :class:`~dlm_sway.core.scoring.ScalableDifferentialBackend` with a
    linear-blend between base and ft responses, so probes that need
    ``as_scaled_adapter`` (N2 AdapterAblation) are unit-testable.
    """

    def __init__(self, *, base: DummyResponses, ft: DummyResponses) -> None:
        self._base_r = base
        self._ft_r = ft
        self._base = _DummyView("base", base)
        self._ft = _DummyView("ft", ft)
        self._active: str | None = None

    @contextmanager
    def as_base(self) -> Iterator[_DummyView]:
        self._enter("base")
        try:
            yield self._base
        finally:
            self._exit()

    @contextmanager
    def as_finetuned(self) -> Iterator[_DummyView]:
        self._enter("ft")
        try:
            yield self._ft
        finally:
            self._exit()

    @contextmanager
    def as_scaled_adapter(self, lam: float) -> Iterator[_DummyView]:
        self._enter(f"scaled({lam})")
        try:
            yield _InterpolatedView(self._base_r, self._ft_r, lam)
        finally:
            self._exit()

    @contextmanager
    def as_null_adapter(self, seed: int, *, init_scale: float = 0.02) -> Iterator[_DummyView]:
        self._enter(f"null({seed})")
        try:
            yield _NullView(self._base_r, seed=seed, init_scale=init_scale)
        finally:
            self._exit()

    def _enter(self, mode: str) -> None:
        if self._active is not None:
            raise RuntimeError(
                f"DifferentialBackend view already active ({self._active!r}); "
                f"exit the current view before entering {mode!r}."
            )
        self._active = mode

    def _exit(self) -> None:
        self._active = None
