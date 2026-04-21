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

import hashlib
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from dlm_sway.backends._instrumentation import BackendInstrumentation
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

    def __init__(
        self,
        mode: Mode,
        responses: DummyResponses,
        inst: BackendInstrumentation | None = None,
    ) -> None:
        # ``id`` is what surfaces in cache keys and trace events; widened
        # to ``str`` so scaled / null views can override it with
        # e.g. ``"scaled_0.50"`` / ``"null_42"`` without a cast dance.
        self.id: str = mode
        self._mode: Mode = mode
        self._r = responses
        # Private instrumentation when the caller didn't supply one —
        # keeps direct ``_DummyView("base", DummyResponses())``
        # constructions (common in probe tests that grab a view
        # without going through the differential backend) working
        # transparently.
        self._inst: BackendInstrumentation = inst if inst is not None else BackendInstrumentation()

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
        return self._inst.cached(
            "logprob_of",
            self.id,
            f"{prompt}\x00{completion}",
            0,
            lambda: self._r.logprobs.get((prompt, completion), -10.0),
        )

    def rolling_logprob(self, text: str) -> RollingLogprob:
        return self._inst.cached(
            "rolling_logprob",
            self.id,
            text,
            0,
            lambda: self._compute_rolling_logprob(text),
        )

    def _compute_rolling_logprob(self, text: str) -> RollingLogprob:
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
        return self._inst.cached(
            "next_token_dist",
            self.id,
            prompt,
            top_k,
            lambda: self._compute_next_token_dist(prompt, top_k=top_k),
        )

    def _compute_next_token_dist(self, prompt: str, *, top_k: int = 256) -> TokenDist:
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
            # More uniform mass across the top-k tokens. A *literally*
            # uniform distribution looks like a broken lm_head to
            # ``_divergence``'s degenerate-distribution guard, so we
            # add a tiny monotonic perturbation — the spread stays
            # small enough that the ft dist still looks "broad" to
            # KL/JS but clears the 1e-9 uniformity threshold a real
            # model would also clear via fp32 accumulation noise.
            lp = np.full(k, -math.log(k), dtype=np.float32)
            lp += np.linspace(-1e-4, 1e-4, k, dtype=np.float32)
        # B6: tail_logprob=None means "no measurable tail" (k covers vocab
        # or residual underflowed); reserve floats for measurable mass.
        residual = 1.0 - float(np.exp(lp).sum())
        tail_lp = math.log(residual) if residual > 1e-12 else None
        return TokenDist(
            token_ids=np.arange(k, dtype=np.int64),
            logprobs=lp,
            vocab_size=vocab,
            tail_logprob=tail_lp,
        )


class _NullView(_DummyView):
    """A dummy view that perturbs the base distribution with seeded noise.

    Used by :meth:`DummyDifferentialBackend.as_null_adapter`. The
    perturbation is small (matches an ``init_scale=0.02`` adapter) so
    the null-vs-base divergence stays well below real-adapter territory
    in probe tests.

    ``rank_scale`` simulates changing the effective LoRA rank: the
    output variance of ``A·B`` scales linearly with rank, so the noise
    std carries a ``sqrt(rank_scale)`` factor. Default 1.0 preserves
    pre-S10 behavior exactly.
    """

    def __init__(
        self,
        base_responses: DummyResponses,
        seed: int,
        init_scale: float,
        rank_scale: float = 1.0,
    ) -> None:
        super().__init__("base", base_responses)
        self._seed = seed
        self._init_scale = init_scale
        self._rank_scale = rank_scale

    def next_token_dist(self, prompt: str, *, top_k: int = 256) -> TokenDist:
        base_dist = super().next_token_dist(prompt, top_k=top_k)
        # F08 — Python's built-in ``hash(str)`` is salted per-process via
        # ``PYTHONHASHSEED``, so the same ``(self._seed, prompt)`` pair
        # produced different RNG streams across interpreter invocations.
        # That violated the README's determinism contract and meant the
        # null-stats disk cache (~/.dlm-sway/null-stats/) could serve
        # stale values on a restart. ``hashlib.md5`` is stable; taking
        # the first 8 hex digits keeps the arithmetic in 32-bit range.
        prompt_hash = int(hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(self._seed + prompt_hash % 1_000_003)
        effective_scale = self._init_scale * math.sqrt(self._rank_scale)
        noise = rng.normal(0.0, effective_scale, size=base_dist.logprobs.shape).astype(np.float32)
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

    Dummy declares ``safe_for_concurrent_views = False`` to mirror the
    shipped backends' posture; tests that want to exercise the concurrent
    scheduling path can subclass and set it ``True``.

    Also implements
    :class:`~dlm_sway.core.scoring.ScalableDifferentialBackend` with a
    linear-blend between base and ft responses, so probes that need
    ``as_scaled_adapter`` (N2 AdapterAblation) are unit-testable.
    """

    safe_for_concurrent_views: bool = False

    def __init__(self, *, base: DummyResponses, ft: DummyResponses) -> None:
        self._base_r = base
        self._ft_r = ft
        # Sprint 07: one shared cache + trace + stats instance that
        # every view yielded by this backend reads/writes. Tests can
        # peek at ``backend._inst.stats`` to assert cache behavior.
        self._inst = BackendInstrumentation()
        self._base = _DummyView("base", base, inst=self._inst)
        self._ft = _DummyView("ft", ft, inst=self._inst)
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
            view = _InterpolatedView(self._base_r, self._ft_r, lam)
            view._inst = self._inst
            view.id = f"scaled_{lam:.2f}"
            yield view
        finally:
            self._exit()

    @contextmanager
    def as_null_adapter(
        self,
        seed: int,
        *,
        init_scale: float = 0.02,
        rank_scale: float = 1.0,
    ) -> Iterator[_DummyView]:
        if rank_scale <= 0.0 or not math.isfinite(rank_scale):
            raise ValueError(f"rank_scale must be positive and finite; got {rank_scale!r}")
        label = f"null({seed})" if rank_scale == 1.0 else f"null({seed},rank={rank_scale:.2f})"
        view_id = f"null_{seed}" if rank_scale == 1.0 else f"null_{seed}_rank{rank_scale:.2f}"
        self._enter(label)
        try:
            view = _NullView(self._base_r, seed=seed, init_scale=init_scale, rank_scale=rank_scale)
            view._inst = self._inst
            view.id = view_id
            yield view
        finally:
            self._exit()

    def preflight_finite_check(self) -> tuple[bool, str]:
        """Smoke a single forward pass per view; reject non-finite logits.

        For the dummy backend the canned data is finite by construction
        unless tests deliberately seed NaN-laden ``TokenDist`` entries —
        which is exactly what S01 tests do to verify the runner gate.
        """
        prompt = "preflight"
        try:
            with self.as_base() as base_view:
                base_dist = base_view.next_token_dist(prompt, top_k=8)
            with self.as_finetuned() as ft_view:
                ft_dist = ft_view.next_token_dist(prompt, top_k=8)
        except Exception as exc:  # noqa: BLE001
            return False, f"preflight raised {type(exc).__name__}: {exc}"

        for label, dist in (("base", base_dist), ("ft", ft_dist)):
            if not np.all(np.isfinite(dist.logprobs)):
                n_bad = int((~np.isfinite(dist.logprobs)).sum())
                return (
                    False,
                    f"{label} view produced {n_bad}/{dist.logprobs.size} non-finite "
                    f"logprob(s) on prompt {prompt!r}",
                )
        return True, ""

    def _enter(self, mode: str) -> None:
        if self._active is not None:
            raise RuntimeError(
                f"DifferentialBackend view already active ({self._active!r}); "
                f"exit the current view before entering {mode!r}."
            )
        self._active = mode

    def _exit(self) -> None:
        self._active = None
