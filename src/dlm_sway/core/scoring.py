"""Scoring protocols: logprobs, next-token distributions, differential toggling.

Scoring is **separate** from generation because not every backend can
provide logits. Every numeric sway probe depends on at least one of
three operations:

1. ``logprob_of(prompt, completion)`` — score a completion against a
   prompt (A1, B2, B3, C2, …).
2. ``rolling_logprob(text)`` — perplexity over a piece of text (B1,
   C2).
3. ``next_token_dist(prompt, top_k)`` — the raw next-token distribution
   at a single position (A1, N2).

The :class:`DifferentialBackend` is the key performance primitive:
both base and fine-tuned views share the same loaded weights and KV
cache layout, toggled via PEFT's :meth:`set_adapter` /
:meth:`disable_adapter`. A naive "load twice" implementation would
double memory and halve throughput.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from dlm_sway.core.model import Model


@dataclass(frozen=True, slots=True)
class RollingLogprob:
    """Per-token logprobs over a piece of text, plus summary stats.

    Attributes
    ----------
    token_ids:
        The tokenizer output for ``text``. Length ``N``.
    logprobs:
        ``log p(token_i | token_<i)`` for each position i ≥ 1. Length
        ``N-1``.
    num_tokens:
        ``N`` — included for convenience; ``len(token_ids)``.
    total_logprob:
        Sum of :attr:`logprobs`.
    """

    token_ids: NDArray[np.int64]
    logprobs: NDArray[np.float32]
    num_tokens: int
    total_logprob: float

    @property
    def mean_logprob(self) -> float:
        n = self.logprobs.size
        return float(self.total_logprob / n) if n else 0.0

    @property
    def perplexity(self) -> float:
        """``exp(-mean_logprob)``. Base-e, natural perplexity."""
        return float(np.exp(-self.mean_logprob))


@dataclass(frozen=True, slots=True)
class TokenDist:
    """A (possibly top-k truncated) next-token probability distribution.

    For KL / JS divergence probes sway needs matched distributions
    across base and fine-tuned views. The runner is responsible for
    aligning ``top_k`` token slices between two ``TokenDist`` objects
    before handing them to divergence math.
    """

    token_ids: NDArray[np.int64]
    """Token ids, descending by probability. Length ``k``."""
    logprobs: NDArray[np.float32]
    """Log-probabilities for :attr:`token_ids`. Length ``k``."""
    vocab_size: int
    """Full vocab size — needed to renormalize top-k truncated slices."""
    tail_logprob: float = field(default=0.0)
    """log of (1 - sum of exp(logprobs[:k])); 0 if top_k covers the full vocab."""


@runtime_checkable
class ScoringBackend(Protocol):
    """Logit-level access to a loaded model."""

    def logprob_of(self, prompt: str, completion: str) -> float:
        """Sum of log-probabilities of ``completion`` tokens given ``prompt``.

        The prompt is *not* scored; only the completion contributes. The
        value is in nats (natural log). Longer completions are
        monotonically more negative — callers normalize by length if
        they need a rate.
        """
        ...

    def rolling_logprob(self, text: str) -> RollingLogprob:
        """Compute per-token logprobs for the whole of ``text``.

        Equivalent to lm-eval's ``loglikelihood_rolling``. Used for
        perplexity comparison on held-out content (B1 SIS, C2).
        """
        ...

    def next_token_dist(self, prompt: str, *, top_k: int = 256) -> TokenDist:
        """Next-token distribution at the position after ``prompt``.

        Truncated to ``top_k`` for memory; callers doing divergence math
        over the top-k slice accept the (typically negligible) error vs
        full-vocab KL.
        """
        ...


@runtime_checkable
class DifferentialBackend(Protocol):
    """A backend that holds base + fine-tuned views on a single loaded model.

    The idiomatic usage is::

        with backend.as_base() as base_view:
            p_base = base_view.next_token_dist(prompt)
        with backend.as_finetuned() as ft_view:
            p_ft = ft_view.next_token_dist(prompt)

    Implementations toggle PEFT adapters via
    :meth:`peft.PeftModel.set_adapter` / :meth:`disable_adapter`.

    Invariant: the two views must be **not simultaneously usable**. A
    caller holding a ``base_view`` after entering the ``as_finetuned``
    context is a programmer error and implementations MUST detect and
    raise.
    """

    def as_base(self) -> AbstractContextManager[_ScoringModel]: ...

    def as_finetuned(self) -> AbstractContextManager[_ScoringModel]: ...


@runtime_checkable
class ScalableDifferentialBackend(DifferentialBackend, Protocol):
    """A differential backend that can also scale the LoRA additive term.

    LoRA applies ``W + (alpha/r) · B @ A`` to a base weight matrix. This
    protocol exposes a context manager that temporarily multiplies that
    additive term by ``lam`` for everything inside the ``with`` block.

    ``lam = 0.0`` is equivalent to :meth:`as_base`.
    ``lam = 1.0`` is equivalent to :meth:`as_finetuned`.
    ``lam = 1.25`` overshoots — useful for N2 AdapterAblation's
    response-curve measurement.

    Only the HF backend ships an implementation in v0.1. Probes that
    need scaling check via ``isinstance(backend, ScalableDifferentialBackend)``
    at runtime and SKIP gracefully when unavailable.
    """

    def as_scaled_adapter(self, lam: float) -> AbstractContextManager[_ScoringModel]: ...


@runtime_checkable
class NullCalibratedBackend(DifferentialBackend, Protocol):
    """A differential backend that can produce a "null adapter" view.

    A null adapter has the *same structure* (rank, alpha, target modules)
    as the real adapter but with weights drawn from a zero-mean Gaussian.
    Running probes against this view yields the baseline "how much
    signal does random noise produce" distribution — the denominator in
    every numeric probe's z-score.

    The context manager takes a ``seed`` so calibration runs can be
    reproduced and multiple independent null samples can be drawn to
    estimate ``std``.

    Implementations MUST restore the real adapter on exit, including
    on exceptions, so a caller can freely interleave null and real
    calibrations within the same backend lifetime.
    """

    def as_null_adapter(
        self, seed: int, *, init_scale: float = 0.02
    ) -> AbstractContextManager[_ScoringModel]: ...


# Helper Protocol for type-checking the yielded context object: it
# must satisfy both Model and ScoringBackend. mypy doesn't support
# intersection types, so we spell it out explicitly.
@runtime_checkable
class _ScoringModel(Model, ScoringBackend, Protocol):
    """A Model that also exposes ScoringBackend."""

    ...


ScoringModel = _ScoringModel
"""Public alias for the intersection ``Model & ScoringBackend``.

Exported for backend and probe implementations that need to annotate
variables of this combined type.
"""
