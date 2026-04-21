"""HTTP scoring backend for OpenAI-compatible completions servers (S13, F7).

Targets the three canonical deployments of the OpenAI completions API
shape:

- **OpenAI platform** — ``https://api.openai.com``, legacy ``/v1/completions``
  still serves gpt-3.5-turbo-instruct and fine-tuned variants of it.
- **vLLM serve** — ``vllm serve <model>`` exposes an OpenAI-compatible
  ``/v1/completions`` on the local port, supports ``echo=True`` and
  ``logprobs=N``.
- **Ollama** — recent versions expose the same shape via
  ``http://localhost:11434/v1/completions`` with ``logprobs`` support.

Unlike the HF backend, this one implements :class:`ScoringBackend`
*only* — there's no local model to toggle between base and adapter.
Users who want a full differential run wire two ``ApiScoringBackend``
instances behind
:class:`~dlm_sway.backends.two_model.TwoModelDifferential` — one
pointing at the base model, one at the fine-tuned endpoint.

This is also the first shipped backend that sets
``safe_for_concurrent_views = True``. The endpoint is stateless, and
the S07 runner scaffolding can dispatch probes in parallel as soon
as the pool implementation lands.

Shape of a typical response (OpenAI-compatible)::

    {
      "choices": [{
        "text": "...",
        "logprobs": {
          "tokens": ["The", " cat"],
          "token_logprobs": [null, -2.3],
          "top_logprobs": [{"The": -0.1, ...}, {...}]
        }
      }]
    }

Lazy httpx/tenacity imports keep sway usable without the ``[api]``
extra installed. A descriptive :class:`RuntimeError` surfaces when
a user reaches for this backend without the deps.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

from dlm_sway.backends._instrumentation import BackendInstrumentation
from dlm_sway.core.errors import ProbeError
from dlm_sway.core.model import LoadedModel, Model
from dlm_sway.core.scoring import RollingLogprob, TokenDist

if TYPE_CHECKING:
    from collections.abc import Iterator

    import httpx


#: Default vocab size reported when the caller doesn't supply one.
#: Divergence probes that use ``TokenDist.vocab_size`` will see a
#: ``None`` ``tail_logprob`` and degrade gracefully; no correctness
#: risk, just a less precise KL/JS residual.
_UNKNOWN_VOCAB_SIZE: int = 0


class ApiScoringBackend:
    """Scoring against an OpenAI-compatible ``/v1/completions`` endpoint.

    Parameters
    ----------
    base_url:
        Root URL of the completions server (``https://api.openai.com``,
        ``http://localhost:11434``, etc). The ``/v1/completions`` path
        is appended; do **not** include it here.
    model_name:
        The model identifier the server expects in the ``model`` field
        of each request (``gpt-3.5-turbo-instruct``, a local vLLM
        ``--model`` name, an Ollama pulled name).
    api_key:
        Bearer token for the ``Authorization`` header. ``None``
        (default) consults ``SWAY_API_KEY`` then ``OPENAI_API_KEY``;
        still ``None`` means no auth header is sent (typical for local
        vLLM / Ollama).
    vocab_size:
        Full vocab size of the server's tokenizer. Used by ``TokenDist``
        to compute the tail-probability residual. Pass the tokenizer's
        vocab_size from ``transformers.AutoTokenizer(...)``; when
        unknown, leave as ``None`` — divergence probes handle the
        missing tail gracefully.
    timeout_s:
        Per-request HTTP timeout.
    max_retries:
        Tenacity retry count on 5xx / connection errors. Set to 0 to
        disable retries (useful for unit tests).
    """

    #: Stateless HTTP — safe for the S07 concurrent-probe scheduler.
    safe_for_concurrent_views: bool = True

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str | None = None,
        vocab_size: int | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        try:
            import httpx  # noqa: F401 — import-or-raise probe
        except ImportError as exc:
            raise RuntimeError(
                "ApiScoringBackend requires the [api] extra: pip install 'dlm-sway[api]'"
            ) from exc

        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._api_key = api_key if api_key is not None else _default_api_key()
        self._vocab_size = vocab_size if vocab_size is not None else _UNKNOWN_VOCAB_SIZE
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        # BackendInstrumentation shares the S07 LRU + trace surface so
        # `ApiScoringBackend` participates in the same cache stats the
        # HF and dummy backends already report.
        self._inst = BackendInstrumentation()
        self.id = f"api:{model_name}"
        self._client: httpx.Client | None = None

    # -- Lifecycle -----------------------------------------------------

    def _ensure_client(self) -> Any:
        """Lazily construct the httpx client. One per backend instance."""
        if self._client is None:
            import httpx

            headers: dict[str, str] = {"content-type": "application/json"}
            if self._api_key:
                headers["authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout_s,
                headers=headers,
            )
        return self._client

    def close(self) -> None:
        """Release the httpx connection pool. Safe to call more than once."""
        if self._client is not None:
            self._client.close()
            self._client = None
        inst = getattr(self, "_inst", None)
        if inst is not None:
            inst.close()

    # -- Model interface (for Model & ScoringBackend composition) ------

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> str:
        """Complete ``prompt`` by ``max_new_tokens`` tokens.

        Thin wrapper around ``/v1/completions`` — used by
        :mod:`dlm_sway.probes.leakage` and similar probes that need
        generated text rather than logprobs. Deterministic when
        ``temperature=0``; otherwise the server's sampling RNG governs.
        """
        payload: dict[str, Any] = {
            "model": self._model_name,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        # ``seed`` is supported by OpenAI + vLLM; Ollama ignores it
        # without erroring. We always send it so deterministic seeds
        # do deterministic things wherever they're honored.
        payload["seed"] = int(seed)
        data = self._post_completions(payload)
        try:
            return str(data["choices"][0]["text"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ProbeError("api.generate", f"malformed completion response: {data!r}") from exc

    # -- ScoringBackend ------------------------------------------------

    def logprob_of(self, prompt: str, completion: str) -> float:
        """Sum of token logprobs for ``completion`` given ``prompt``.

        Implementation: POST ``prompt + completion`` with ``echo=True,
        max_tokens=0, logprobs=0``. The server echoes per-token
        logprobs for the full input; we walk forward character-by-
        character through the echoed tokens until we've covered the
        prompt, then sum logprobs of everything after.

        The API doesn't return token ids per se — for this method we
        only need logprobs and string lengths, so the sum is a clean
        contract regardless of tokenizer.
        """
        if not completion:
            raise ProbeError("api.logprob_of", "completion tokenized to zero tokens")
        return self._inst.cached(
            "logprob_of",
            self.id,
            f"{prompt}\x00{completion}",
            0,
            lambda: self._logprob_of_uncached(prompt, completion),
        )

    def _logprob_of_uncached(self, prompt: str, completion: str) -> float:
        payload = {
            "model": self._model_name,
            "prompt": prompt + completion,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 0,
        }
        data = self._post_completions(payload)
        tokens, token_logprobs = _extract_echo_logprobs(data)
        if not tokens:
            raise ProbeError(
                "api.logprob_of",
                "server returned empty echo; cannot score completion",
            )

        prompt_end_idx = _split_echo_at_char(tokens, len(prompt))
        completion_logprobs = token_logprobs[prompt_end_idx:]
        # First token's logprob is ``None`` by API contract; guard only
        # if the split lands right at the boundary. Sum the rest.
        total = 0.0
        for lp in completion_logprobs:
            if lp is None:
                continue
            total += float(lp)
        if not math.isfinite(total):
            raise ProbeError("api.logprob_of", f"non-finite logprob sum from API ({total})")
        return total

    def rolling_logprob(self, text: str) -> RollingLogprob:
        """Per-token logprobs of ``text`` end-to-end.

        POSTs ``text`` with ``echo=True, max_tokens=0, logprobs=0``;
        converts the returned ``token_logprobs`` array into a
        :class:`RollingLogprob`. Token ids are synthesized as
        ``arange(N)`` — downstream probes don't introspect ids for
        rolling logprobs (only for :meth:`next_token_dist`).
        """
        return self._inst.cached(
            "rolling_logprob",
            self.id,
            text,
            0,
            lambda: self._rolling_logprob_uncached(text),
        )

    def _rolling_logprob_uncached(self, text: str) -> RollingLogprob:
        payload = {
            "model": self._model_name,
            "prompt": text,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 0,
        }
        data = self._post_completions(payload)
        tokens, token_logprobs = _extract_echo_logprobs(data)
        n = len(tokens)
        # token_logprobs[0] is ``None`` (no context for the first
        # token), so rolling logprobs are token_logprobs[1:] — length
        # n-1, matching the RollingLogprob contract.
        numeric = [float(lp) for lp in token_logprobs[1:] if lp is not None]
        lp_arr = np.asarray(numeric, dtype=np.float32)
        total = float(lp_arr.sum()) if lp_arr.size else 0.0
        return RollingLogprob(
            token_ids=np.arange(n, dtype=np.int64),
            logprobs=lp_arr,
            num_tokens=n,
            total_logprob=total,
        )

    def next_token_dist(self, prompt: str, *, top_k: int = 256) -> TokenDist:
        """Top-k next-token distribution at the position after ``prompt``.

        POSTs with ``echo=False, max_tokens=1, logprobs=<top_k>``. The
        response's ``top_logprobs[0]`` is a ``{token_str: logprob}``
        dict; we sort by descending logprob and build a
        :class:`TokenDist`. Token ids are synthesized (APIs don't
        expose vocab indices) — divergence probes that align by id
        order work correctly because the sort is deterministic.

        ``vocab_size`` from the constructor populates the residual-
        tail math when set; unknown → ``tail_logprob=None`` and
        divergence helpers fall back to even-vocab redistribution.
        """
        return self._inst.cached(
            "next_token_dist",
            self.id,
            prompt,
            top_k,
            lambda: self._next_token_dist_uncached(prompt, top_k=top_k),
        )

    def _next_token_dist_uncached(self, prompt: str, *, top_k: int) -> TokenDist:
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "max_tokens": 1,
            "echo": False,
            "logprobs": max(1, min(top_k, 20)),  # OpenAI caps at 20
        }
        data = self._post_completions(payload)
        try:
            top_logprobs: dict[str, float] = data["choices"][0]["logprobs"]["top_logprobs"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProbeError(
                "api.next_token_dist", f"malformed top_logprobs response: {data!r}"
            ) from exc
        if not top_logprobs:
            raise ProbeError("api.next_token_dist", "server returned empty top_logprobs")
        # Sort descending by logprob; truncate to requested k (the
        # server cap may have been lower).
        items = sorted(top_logprobs.items(), key=lambda kv: -kv[1])[:top_k]
        lp_vals = np.asarray([lp for _, lp in items], dtype=np.float32)
        if not np.all(np.isfinite(lp_vals)):
            raise ProbeError(
                "api.next_token_dist",
                f"non-finite top_logprob from API ({lp_vals.tolist()})",
            )
        # Synthesize ids — see docstring.
        ids = np.arange(len(items), dtype=np.int64)
        # Residual: 1 - sum(exp(lp)). Only meaningful when vocab_size
        # is known and > k; otherwise we report None.
        tail_lp: float | None = None
        if self._vocab_size > len(items):
            residual = 1.0 - float(np.exp(lp_vals).sum())
            if residual > 1e-12:
                tail_lp = math.log(residual)
            elif residual >= 0.0:
                tail_lp = 0.0
        vocab = self._vocab_size if self._vocab_size else len(items)
        return TokenDist(
            token_ids=ids,
            logprobs=lp_vals,
            vocab_size=vocab,
            tail_logprob=tail_lp,
        )

    # -- PreflightCheckable --------------------------------------------

    def preflight_finite_check(self) -> tuple[bool, str]:
        """Smoke one completions call; reject on non-finite / empty response.

        Hit ``/v1/completions`` with ``prompt="hello"``, ``max_tokens=1``,
        ``logprobs=1``. A live OpenAI-compatible server returns within
        a few seconds; anything that errors, times out, or returns
        non-finite logprobs fails preflight before the suite runs.
        """
        try:
            dist = self.next_token_dist("hello", top_k=1)
        except Exception as exc:  # noqa: BLE001 — preflight is catch-all
            return False, f"preflight raised {type(exc).__name__}: {exc}"
        if not np.all(np.isfinite(dist.logprobs)):
            n_bad = int((~np.isfinite(dist.logprobs)).sum())
            return (
                False,
                f"api view produced {n_bad}/{dist.logprobs.size} non-finite logprob(s)",
            )
        return True, ""

    # -- DifferentialBackend stub (for users who wire two API backends
    #    behind TwoModelDifferential, this backend exposes as_base only
    #    as a convenience — the TwoModelDifferential does the real
    #    toggling). Not required by the Protocol.

    @contextmanager
    def as_base(self) -> Iterator[Model]:
        """Yield ``self`` as a Model/ScoringBackend view.

        The API backend has no distinction between "base" and "ft"; a
        single ``ApiScoringBackend`` represents one endpoint. Kept for
        syntactic symmetry so a user can stack a single API backend
        against a local HF backend via :class:`TwoModelDifferential`.
        """
        yield self  # Model is a Protocol; self satisfies it structurally

    @contextmanager
    def as_finetuned(self) -> Iterator[Model]:
        """Same view as :meth:`as_base` — see that docstring."""
        yield self

    # -- HTTP plumbing --------------------------------------------------

    def _post_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST to ``/v1/completions`` with tenacity retry on 5xx/net errors."""
        client = self._ensure_client()
        retrying = _build_retrier(self._max_retries)

        def _once() -> dict[str, Any]:
            import httpx

            try:
                response = client.post("/v1/completions", json=payload)
            except httpx.RequestError as exc:
                raise ProbeError("api.http", f"api request error: {exc}") from exc
            if response.status_code >= 500:
                response.raise_for_status()
            if response.status_code >= 400:
                raise ProbeError(
                    "api.http",
                    f"api returned {response.status_code}: {response.text[:200]}",
                )
            try:
                return response.json()  # type: ignore[no-any-return]
            except ValueError as exc:
                raise ProbeError(
                    "api.http",
                    f"api returned non-JSON body: {response.text[:200]}",
                ) from exc

        for attempt in retrying:
            with attempt:
                return _once()
        # retrying generators always yield at least once; if we reach
        # here the retry loop exhausted itself, which tenacity surfaces
        # via RetryError — defensive fallback.
        raise ProbeError("api.http", "retry loop exhausted without returning a response")

    # -- Instrumentation passthrough (for S07 cache + stats) -----------

    @property
    def _instrumentation(self) -> BackendInstrumentation:
        return self._inst

    def cache_identity(self) -> str:
        """Stable identity for the null-stats disk cache (S02/S10)."""
        return f"api:{self._base_url}:{self._model_name}"

    def load(self) -> LoadedModel:
        """Satisfy callers that probe for ``load()`` — there is no local
        model to return, so we raise with a pointed hint."""
        raise ProbeError(
            "api.load",
            "ApiScoringBackend has no local LoadedModel — use the backend's "
            "scoring methods directly, or wire through TwoModelDifferential.",
        )


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _default_api_key() -> str | None:
    """Read ``SWAY_API_KEY`` or ``OPENAI_API_KEY`` from the environment."""
    return os.environ.get("SWAY_API_KEY") or os.environ.get("OPENAI_API_KEY")


def _extract_echo_logprobs(
    data: dict[str, Any],
) -> tuple[list[str], list[float | None]]:
    """Pull ``(tokens, token_logprobs)`` from an echo response.

    Raises :class:`ProbeError` on shapes that don't carry the expected
    fields — the server either didn't honor ``echo=True``/``logprobs``
    or returned an error-ish payload.
    """
    try:
        choice = data["choices"][0]
        lp = choice["logprobs"]
        tokens = list(lp["tokens"])
        token_logprobs = list(lp["token_logprobs"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ProbeError("api.http", f"expected echo logprobs shape, got: {data!r}") from exc
    if len(tokens) != len(token_logprobs):
        raise ProbeError(
            "api.http",
            f"tokens/token_logprobs length mismatch: {len(tokens)} vs {len(token_logprobs)}",
        )
    return tokens, token_logprobs


def _split_echo_at_char(tokens: list[str], prompt_chars: int) -> int:
    """Find the token index where the character count crosses ``prompt_chars``.

    Walks the token strings accumulating character lengths until the
    running total meets or exceeds ``prompt_chars``. Returns the index
    of the first token *after* the prompt — the index at which the
    completion begins. Used by :meth:`logprob_of` to split an echo
    response into prompt and completion spans.

    When the boundary falls mid-token (the prompt doesn't align with a
    token edge), we include the partial token in the prompt side —
    conservative for ``logprob_of`` because any mis-attribution leans
    toward over-counting the prompt's contribution, not the
    completion's.
    """
    if prompt_chars <= 0:
        return 0
    total = 0
    for i, tok in enumerate(tokens):
        total += len(tok)
        if total >= prompt_chars:
            return i + 1
    return len(tokens)


def _build_retrier(max_retries: int) -> Any:
    """Tenacity retrier for 5xx + network errors; exponential backoff.

    ``max_retries=0`` returns a single-shot no-retry iterator so unit
    tests can exercise success paths without tenacity's exponential
    sleep delaying CI.
    """
    import httpx
    from tenacity import (
        Retrying,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    return Retrying(
        stop=stop_after_attempt(max(1, max_retries + 1)),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
        reraise=True,
    )


__all__ = ["ApiScoringBackend"]
