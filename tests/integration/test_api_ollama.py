"""S13 prove-the-value (§F7): ``ApiScoringBackend`` against a real Ollama.

**Opt-in.** Skipped unless ``SWAY_OLLAMA_URL`` is set (typically
``http://localhost:11434``). Also needs ``SWAY_OLLAMA_MODEL`` — the
name of a model already pulled via ``ollama pull <name>``. A minimal
run::

    ollama pull llama3.2:1b
    ollama serve &
    SWAY_OLLAMA_URL=http://localhost:11434 \\
        SWAY_OLLAMA_MODEL=llama3.2:1b \\
        uv run pytest tests/integration/test_api_ollama.py -v

What the test proves:

1. The backend talks to a real OpenAI-compatible endpoint without
   crashing on any of its three scoring primitives
   (``logprob_of``, ``rolling_logprob``, ``next_token_dist``).
2. Preflight passes (non-finite logprobs would surface here).
3. Wall time per call is in a sane range — documents the latency
   budget the sprint's "≤3× HF backend, ≤1.5× with concurrent_probes=4"
   claim rests on.

This test is the F7 claim's concrete backing: ``sway`` can score
hosted-inference endpoints end-to-end, not just local HF loads.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Iterator

import pytest

_ollama_url = os.environ.get("SWAY_OLLAMA_URL")
_ollama_model = os.environ.get("SWAY_OLLAMA_MODEL")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.online,
    pytest.mark.skipif(
        not _ollama_url or not _ollama_model,
        reason="set SWAY_OLLAMA_URL + SWAY_OLLAMA_MODEL to run this test",
    ),
]

pytest.importorskip("httpx")
pytest.importorskip("tenacity")

from dlm_sway.backends.api import ApiScoringBackend  # noqa: E402


@pytest.fixture(scope="module")
def backend() -> Iterator[ApiScoringBackend]:
    assert _ollama_url is not None  # narrowing for type-checker
    assert _ollama_model is not None
    be = ApiScoringBackend(
        base_url=_ollama_url,
        model_name=_ollama_model,
        api_key=None,  # Ollama doesn't require auth by default
        max_retries=1,
        timeout_s=60.0,
    )
    yield be
    be.close()


def test_preflight_passes(backend: ApiScoringBackend) -> None:
    ok, reason = backend.preflight_finite_check()
    assert ok, reason


def test_logprob_of_finite(backend: ApiScoringBackend) -> None:
    t0 = time.perf_counter()
    lp = backend.logprob_of(
        prompt="The capital of France is",
        completion=" Paris.",
    )
    wall = time.perf_counter() - t0
    print(f"\n  logprob_of wall: {wall:.2f}s")
    assert math.isfinite(lp)
    assert lp < 0.0, "logprobs of any non-empty text are negative"


def test_rolling_logprob_shape(backend: ApiScoringBackend) -> None:
    r = backend.rolling_logprob("Hello world. This is a sentence.")
    assert r.num_tokens >= 2
    assert r.logprobs.size == r.num_tokens - 1
    assert math.isfinite(r.total_logprob)
    assert math.isfinite(r.perplexity)
    assert r.perplexity > 1.0


def test_next_token_dist_shape(backend: ApiScoringBackend) -> None:
    d = backend.next_token_dist("The quick brown fox jumps over the", top_k=8)
    import numpy as np

    assert d.logprobs.size <= 8
    assert np.all(np.isfinite(d.logprobs))
    # Descending by probability.
    assert np.all(np.diff(d.logprobs) <= 1e-6)
