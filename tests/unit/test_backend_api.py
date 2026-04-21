"""Tests for :mod:`dlm_sway.backends.api` (S13 / F7).

The backend hits real HTTP in production. Here we use httpx's
``MockTransport`` to intercept every request and hand back canned
OpenAI-shaped responses. That lets us exercise the full request-build
+ response-parse roundtrip without running a server.
"""

from __future__ import annotations

import json as _json
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

pytest.importorskip("httpx")
pytest.importorskip("tenacity")

import httpx  # noqa: E402

from dlm_sway.backends.api import (  # noqa: E402
    ApiScoringBackend,
    _split_echo_at_char,
)
from dlm_sway.core.errors import ProbeError  # noqa: E402


def _echo_response(tokens: list[str], logprobs: list[float | None]) -> dict[str, Any]:
    """Shape an ``echo=True`` /v1/completions response."""
    return {
        "choices": [
            {
                "text": "".join(tokens),
                "logprobs": {
                    "tokens": tokens,
                    "token_logprobs": logprobs,
                    "top_logprobs": [],
                },
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": len(tokens),
            "completion_tokens": 0,
            "total_tokens": len(tokens),
        },
    }


def _top_logprobs_response(top: dict[str, float]) -> dict[str, Any]:
    """Shape a ``max_tokens=1, logprobs=K`` response."""
    return {
        "choices": [
            {
                "text": next(iter(top)),
                "logprobs": {
                    "tokens": [next(iter(top))],
                    "token_logprobs": [next(iter(top.values()))],
                    "top_logprobs": [top],
                },
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _make_backend(
    handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any
) -> ApiScoringBackend:
    """Construct an ApiScoringBackend whose httpx client uses a MockTransport."""
    backend = ApiScoringBackend(
        base_url="https://mock.example/",
        model_name="mock-model",
        api_key=None,
        max_retries=0,
        **kwargs,
    )
    # Swap the client with one wired to a MockTransport. We reuse the
    # backend's own lazy-init by forcing the client before any method
    # calls it.
    backend._client = httpx.Client(
        base_url="https://mock.example",
        transport=httpx.MockTransport(handler),
        headers={"content-type": "application/json"},
    )
    return backend


@pytest.fixture(autouse=True)
def _clear_api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests hermetic — ignore any API keys the dev machine has set."""
    monkeypatch.delenv("SWAY_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


class TestLogprobOf:
    def test_sums_completion_tokens(self) -> None:
        """prompt/completion that split cleanly on a token boundary.

        With ``prompt="The cat"`` + ``completion=" sat."`` the split
        lands after the second token, so the completion's logprobs
        are exactly the last two tokens' values.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            body = _json.loads(request.content)
            assert body["echo"] is True
            assert body["max_tokens"] == 0
            # Echo tokens for "The cat sat." — 4 tokens; first is
            # None (no left context), the rest have logprobs.
            return httpx.Response(
                200,
                json=_echo_response(
                    tokens=["The", " cat", " sat", "."],
                    logprobs=[None, -2.1, -1.7, -0.9],
                ),
            )

        backend = _make_backend(handler)
        # prompt_chars=7 ("The cat"): after "The" total=3, after " cat"
        # total=7 (>=7) → split index 2. Completion = [" sat", "."].
        lp = backend.logprob_of(prompt="The cat", completion=" sat.")
        assert lp == pytest.approx(-1.7 + -0.9)

    def test_mid_token_prompt_leans_conservative(self) -> None:
        """When the prompt boundary falls inside a token, that token
        lands on the prompt side — over-counts the prompt, under-
        attributes to the completion. Documented in ``_split_echo_at_char``.
        """

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=_echo_response(
                    tokens=["The", " cat", " sat", "."],
                    logprobs=[None, -2.1, -1.7, -0.9],
                ),
            )

        backend = _make_backend(handler)
        # prompt_chars=4 ("The ") — boundary is inside " cat".
        # Conservative: " cat" joins the prompt side; completion = [" sat", "."].
        lp = backend.logprob_of(prompt="The ", completion="cat sat.")
        assert lp == pytest.approx(-1.7 + -0.9)

    def test_empty_completion_raises(self) -> None:
        backend = _make_backend(lambda _: httpx.Response(200, json={}))
        with pytest.raises(ProbeError, match="tokenized to zero"):
            backend.logprob_of(prompt="hi", completion="")

    def test_caches_repeated_calls(self) -> None:
        """S07 cache dedupes identical (prompt, completion) pairs."""
        calls = {"count": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            return httpx.Response(
                200,
                json=_echo_response(
                    tokens=["Hi", "!"],
                    logprobs=[None, -3.0],
                ),
            )

        backend = _make_backend(handler)
        backend.logprob_of("Hi", "!")
        backend.logprob_of("Hi", "!")
        assert calls["count"] == 1


class TestRollingLogprob:
    def test_returns_numeric_array(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=_echo_response(
                    tokens=["Hello", " world", "!"],
                    logprobs=[None, -2.5, -1.5],
                ),
            )

        backend = _make_backend(handler)
        r = backend.rolling_logprob("Hello world!")
        assert r.num_tokens == 3
        assert r.logprobs.dtype == np.float32
        # First None is dropped; the rest form the rolling array.
        np.testing.assert_allclose(r.logprobs, [-2.5, -1.5])
        assert r.total_logprob == pytest.approx(-4.0)

    def test_perplexity_finite(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=_echo_response(
                    tokens=["a", "b"],
                    logprobs=[None, -1.0],
                ),
            )

        backend = _make_backend(handler)
        r = backend.rolling_logprob("ab")
        assert r.perplexity > 1.0
        assert np.isfinite(r.perplexity)


class TestNextTokenDist:
    def test_parses_top_logprobs_descending(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = _json.loads(request.content)
            assert body["max_tokens"] == 1
            assert body["echo"] is False
            assert body["logprobs"] >= 1
            return httpx.Response(
                200,
                json=_top_logprobs_response(
                    # Logprobs that sum to < 1 in prob-space so the
                    # residual is a real positive tail mass (~0.22).
                    {" cat": -0.5, " dog": -2.0, " fish": -3.5, " bird": -5.0},
                ),
            )

        backend = _make_backend(handler, vocab_size=50_000)
        d = backend.next_token_dist("The quick brown fox chased the", top_k=4)
        assert d.token_ids.shape == (4,)
        assert d.logprobs.shape == (4,)
        # Sorted descending by logprob.
        assert np.all(np.diff(d.logprobs) <= 0)
        assert d.logprobs[0] == pytest.approx(-0.5)
        # vocab_size threads through; tail_logprob is a finite residual.
        assert d.vocab_size == 50_000
        assert d.tail_logprob is not None
        assert d.tail_logprob < 0

    def test_unknown_vocab_produces_none_tail(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_top_logprobs_response({"a": -0.5, "b": -1.5}))

        backend = _make_backend(handler)  # no vocab_size
        d = backend.next_token_dist("x", top_k=2)
        assert d.tail_logprob is None


class TestPreflight:
    def test_passes_on_finite_dist(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_top_logprobs_response({"ok": -0.1}))

        backend = _make_backend(handler, vocab_size=100)
        ok, reason = backend.preflight_finite_check()
        assert ok, reason

    def test_fails_on_nan_logprob(self) -> None:
        # JSON doesn't spec NaN; httpx's json kwarg rejects it. Hand-
        # craft the body with Python's default json module (which does
        # emit ``NaN``) so the fixture behaves like a lenient upstream.
        body = _json.dumps(_top_logprobs_response({"nope": float("nan")}))

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=body,
                headers={"content-type": "application/json"},
            )

        backend = _make_backend(handler, vocab_size=100)
        ok, reason = backend.preflight_finite_check()
        assert not ok
        assert "non-finite" in reason.lower() or "nan" in reason.lower()

    def test_fails_on_http_error(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="internal server error")

        backend = _make_backend(handler, vocab_size=100)
        ok, reason = backend.preflight_finite_check()
        assert not ok
        assert "500" in reason or "server" in reason.lower() or "error" in reason.lower()


class TestHttpErrorPaths:
    def test_4xx_raises_probe_error(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="bad api key")

        backend = _make_backend(handler)
        with pytest.raises(ProbeError, match="401"):
            backend.logprob_of("hi", "there")

    def test_retry_path_eventually_succeeds(self) -> None:
        """With retries enabled, a 503 followed by a 200 returns 200's body."""
        sequence = iter(
            [
                httpx.Response(503, text="overloaded"),
                httpx.Response(
                    200,
                    json=_echo_response(
                        tokens=["hi", " there"],
                        logprobs=[None, -2.0],
                    ),
                ),
            ]
        )

        def handler(_: httpx.Request) -> httpx.Response:
            return next(sequence)

        backend = ApiScoringBackend(
            base_url="https://mock.example",
            model_name="mock-model",
            api_key=None,
            max_retries=2,
        )
        # Force the client with a MockTransport + short (mocked) wait;
        # tenacity's wait_exponential is fine at this tiny retry count.
        backend._client = httpx.Client(
            base_url="https://mock.example",
            transport=httpx.MockTransport(handler),
        )
        lp = backend.logprob_of("hi", " there")
        assert lp == pytest.approx(-2.0)


class TestSplitEchoAtChar:
    def test_prompt_at_token_boundary(self) -> None:
        assert _split_echo_at_char(["The", " cat"], prompt_chars=3) == 1

    def test_prompt_mid_token_includes_token_in_prompt(self) -> None:
        # "Th" is 2 chars of "The" (3 chars) — prompt extends into the
        # token; conservative: include the whole token on the prompt side.
        assert _split_echo_at_char(["The", " cat"], prompt_chars=2) == 1

    def test_prompt_past_all_tokens(self) -> None:
        assert _split_echo_at_char(["a", "b"], prompt_chars=10) == 2

    def test_zero_prompt(self) -> None:
        assert _split_echo_at_char(["x"], prompt_chars=0) == 0


class TestConstructor:
    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SWAY_API_KEY", "env-token")
        backend = ApiScoringBackend(base_url="http://x", model_name="m", max_retries=0)
        assert backend._api_key == "env-token"

    def test_openai_api_key_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SWAY_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "openai-token")
        backend = ApiScoringBackend(base_url="http://x", model_name="m", max_retries=0)
        assert backend._api_key == "openai-token"

    def test_safe_for_concurrent_views_is_true(self) -> None:
        """Flag is class-level; no instantiation needed."""
        assert ApiScoringBackend.safe_for_concurrent_views is True

    def test_cache_identity_stable(self) -> None:
        backend = ApiScoringBackend(
            base_url="http://host.example:8000",
            model_name="llama-3-8b",
            max_retries=0,
        )
        assert backend.cache_identity() == "api:http://host.example:8000:llama-3-8b"
