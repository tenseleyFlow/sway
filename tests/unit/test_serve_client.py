"""Unit tests for :class:`dlm_sway.serve.client.ServeClient`.

The happy path uses an httpx ``MockTransport`` to canned-respond on
each route so we exercise the client's request-shaping (URL build,
auth header) and response-parsing without binding a port. The error
branches drive :meth:`ServeClient._parse_response` directly so we
cover transport failures and malformed payloads in isolation.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("httpx")

import httpx  # noqa: E402

from dlm_sway.serve.client import ServeClient, ServeClientError  # noqa: E402
from dlm_sway.suite.spec import SwaySpec  # noqa: E402


def _spec_payload() -> dict[str, Any]:
    return {
        "version": 1,
        "models": {
            "base": {"kind": "dummy", "base": "dummy-base"},
            "ft": {"kind": "dummy", "base": "dummy-base"},
        },
        "defaults": {"seed": 0, "differential": True},
        "suite": [
            {"name": "dk", "kind": "delta_kl", "prompts": ["hello"]},
        ],
    }


class _StubClient(ServeClient):
    """ServeClient overridden to route through an httpx MockTransport
    instead of opening a real connection — same code path otherwise."""

    def __init__(
        self,
        handler: Any,
        *,
        api_key: str | None = None,
    ) -> None:
        super().__init__("http://testserver", api_key=api_key)
        self._transport = httpx.MockTransport(handler)

    def _get(self, path: str) -> dict[str, Any]:
        with httpx.Client(transport=self._transport, base_url=self._url) as client:
            resp = client.get(path, headers=self._headers)
        return self._parse_response(resp, path=path)

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(transport=self._transport, base_url=self._url) as client:
            resp = client.post(path, headers=self._headers, json=body)
        return self._parse_response(resp, path=path)


class TestServeClientHappy:
    def test_health_returns_dict(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            return httpx.Response(200, json={"status": "ok", "uptime_seconds": 1.0})

        client = _StubClient(handler)
        body = client.health()
        assert body["status"] == "ok"
        assert seen["path"] == "/health"

    def test_run_sends_spec_and_path(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"probes": [], "request_seconds": 0.1})

        client = _StubClient(handler)
        spec = SwaySpec.model_validate(_spec_payload())
        body = client.run(spec, spec_path="custom/path.yaml")
        assert "probes" in body
        assert captured["body"]["spec_path"] == "custom/path.yaml"
        assert captured["body"]["spec"]["version"] == 1

    def test_score_passes_probe_names(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"probes": [], "request_seconds": 0.1})

        client = _StubClient(handler)
        spec = SwaySpec.model_validate(_spec_payload())
        client.score(spec, probe_names=["dk"])
        assert captured["body"]["probe_names"] == ["dk"]

    def test_score_omits_probe_names_when_none(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"probes": [], "request_seconds": 0.1})

        client = _StubClient(handler)
        spec = SwaySpec.model_validate(_spec_payload())
        client.score(spec)
        # Default behavior: omit the field rather than send null.
        assert "probe_names" not in captured["body"]


class TestServeClientAuth:
    def test_run_attaches_bearer_when_api_key_set(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["auth"] = request.headers.get("Authorization")
            return httpx.Response(200, json={"probes": []})

        client = _StubClient(handler, api_key="abc")
        spec = SwaySpec.model_validate(_spec_payload())
        client.run(spec)
        assert captured["auth"] == "Bearer abc"

    def test_no_auth_header_when_api_key_unset(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["auth"] = request.headers.get("Authorization")
            return httpx.Response(200, json={"status": "ok"})

        client = _StubClient(handler)
        client.health()
        assert captured["auth"] is None

    def test_401_raises_serve_client_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            del request
            return httpx.Response(401, json={"detail": "missing or invalid"})

        client = _StubClient(handler)
        spec = SwaySpec.model_validate(_spec_payload())
        with pytest.raises(ServeClientError, match="missing or invalid"):
            client.run(spec)


class TestServeClientErrorPaths:
    def test_missing_httpx_dependency_raises_clean_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If httpx isn't installed, the client surfaces a SwayError-shaped
        message pointing at the [serve] extra."""
        client = ServeClient("http://nope")
        # Force the httpx import inside _get to fail.
        import builtins

        real_import = builtins.__import__

        def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "httpx":
                raise ImportError("nope")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        with pytest.raises(ServeClientError, match=r"\[serve\]"):
            client.health()

    def test_transport_failure_wraps_to_serve_client_error(self) -> None:
        """An httpx connection error becomes a ServeClientError with the
        request method/path in the message."""
        client = ServeClient("http://127.0.0.1:1")  # unused port
        with pytest.raises(ServeClientError, match="GET /health failed"):
            client.health()

    def test_non_dict_response_raises(self) -> None:
        """When the daemon ever returns a JSON list at the root, the
        client refuses it instead of silently mis-typing."""

        def _handler(request: httpx.Request) -> httpx.Response:
            del request
            return httpx.Response(200, json=[1, 2, 3])

        # Drive _parse_response directly: build a fake response object.
        resp = httpx.Response(200, json=[1, 2, 3])
        with pytest.raises(ServeClientError, match="non-object JSON"):
            ServeClient._parse_response(resp, path="/x")

    def test_4xx_pulls_detail_from_payload(self) -> None:
        resp = httpx.Response(400, json={"detail": "bad spec"})
        with pytest.raises(ServeClientError, match="bad spec"):
            ServeClient._parse_response(resp, path="/run")

    def test_4xx_with_non_json_falls_back_to_text(self) -> None:
        resp = httpx.Response(500, text="upstream broke")
        with pytest.raises(ServeClientError, match="upstream broke"):
            ServeClient._parse_response(resp, path="/run")

    def test_url_property_strips_trailing_slash(self) -> None:
        client = ServeClient("http://localhost:8787/")
        assert client.url == "http://localhost:8787"
