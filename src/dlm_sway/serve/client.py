"""Python client for the ``sway serve`` daemon.

Surface designed for two use cases:

1. **Notebook / REPL** — ``ServeClient(url).run(spec)`` returns the
   parsed report dict; users get the warm-backend speedup with a
   one-liner.
2. **CLI delegation** — ``sway watch --serve-url`` (S34) and any
   future "client mode" CLI can build a client and forward requests.

Lives in the ``[serve]`` extra (httpx is the dep). Returns the same
JSON shape as ``sway run --json-out`` so notebook code can swap
``sway.run(...)`` for ``ServeClient(url).run(...)`` with no other
changes.
"""

from __future__ import annotations

from typing import Any

from dlm_sway.core.errors import SwayError
from dlm_sway.suite.spec import SwaySpec


class ServeClientError(SwayError):
    """Raised when the daemon returned an error or was unreachable.

    Distinct from generic httpx exceptions so callers can catch a
    sway-shaped exception family. Wraps the underlying httpx error
    via ``__cause__``.
    """


class ServeClient:
    """HTTP client for ``sway serve``.

    Parameters
    ----------
    url:
        Base URL of the daemon, e.g. ``http://localhost:8787``.
    timeout:
        Per-request timeout in seconds. Default 120s — model loads
        on a cold cache can take 30-60s; the floor is generous.
    api_key:
        Bearer token, if the daemon was launched with ``--api-key``.

    The client is **stateless** — no persistent connection pool is
    held across calls. For high-throughput callers (notebooks doing
    many quick scores) the per-call connection cost is dwarfed by
    the inference cost; for low-throughput callers, the simpler
    contract is worth more than the saved milliseconds.
    """

    def __init__(self, url: str, *, timeout: float = 120.0, api_key: str | None = None) -> None:
        self._url = url.rstrip("/")
        self._timeout = float(timeout)
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key is not None:
            self._headers["Authorization"] = f"Bearer {api_key}"

    @property
    def url(self) -> str:
        return self._url

    def health(self) -> dict[str, Any]:
        """Hit ``GET /health``. Returns the daemon's health payload."""
        return self._get("/health")

    def stats(self) -> dict[str, Any]:
        """Hit ``GET /stats``. Returns request count / mean latency."""
        return self._get("/stats")

    def run(self, spec: SwaySpec, *, spec_path: str = "<client>") -> dict[str, Any]:
        """Hit ``POST /run`` with the spec.

        Returns the parsed JSON response — same shape as the on-disk
        report ``sway run --json-out`` would write, plus a
        ``request_seconds`` field with the daemon's measured execution
        time (excluding HTTP round-trip).
        """
        body = {"spec": spec.model_dump(mode="json"), "spec_path": spec_path}
        return self._post("/run", body)

    def score(
        self,
        spec: SwaySpec,
        *,
        probe_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Hit ``POST /score``. Same as ``run`` but returns only the
        probe-level results (no folded SwayScore) and supports a
        ``probe_names`` filter for partial runs."""
        body: dict[str, Any] = {"spec": spec.model_dump(mode="json")}
        if probe_names is not None:
            body["probe_names"] = list(probe_names)
        return self._post("/score", body)

    # -- internals -------------------------------------------------------

    def _get(self, path: str) -> dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:
            raise ServeClientError(
                "ServeClient requires the [serve] extra: pip install 'dlm-sway[serve]'"
            ) from exc
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(self._url + path, headers=self._headers)
        except httpx.HTTPError as exc:
            raise ServeClientError(f"GET {path} failed: {exc}") from exc
        return self._parse_response(resp, path=path)

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:
            raise ServeClientError(
                "ServeClient requires the [serve] extra: pip install 'dlm-sway[serve]'"
            ) from exc
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(self._url + path, headers=self._headers, json=body)
        except httpx.HTTPError as exc:
            raise ServeClientError(f"POST {path} failed: {exc}") from exc
        return self._parse_response(resp, path=path)

    @staticmethod
    def _parse_response(resp: Any, *, path: str) -> dict[str, Any]:
        # Trust httpx's status code; reach into ``.detail`` first since
        # FastAPI's HTTPException uses that key by default.
        if resp.status_code >= 400:
            try:
                payload = resp.json()
            except Exception:  # noqa: BLE001
                payload = {"detail": resp.text}
            detail = payload.get("detail") if isinstance(payload, dict) else str(payload)
            raise ServeClientError(f"{path} returned {resp.status_code}: {detail}")
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise ServeClientError(f"{path} returned non-JSON body: {resp.text[:200]}") from exc
        if not isinstance(data, dict):
            raise ServeClientError(f"{path} returned non-object JSON: {type(data).__name__}")
        return data
