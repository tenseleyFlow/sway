"""FastAPI app for the ``sway serve`` daemon.

Three concerns sit in this module:

1. ``create_app(...)`` — factory that returns a FastAPI instance with
   the four documented endpoints wired up. Used by the CLI's uvicorn
   launcher and unit tests' :class:`fastapi.testclient.TestClient`.
2. Pydantic request/response models for ``/run`` and ``/score``.
3. The endpoint handlers themselves, which delegate to
   :func:`dlm_sway.suite.runner.run` against backends fetched from
   :class:`dlm_sway.serve.cache.BackendCache`.

Auth is intentionally minimal in v1: localhost-only by default; binding
to ``0.0.0.0`` is rejected at the CLI layer unless ``--api-key`` is
set, and an API-key middleware validates the bearer token on every
request when an API key was provided. Full OAuth is out of scope.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import typer
from pydantic import BaseModel

from dlm_sway import __version__
from dlm_sway.core.errors import SwayError
from dlm_sway.serve.cache import BackendCache
from dlm_sway.suite.spec import SwaySpec

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    """Body of ``POST /run`` — full spec plus optional overrides."""

    spec: SwaySpec
    spec_path: str = "<serve>"
    seed: int | None = None


class ScoreRequest(BaseModel):
    """Body of ``POST /score`` — full spec plus a subset filter.

    When ``probe_names`` is set, only those probes execute; the
    response shape is the per-probe entry list (no folded SwayScore).
    """

    spec: SwaySpec
    probe_names: list[str] | None = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    *,
    cache: BackendCache | None = None,
    api_key: str | None = None,
) -> Any:
    """Build a FastAPI app wired to a :class:`BackendCache`.

    Parameters
    ----------
    cache:
        Optional pre-built cache. When omitted, a new cache with
        ``max_size=2`` is created. Tests inject a dummy-backed cache
        via this parameter; the CLI passes one with the user's
        ``--max-loaded-models``.
    api_key:
        Optional bearer token. When set, every request must carry
        ``Authorization: Bearer <key>`` or it gets a 401. When
        ``None`` (the default), no auth — only safe for localhost
        binds, which the CLI enforces.

    Returns
    -------
    A :class:`fastapi.FastAPI` instance ready to hand to uvicorn or
    a TestClient.
    """
    try:
        from contextlib import asynccontextmanager

        from fastapi import FastAPI, HTTPException, Request, status
        from fastapi.responses import JSONResponse
    except ImportError as exc:
        raise SwayError(
            "sway serve requires the [serve] extra: pip install 'dlm-sway[serve]'"
        ) from exc

    if cache is None:
        cache = BackendCache(max_size=2)
    started_at = time.monotonic()
    request_count = 0
    total_run_seconds = 0.0

    # Capture cache in the lifespan closure so the shutdown leg runs
    # ``evict_all`` (the on_event API is deprecated in FastAPI 0.110+).
    _cache_for_lifespan = cache

    @asynccontextmanager
    async def _lifespan(app: Any) -> Any:
        del app
        try:
            yield
        finally:
            _cache_for_lifespan.evict_all()

    app = FastAPI(
        title="sway",
        version=__version__,
        description=(
            "Warm-backend HTTP API for sway. POST /run accepts a spec; "
            "the daemon keeps backends loaded between calls."
        ),
        lifespan=_lifespan,
    )

    # -- auth middleware --------------------------------------------------

    if api_key is not None:
        expected_header = f"Bearer {api_key}"

        @app.middleware("http")
        async def _auth_middleware(
            request: Request,
            call_next: Callable[[Request], Awaitable[Any]],
        ) -> Any:
            # Health is unauthenticated so an external probe (e.g. a
            # k8s liveness check) can hit it without distributing the
            # token. Every other endpoint requires the bearer.
            if request.url.path == "/health":
                return await call_next(request)
            got = request.headers.get("Authorization", "")
            if got != expected_header:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "missing or invalid Authorization header"},
                )
            return await call_next(request)

    # -- helpers ---------------------------------------------------------

    def _run_with_warm_backend(spec: SwaySpec, *, spec_path: str) -> Any:
        """Common path for /run and /score: pick the right backend
        from cache, call the runner, return a serializable result."""
        from dlm_sway.suite.runner import run as run_suite

        # Same logic as cli/_execute_spec but takes an in-memory spec
        # instead of a path, and uses the cache for backend identity.
        if spec.defaults.differential:
            cached = cache.get_or_load(spec.models.ft)
        else:
            # Two-separate (base + ft as distinct backends) isn't yet
            # cached at this layer — falls through to a fresh build
            # per request. Cleanest path until a multi-key cache
            # entry shape lands.
            from dlm_sway.backends import build_two_separate

            backend = build_two_separate(spec.models)
            try:
                return run_suite(spec, backend, spec_path=spec_path)
            finally:
                close = getattr(backend, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:  # noqa: BLE001
                        _LOG.warning("two-separate backend close raised: %s", exc)
        return run_suite(spec, cached.backend, spec_path=spec_path)

    # -- /health ---------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "sway_version": __version__,
            "uptime_seconds": time.monotonic() - started_at,
            "loaded_models": [
                {
                    "kind": s.kind,
                    "base": s.base,
                    "adapter": str(s.adapter) if s.adapter is not None else None,
                    "dtype": s.dtype,
                    "device": s.device,
                }
                for s in cache.loaded_specs()
            ],
            "max_loaded_models": cache.max_size,
        }

    # -- /stats ----------------------------------------------------------

    @app.get("/stats")
    async def stats() -> dict[str, Any]:
        return {
            "uptime_seconds": time.monotonic() - started_at,
            "request_count": request_count,
            "total_run_seconds": total_run_seconds,
            "mean_run_seconds": (total_run_seconds / request_count if request_count > 0 else None),
            "cached_backends": len(cache.loaded_keys()),
            "max_loaded_models": cache.max_size,
        }

    # -- /run ------------------------------------------------------------

    @app.post("/run")
    async def run_endpoint(req: RunRequest) -> dict[str, Any]:
        nonlocal request_count, total_run_seconds
        started = time.monotonic()
        try:
            result = _run_with_warm_backend(req.spec, spec_path=req.spec_path)
        except SwayError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        elapsed = time.monotonic() - started
        request_count += 1
        total_run_seconds += elapsed
        # Reuse the on-disk-JSON serializer so the daemon's response
        # shape matches what ``sway run --json-out`` would write — one
        # less JSON contract to maintain.
        import json as _json

        from dlm_sway.suite import report
        from dlm_sway.suite.score import compute as compute_score

        score_obj = compute_score(result)
        bundled: dict[str, Any] = _json.loads(report.to_json(result, score_obj))
        bundled["request_seconds"] = elapsed
        return bundled

    # -- /score ----------------------------------------------------------

    @app.post("/score")
    async def score_endpoint(req: ScoreRequest) -> dict[str, Any]:
        nonlocal request_count, total_run_seconds
        # /score is a thin filter over /run: same code path, but only
        # the named probes execute. Filtering happens at the spec
        # level (we mutate a copy of spec.suite) so the runner's
        # null-calibration logic still sees the right downstream
        # kinds. Frozen pydantic models require .model_copy.
        if req.probe_names is not None:
            wanted = set(req.probe_names)
            filtered_suite = [p for p in req.spec.suite if p.get("name") in wanted]
            spec = req.spec.model_copy(update={"suite": filtered_suite})
        else:
            spec = req.spec
        started = time.monotonic()
        try:
            result = _run_with_warm_backend(spec, spec_path="<serve:/score>")
        except SwayError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        elapsed = time.monotonic() - started
        request_count += 1
        total_run_seconds += elapsed
        # /score returns just the probe entries — no folded SwayScore.
        # Reuse the same per-probe serializer the to_json path uses
        # for consistency.
        import json as _json

        from dlm_sway.suite import report
        from dlm_sway.suite.score import compute as compute_score

        score_obj = compute_score(result)
        bundled: dict[str, Any] = _json.loads(report.to_json(result, score_obj))
        return {
            "probes": bundled["probes"],
            "request_seconds": elapsed,
        }

    # Stash the cache + counters on the app so tests can introspect.
    app.state.sway_cache = cache

    return app


def parse_host_port(host: str, port: int) -> tuple[str, int]:
    """Validate host + port at CLI layer; raise via typer for clean exit.

    Centralizes the "0.0.0.0 without auth = refuse" rule so the CLI
    and any test harness share it.
    """
    if not (1 <= port <= 65535):
        raise typer.BadParameter(f"port must be 1..65535, got {port}")
    return host, port
