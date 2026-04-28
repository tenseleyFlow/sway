"""Unit tests for :mod:`dlm_sway.serve.app`.

These exercise the FastAPI surface end-to-end via
``fastapi.testclient.TestClient`` — no uvicorn, no real network — and
ride a pre-seeded :class:`BackendCache` so the dummy backend never
goes through the rejected build path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

# Skip the whole module cleanly if the [serve] extra isn't installed —
# avoids forcing every test contributor to ``pip install fastapi``.
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from dlm_sway.backends.dummy import (  # noqa: E402
    DummyDifferentialBackend,
    DummyResponses,
)
from dlm_sway.core.model import ModelSpec  # noqa: E402
from dlm_sway.serve.app import create_app, parse_host_port  # noqa: E402
from dlm_sway.serve.cache import BackendCache, CachedBackend, cache_key_for  # noqa: E402
from dlm_sway.suite.spec import SwaySpec  # noqa: E402


def _spec_payload(*, base: str = "dummy-base") -> dict[str, Any]:
    """A minimal valid SwaySpec dict with one delta_kl probe."""
    return {
        "version": 1,
        "models": {
            "base": {"kind": "dummy", "base": base},
            "ft": {"kind": "dummy", "base": base},
        },
        "defaults": {"seed": 0, "differential": True},
        "suite": [
            {"name": "dk", "kind": "delta_kl", "prompts": ["hello world"]},
        ],
    }


def _seed_dummy(cache: BackendCache, model_spec: ModelSpec) -> DummyDifferentialBackend:
    """Pre-load a dummy backend into the cache under ``model_spec``'s key."""
    backend = DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
    key = cache_key_for(model_spec)
    entry = CachedBackend(key=key, backend=backend, model_spec=model_spec, load_seconds=0.1)
    with cache._lock:  # noqa: SLF001
        cache._entries[key] = entry  # noqa: SLF001
    return backend


def _make_seeded_app(*, api_key: str | None = None) -> tuple[Any, BackendCache]:
    """Build the app and pre-seed its cache with a dummy backend
    under the spec the tests POST."""
    cache = BackendCache(max_size=2)
    app = create_app(cache=cache, api_key=api_key)
    payload = _spec_payload()
    spec = SwaySpec.model_validate(payload)
    _seed_dummy(cache, spec.models.ft)
    return app, cache


class TestHealth:
    def test_health_returns_uptime_and_loaded_models(self) -> None:
        app, _cache = _make_seeded_app()
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["uptime_seconds"] >= 0.0
        assert body["max_loaded_models"] == 2
        assert isinstance(body["loaded_models"], list)
        assert any(m["base"] == "dummy-base" for m in body["loaded_models"])

    def test_health_unauthenticated_when_api_key_set(self) -> None:
        """A k8s liveness probe must be able to hit /health without
        the bearer token."""
        app, _cache = _make_seeded_app(api_key="secret")
        with TestClient(app) as client:
            resp = client.get("/health")  # No Authorization header.
        assert resp.status_code == 200


class TestStats:
    def test_stats_zero_when_no_runs(self) -> None:
        app, _cache = _make_seeded_app()
        with TestClient(app) as client:
            resp = client.get("/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["request_count"] == 0
        assert body["mean_run_seconds"] is None
        assert body["cached_backends"] == 1


class TestRun:
    def test_run_returns_full_report_shape(self) -> None:
        app, _cache = _make_seeded_app()
        with TestClient(app) as client:
            resp = client.post("/run", json={"spec": _spec_payload()})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        # Round-trips the same JSON shape sway run --json-out emits.
        assert "score" in body
        assert "probes" in body
        assert isinstance(body["probes"], list)
        assert any(p["name"] == "dk" for p in body["probes"])
        assert "request_seconds" in body
        assert body["request_seconds"] >= 0.0

    def test_run_increments_stats(self) -> None:
        app, _cache = _make_seeded_app()
        with TestClient(app) as client:
            client.post("/run", json={"spec": _spec_payload()})
            client.post("/run", json={"spec": _spec_payload()})
            stats = client.get("/stats").json()
        assert stats["request_count"] == 2
        assert stats["mean_run_seconds"] is not None
        assert stats["mean_run_seconds"] >= 0.0

    def test_run_400_on_invalid_spec(self) -> None:
        app, _cache = _make_seeded_app()
        with TestClient(app) as client:
            # Missing required fields → pydantic validation error → 422.
            resp = client.post("/run", json={"spec": {"version": 1}})
        # FastAPI emits 422 for body validation failures.
        assert resp.status_code in (400, 422)


class TestScore:
    def test_score_filters_by_probe_names(self) -> None:
        app, _cache = _make_seeded_app()
        # Build a spec with two probes, then ask /score for one.
        spec = _spec_payload()
        spec["suite"].append({"name": "dk2", "kind": "delta_kl", "prompts": ["different"]})
        with TestClient(app) as client:
            resp = client.post(
                "/score",
                json={"spec": spec, "probe_names": ["dk2"]},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        names = {p["name"] for p in body["probes"]}
        assert names == {"dk2"}

    def test_score_runs_all_when_probe_names_none(self) -> None:
        app, _cache = _make_seeded_app()
        spec = _spec_payload()
        spec["suite"].append({"name": "dk2", "kind": "delta_kl", "prompts": ["other"]})
        with TestClient(app) as client:
            resp = client.post("/score", json={"spec": spec})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        names = {p["name"] for p in body["probes"]}
        assert names == {"dk", "dk2"}


class TestAuth:
    def test_run_requires_bearer_when_api_key_set(self) -> None:
        app, _cache = _make_seeded_app(api_key="topsecret")
        with TestClient(app) as client:
            resp = client.post("/run", json={"spec": _spec_payload()})
        assert resp.status_code == 401
        assert "Authorization" in resp.json()["detail"]

    def test_run_succeeds_with_valid_bearer(self) -> None:
        app, _cache = _make_seeded_app(api_key="topsecret")
        with TestClient(app) as client:
            resp = client.post(
                "/run",
                json={"spec": _spec_payload()},
                headers={"Authorization": "Bearer topsecret"},
            )
        assert resp.status_code == 200, resp.text

    def test_run_rejects_wrong_bearer(self) -> None:
        app, _cache = _make_seeded_app(api_key="topsecret")
        with TestClient(app) as client:
            resp = client.post(
                "/run",
                json={"spec": _spec_payload()},
                headers={"Authorization": "Bearer otherkey"},
            )
        assert resp.status_code == 401


class TestCacheRoundTripThroughApp:
    def test_run_does_not_evict_below_cap(self) -> None:
        app, cache = _make_seeded_app()
        with TestClient(app) as client:
            client.post("/run", json={"spec": _spec_payload()})
            # Cap is 2; we only ever loaded 1 entry, so it must still
            # be there. Check inside the with-block — TestClient's
            # __exit__ runs the shutdown handler which evicts everything.
            assert len(cache.loaded_keys()) == 1
        # After the lifespan exits, evict_all() has run.
        assert len(cache.loaded_keys()) == 0


class TestParseHostPort:
    def test_rejects_out_of_range_port(self) -> None:
        import typer

        with pytest.raises(typer.BadParameter):
            parse_host_port("127.0.0.1", 0)
        with pytest.raises(typer.BadParameter):
            parse_host_port("127.0.0.1", 70_000)

    def test_accepts_valid_host_port(self) -> None:
        host, port = parse_host_port("127.0.0.1", 8787)
        assert host == "127.0.0.1"
        assert port == 8787


class TestSpecRoundTrip:
    def test_spec_dump_load_preserves_fields(self, tmp_path: Path) -> None:
        """RunRequest must roundtrip a SwaySpec through JSON without
        losing fields — risk #4 in the sprint plan."""
        from dlm_sway.serve.app import RunRequest

        original = SwaySpec.model_validate(_spec_payload())
        # Round-trip through JSON like FastAPI's wire format does.
        req = RunRequest(spec=original)
        body = req.model_dump_json()
        reparsed = RunRequest.model_validate_json(body)
        assert reparsed.spec == original
        assert reparsed.spec_path == "<serve>"

    def test_spec_with_dlm_source_roundtrip(self) -> None:
        """dlm_source is optional and must survive round-trip."""
        from dlm_sway.serve.app import RunRequest

        payload = _spec_payload()
        payload["dlm_source"] = "/path/to/foo.dlm"
        original = SwaySpec.model_validate(payload)
        req = RunRequest(spec=original)
        reparsed = RunRequest.model_validate_json(req.model_dump_json())
        assert reparsed.spec.dlm_source == "/path/to/foo.dlm"
