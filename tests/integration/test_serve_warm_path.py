"""Integration test: warm-backend speedup is real.

Sprint 36 prove-the-value: first ``POST /run`` cold-loads the HF
backend (~tens of seconds even with the tiny model), and subsequent
runs reuse the cached backend. The exact speedup depends on the
host (cache state, disk speed, CPU vs GPU); we assert the *shape*
of the curve — call N+1 should be substantially faster than call 1
once the load is amortized.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("torch")

from fastapi.testclient import TestClient  # noqa: E402

from dlm_sway.serve.app import create_app  # noqa: E402
from dlm_sway.serve.cache import BackendCache  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.online]


def _spec_payload(tiny_model_dir: Path) -> dict[str, object]:
    """A backend-touching delta_kl spec against the tiny model."""
    return {
        "version": 1,
        "models": {
            "base": {
                "kind": "hf",
                "base": str(tiny_model_dir),
                "dtype": "fp32",
                "device": "cpu",
            },
            "ft": {
                "kind": "hf",
                "base": str(tiny_model_dir),
                "dtype": "fp32",
                "device": "cpu",
            },
        },
        "defaults": {"seed": 0, "differential": True},
        "suite": [
            {"name": "dk", "kind": "delta_kl", "prompts": ["hello"]},
        ],
    }


def test_warm_backend_speedup(tiny_model_dir: Path) -> None:
    """First call pays the load; runs 2 and 3 reuse the cache.

    Assertion is intentionally loose — we want to catch the case
    where the cache silently misses (every call cold-loads), not
    pin a specific ratio that varies with host hardware.
    """
    cache = BackendCache(max_size=1)
    app = create_app(cache=cache)
    payload = _spec_payload(tiny_model_dir)

    with TestClient(app) as client:
        first = client.post("/run", json={"spec": payload})
        assert first.status_code == 200, first.text
        first_body = first.json()
        first_seconds = first_body["request_seconds"]

        second = client.post("/run", json={"spec": payload})
        assert second.status_code == 200, second.text
        second_seconds = second.json()["request_seconds"]

        third = client.post("/run", json={"spec": payload})
        assert third.status_code == 200, third.text
        third_seconds = third.json()["request_seconds"]

        # The cache must report exactly one loaded backend after three
        # calls against the same spec — confirms cache hit, not miss.
        health = client.get("/health").json()
        assert len(health["loaded_models"]) == 1

    # Warm calls should be at least 30% faster than the first. On a
    # tiny model + cold disk, first ≈ 5-15s and warm ≈ 1-3s; the
    # margin handles flaky timing without false-passing a regression
    # to "every call rebuilds the backend" (which would show
    # warm ≈ first within the same order of magnitude).
    assert second_seconds < first_seconds * 0.7, (
        f"warm path not detected: first={first_seconds:.2f}s "
        f"second={second_seconds:.2f}s third={third_seconds:.2f}s"
    )
    assert third_seconds < first_seconds * 0.7
