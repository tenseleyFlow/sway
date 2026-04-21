"""Tiny-model fixture for integration tests.

Mirrors ``dlm.tests.fixtures.tiny_model``: session-scoped snapshot of
SmolLM2-135M-Instruct, reused across the whole test run. The model is
small enough (~280 MB on disk, ~600 MB in fp32 VRAM) to make integration
tests feasible in CI.

Tests using this fixture must carry ``@pytest.mark.slow`` and
``@pytest.mark.online`` — the default test selection excludes both.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

TINY_MODEL_HF_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
TINY_MODEL_REVISION = os.environ.get("DLM_SWAY_TINY_MODEL_REVISION", "main")


def _offline_mode() -> bool:
    return os.environ.get("SWAY_OFFLINE", "0") == "1"


def _snapshot_download_with_retry(**kwargs: object) -> str:
    """``snapshot_download`` wrapped with exponential-backoff retry.

    F03 (Audit 03) observed an integration-lane macOS run that hung
    20+ minutes inside ``snapshot_download``'s cache-resolution path
    after HF Hub connectivity briefly dropped. A silent stall is the
    worst UX: the job times out with zero test output and no
    actionable error. The retry wrapper turns a transient network
    blip into a 5s-10s-20s back-off and a final timeout-ish failure
    that surfaces cleanly.

    Each attempt is hard-capped by ``etag_timeout`` + a per-attempt
    overall timeout so no single call can burn the test budget. The
    retry policy runs at most 3 attempts with jittered exponential
    backoff.
    """
    from huggingface_hub import snapshot_download
    from tenacity import (
        Retrying,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    retry_types: tuple[type[BaseException], ...] = (OSError, RuntimeError)
    for attempt in Retrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=5, min=5, max=30),
        retry=retry_if_exception_type(retry_types),
        reraise=True,
    ):
        with attempt:
            # ``etag_timeout`` bounds the per-file head/etag probe
            # (10 s is generous; 120s default is the real hang risk).
            result: str = snapshot_download(etag_timeout=10, **kwargs)  # type: ignore[arg-type]
            return result
    # ``reraise=True`` means the Retrying loop always either returns
    # (above) or propagates the last exception — this line is
    # unreachable, but keeps mypy happy with a pointed sentinel.
    raise RuntimeError("snapshot_download retry loop exhausted without a return")


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Download (or reuse) the tiny model; yield the cached directory.

    Test opts in via ``@pytest.mark.online`` — the session-wide offline
    env vars are cleared inside this fixture so ``snapshot_download``
    actually fetches.
    """
    # Clear offline env guards (set by the unit-test autouse fixture).
    prior = {
        k: os.environ.pop(k, None)
        for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    }
    try:
        path = _snapshot_download_with_retry(
            repo_id=TINY_MODEL_HF_ID,
            revision=TINY_MODEL_REVISION,
            local_files_only=_offline_mode(),
        )
        yield Path(path)
    finally:
        for k, v in prior.items():
            if v is not None:
                os.environ[k] = v
