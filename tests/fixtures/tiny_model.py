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


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Download (or reuse) the tiny model; yield the cached directory.

    Test opts in via ``@pytest.mark.online`` — the session-wide offline
    env vars are cleared inside this fixture so ``snapshot_download``
    actually fetches.
    """
    from huggingface_hub import snapshot_download

    # Clear offline env guards (set by the unit-test autouse fixture).
    prior = {
        k: os.environ.pop(k, None)
        for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    }
    try:
        path = snapshot_download(
            repo_id=TINY_MODEL_HF_ID,
            revision=TINY_MODEL_REVISION,
            local_files_only=_offline_mode(),
        )
        yield Path(path)
    finally:
        for k, v in prior.items():
            if v is not None:
                os.environ[k] = v
