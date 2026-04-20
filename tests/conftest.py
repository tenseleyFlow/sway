"""Shared test fixtures.

Keep the default fast-test environment offline and deterministic so unit
tests stay below ~1 s per file. Integration tests override these via
their own ``conftest`` when they need network access.
"""

from __future__ import annotations

import pytest

# Import the probes package once so every shipped probe registers itself
# with the central registry. Tests that exercise build_probe("delta_kl",
# …) rely on this.
import dlm_sway.probes  # noqa: F401


@pytest.fixture(autouse=True)
def _offline_and_no_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests never touch the network.

    Any backend test that needs HF should be marked ``@pytest.mark.online``
    and clear these vars explicitly.
    """
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
