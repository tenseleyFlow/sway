"""CLI tests for ``sway serve``.

These cover argument validation only — actually running uvicorn would
bind a port. The 0.0.0.0-without-api-key refusal is the load-bearing
check (sprint plan risk #1: anyone-on-LAN-drives-your-GPU).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from typer.testing import CliRunner  # noqa: E402

from dlm_sway.cli.app import app  # noqa: E402


def test_serve_in_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "serve" in result.stdout


def test_serve_refuses_public_bind_without_api_key() -> None:
    result = CliRunner().invoke(app, ["serve", "--host", "0.0.0.0", "--port", "8787"])
    assert result.exit_code == 2
    # Mix of stdout/stderr depending on typer version; check both.
    combined = (result.stdout or "") + (result.stderr or "")
    assert "--api-key" in combined
    assert "0.0.0.0" in combined


def test_serve_rejects_zero_max_loaded_models() -> None:
    result = CliRunner().invoke(app, ["serve", "--max-loaded-models", "0"])
    assert result.exit_code == 2


def test_serve_rejects_invalid_port() -> None:
    result = CliRunner().invoke(app, ["serve", "--port", "0"])
    assert result.exit_code != 0
