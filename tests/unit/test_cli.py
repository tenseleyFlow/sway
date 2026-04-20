"""Smoke tests for the dlm-sway CLI.

We avoid exercising backends (they need real models) and instead test
arg parsing, error paths, and the read-only commands (``doctor``,
``report``, and the help surface).
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dlm_sway.cli.app import app


def test_version_exits_zero() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "dlm-sway" in result.stdout


def test_help_lists_all_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("run", "gate", "check", "diff", "autogen", "doctor", "report"):
        assert cmd in result.stdout


def test_doctor_runs(capsys) -> None:  # type: ignore[no-untyped-def]
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0
    # Rich applies color codes by default; assert the bare product name appears.
    assert "dlm-sway" in result.stdout
    assert "backends" in result.stdout


def test_run_without_file_errors(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    result = CliRunner().invoke(app, ["run", str(missing)])
    # Exit code 2 = SwayError bubble-up; 1 = typer missing-arg; accept either.
    assert result.exit_code != 0


def test_report_from_json(tmp_path: Path) -> None:
    sample = {
        "schema_version": 1,
        "sway_version": "0.1.0.dev0",
        "base_model_id": "base",
        "adapter_id": "adp",
        "score": {"overall": 0.7, "band": "healthy", "components": {}, "findings": []},
        "probes": [
            {
                "name": "p1",
                "kind": "delta_kl",
                "verdict": "pass",
                "score": 0.7,
                "message": "ok",
            },
        ],
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(sample), encoding="utf-8")

    terminal = CliRunner().invoke(app, ["report", str(path)])
    assert terminal.exit_code == 0
    assert "p1" in terminal.stdout

    md = CliRunner().invoke(app, ["report", str(path), "--format", "md"])
    assert md.exit_code == 0
    assert "dlm-sway report" in md.stdout

    junit = CliRunner().invoke(app, ["report", str(path), "--format", "junit"])
    assert junit.exit_code == 0
    assert "<testsuite" in junit.stdout


def test_autogen_without_dlm_extra_exits_nonzero(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Force the import path to fail so the CLI prints the extra hint.
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if name.startswith("dlm_sway.integrations.dlm"):
            raise ImportError("simulated missing extra")
        return real_import(name, *args, **kwargs)  # type: ignore[no-untyped-call]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = CliRunner().invoke(app, ["autogen", "any.dlm"])
    assert result.exit_code != 0
