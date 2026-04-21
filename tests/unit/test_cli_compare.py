"""CLI tests for ``sway compare`` (S11)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dlm_sway.cli.app import app


def _write_run(path: Path, *, timestamp: str, score: float, probe_score: float) -> None:
    """Write a minimal result JSON at ``path``.

    The payload only needs the fields ``report.from_json`` and ``compare``
    actually read — probes (name + score), overall score, timestamp.
    """
    payload = {
        "schema_version": 1,
        "sway_version": "0.1.0.dev0",
        "base_model_id": "base",
        "adapter_id": "adp",
        "started_at": timestamp,
        "finished_at": timestamp,
        "score": {
            "overall": score,
            "band": "healthy",
            "components": {},
            "findings": [],
        },
        "probes": [
            {
                "name": "dk",
                "kind": "delta_kl",
                "verdict": "pass",
                "score": probe_score,
                "message": f"dk score={probe_score}",
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestCompareCli:
    def test_two_files_terminal_default(self, tmp_path: Path) -> None:
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        _write_run(a, timestamp="2026-01-01T12:00:00+00:00", score=0.80, probe_score=0.80)
        _write_run(b, timestamp="2026-01-02T12:00:00+00:00", score=0.82, probe_score=0.82)
        result = CliRunner().invoke(app, ["compare", str(a), str(b)])
        assert result.exit_code == 0, result.stdout
        assert "sway compare" in result.stdout
        assert "dk" in result.stdout
        assert "composite" in result.stdout

    def test_markdown_format(self, tmp_path: Path) -> None:
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        _write_run(a, timestamp="2026-01-01T12:00:00+00:00", score=0.80, probe_score=0.80)
        _write_run(b, timestamp="2026-01-02T12:00:00+00:00", score=0.82, probe_score=0.82)
        result = CliRunner().invoke(app, ["compare", str(a), str(b), "--format", "md"])
        assert result.exit_code == 0, result.stdout
        # Markdown header + table pipes.
        assert "# sway compare" in result.stdout
        assert "| probe |" in result.stdout

    def test_json_format(self, tmp_path: Path) -> None:
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        _write_run(a, timestamp="2026-01-01T12:00:00+00:00", score=0.80, probe_score=0.80)
        _write_run(b, timestamp="2026-01-02T12:00:00+00:00", score=0.82, probe_score=0.82)
        result = CliRunner().invoke(app, ["compare", str(a), str(b), "--format", "json"])
        assert result.exit_code == 0, result.stdout
        parsed = json.loads(result.stdout)
        assert parsed["labels"] == ["a", "b"]
        assert "dk" in parsed["scores"]

    def test_fewer_than_two_files_exits_2(self, tmp_path: Path) -> None:
        a = tmp_path / "a.json"
        _write_run(a, timestamp="2026-01-01T12:00:00+00:00", score=0.80, probe_score=0.80)
        result = CliRunner().invoke(app, ["compare", str(a)])
        assert result.exit_code == 2
        assert "at least two" in result.stderr + result.stdout

    def test_unreadable_file_exits_2(self, tmp_path: Path) -> None:
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        _write_run(a, timestamp="2026-01-01T12:00:00+00:00", score=0.80, probe_score=0.80)
        b.write_text("{ not valid json", encoding="utf-8")
        result = CliRunner().invoke(app, ["compare", str(a), str(b)])
        assert result.exit_code == 2

    def test_fail_on_regression_exits_0_when_improving(self, tmp_path: Path) -> None:
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        _write_run(a, timestamp="2026-01-01T12:00:00+00:00", score=0.80, probe_score=0.80)
        _write_run(b, timestamp="2026-01-02T12:00:00+00:00", score=0.85, probe_score=0.85)
        result = CliRunner().invoke(
            app, ["compare", str(a), str(b), "--fail-on-regression", "0.10"]
        )
        assert result.exit_code == 0, result.stdout

    def test_fail_on_regression_exits_1_when_probe_drops(self, tmp_path: Path) -> None:
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        # dk dropped 0.20 — above the 0.10 threshold.
        _write_run(a, timestamp="2026-01-01T12:00:00+00:00", score=0.80, probe_score=0.80)
        _write_run(b, timestamp="2026-01-02T12:00:00+00:00", score=0.60, probe_score=0.60)
        result = CliRunner().invoke(
            app, ["compare", str(a), str(b), "--fail-on-regression", "0.10"]
        )
        assert result.exit_code == 1

    def test_fail_on_regression_zero_disables_gate(self, tmp_path: Path) -> None:
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        # Drop ≥0.10 but threshold=0 → no gate.
        _write_run(a, timestamp="2026-01-01T12:00:00+00:00", score=0.80, probe_score=0.80)
        _write_run(b, timestamp="2026-01-02T12:00:00+00:00", score=0.50, probe_score=0.50)
        result = CliRunner().invoke(app, ["compare", str(a), str(b), "--fail-on-regression", "0"])
        assert result.exit_code == 0, result.stdout
