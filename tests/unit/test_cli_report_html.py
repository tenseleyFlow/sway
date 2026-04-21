"""CLI tests for ``sway report --format html`` (S12)."""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dlm_sway.cli.app import app

# Plotly is shipped via the optional [viz] extra. The
# ``test_missing_plotly_surfaces_install_hint`` test monkeypatches it
# away, so it runs unconditionally; every other test needs plotly loaded.
pytest.importorskip("plotly")


def _write_sample_result(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "sway_version": "0.1.0.dev0",
        "base_model_id": "base",
        "adapter_id": "adp",
        "started_at": "2026-01-01T12:00:00+00:00",
        "finished_at": "2026-01-01T12:00:02+00:00",
        "score": {
            "overall": 0.77,
            "band": "healthy",
            "components": {
                "adherence": 0.8,
                "attribution": 0.7,
                "calibration": 0.8,
                "ablation": 0.7,
            },
            "weights": {
                "adherence": 0.30,
                "attribution": 0.35,
                "calibration": 0.20,
                "ablation": 0.15,
            },
            "findings": [],
        },
        "probes": [
            {
                "name": "dk",
                "kind": "delta_kl",
                "verdict": "pass",
                "score": 0.8,
                "raw": 0.4,
                "z_score": 4.0,
                "message": "ok",
            },
            {
                "name": "abl",
                "kind": "adapter_ablation",
                "verdict": "pass",
                "score": 0.7,
                "raw": 0.9,
                "message": "R²=0.9",
                "evidence": {
                    "lambdas": [0.0, 0.5, 1.0],
                    "mean_divergence_per_lambda": [0.0, 0.1, 0.2],
                    "saturation_lambda": 0.75,
                },
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestReportHtmlCli:
    def test_writes_file(self, tmp_path: Path) -> None:
        result_json = tmp_path / "result.json"
        out_html = tmp_path / "report.html"
        _write_sample_result(result_json)
        invocation = CliRunner().invoke(
            app, ["report", str(result_json), "--format", "html", "--out", str(out_html)]
        )
        assert invocation.exit_code == 0, invocation.stdout + invocation.stderr
        assert out_html.is_file()
        content = out_html.read_text(encoding="utf-8")
        # Structural markers — the wrapper + panel div ids + probe names.
        assert content.startswith("<!doctype html>")
        assert 'id="sway-gauge"' in content
        assert 'id="sway-scatter"' in content
        assert "dk" in content
        # Stderr carries the "wrote HTML → ..." confirmation.
        assert "wrote HTML" in invocation.stderr

    def test_requires_out_flag(self, tmp_path: Path) -> None:
        result_json = tmp_path / "result.json"
        _write_sample_result(result_json)
        invocation = CliRunner().invoke(app, ["report", str(result_json), "--format", "html"])
        # Exit 2 — the 3 MB JS bundle has no business on stdout by default.
        assert invocation.exit_code == 2
        assert "requires --out" in invocation.stderr

    def test_terminal_with_out_errors(self, tmp_path: Path) -> None:
        """``--out`` is only for file-producing formats. Terminal stays
        in the console."""
        result_json = tmp_path / "result.json"
        target = tmp_path / "bogus.txt"
        _write_sample_result(result_json)
        invocation = CliRunner().invoke(
            app,
            ["report", str(result_json), "--format", "terminal", "--out", str(target)],
        )
        assert invocation.exit_code == 2
        assert "does not support --out" in invocation.stderr

    def test_markdown_with_out_writes_file(self, tmp_path: Path) -> None:
        """The ``--out`` flag works for ``--format md`` too (any file format)."""
        result_json = tmp_path / "result.json"
        md_out = tmp_path / "report.md"
        _write_sample_result(result_json)
        invocation = CliRunner().invoke(
            app,
            ["report", str(result_json), "--format", "md", "--out", str(md_out)],
        )
        assert invocation.exit_code == 0, invocation.stdout + invocation.stderr
        assert md_out.is_file()
        assert "# " in md_out.read_text(encoding="utf-8")

    def test_missing_plotly_surfaces_install_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When plotly is unavailable, the CLI exits 2 with the extras hint."""
        result_json = tmp_path / "result.json"
        out_html = tmp_path / "report.html"
        _write_sample_result(result_json)

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name.startswith("plotly"):
                raise ImportError("simulated missing plotly")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        invocation = CliRunner().invoke(
            app, ["report", str(result_json), "--format", "html", "--out", str(out_html)]
        )
        assert invocation.exit_code == 2
        assert "viz" in invocation.stderr
        # File must not have been written on the error path.
        assert not out_html.exists()
