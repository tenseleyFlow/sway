"""Smoke tests for the sway CLI.

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
    assert "sway" in result.stdout


def test_help_lists_all_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("run", "gate", "check", "diff", "autogen", "doctor", "report"):
        assert cmd in result.stdout


def test_doctor_runs(capsys) -> None:  # type: ignore[no-untyped-def]
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0
    # Rich applies color codes by default; assert the bare product name appears.
    assert "sway" in result.stdout
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
    assert "sway report" in md.stdout

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


# -- Sprint 06 additions ----------------------------------------------


class TestDoctorJson:
    """D7: ``sway doctor --json`` must emit a parseable payload."""

    def test_json_is_parseable(self) -> None:
        result = CliRunner().invoke(app, ["doctor", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert "sway_version" in payload
        assert "python" in payload
        assert "platform" in payload
        assert "extras" in payload
        # Every extra bucket is a mapping of module → version-or-null.
        assert set(payload["extras"]) >= {
            "hf",
            "mlx",
            "semsim",
            "style",
            "dlm",
            "viz",
            "api",
            "pytest",
        }
        # F04 regression: load-bearing deps appear under the right extras.
        assert "plotly" in payload["extras"]["viz"]
        assert "sklearn" in payload["extras"]["semsim"]
        assert "httpx" in payload["extras"]["api"]
        assert "tenacity" in payload["extras"]["api"]


class TestListProbes:
    """D6: ``sway list-probes`` prints the registered kinds."""

    def test_emits_every_shipped_kind(self) -> None:
        result = CliRunner().invoke(app, ["list-probes"])
        assert result.exit_code == 0
        for kind in (
            "delta_kl",
            "adapter_revert",
            "prompt_collapse",
            "section_internalization",
            "paraphrase_invariance",
            "preference_flip",
            "style_fingerprint",
            "calibration_drift",
            "leakage",
            "adapter_ablation",
            "null_adapter",
            "external_perplexity",
            "cluster_kl",
        ):
            assert kind in result.stdout

    def test_every_probe_has_a_summary_line(self) -> None:
        """F03 regression — before the module-docstring fallback, half
        the probe rows shipped with an empty summary column."""
        from dlm_sway.probes.base import registry

        result = CliRunner().invoke(app, ["list-probes"])
        assert result.exit_code == 0
        out = result.stdout
        for kind in sorted(registry()):
            # Find the row by its leading ``kind`` token. Rich wraps
            # long summaries across lines, so match any non-empty
            # continuation after the category column.
            idx = out.find(kind)
            assert idx != -1, f"{kind} missing from list-probes output"
            row = out[idx : out.find("\n", idx)]
            # Row format: "kind  category  summary..."
            tokens = row.split()
            # Past the 2nd column (category) there should be at least one
            # summary token. Empty rows surfaced as len(tokens) == 2.
            assert len(tokens) > 2, f"{kind} has an empty summary: {row!r}"


class TestReportFormatEnum:
    """D11: unknown ``--format`` surfaces a clear error, not silent terminal."""

    def test_unknown_format_rejected(self, tmp_path: Path) -> None:
        result_path = tmp_path / "r.json"
        result_path.write_text(
            json.dumps(
                {
                    "sway_version": "0",
                    "base_model_id": "b",
                    "adapter_id": "a",
                    "score": {"overall": 0.0, "band": "noise", "components": {}, "findings": []},
                    "probes": [],
                }
            ),
            encoding="utf-8",
        )
        result = CliRunner().invoke(app, ["report", str(result_path), "--format", "csv"])
        assert result.exit_code != 0
        combined = (result.stdout or "") + (result.output or "")
        assert "csv" in combined.lower() or "invalid" in combined.lower()


class TestCheckBaseInference:
    """D4: ``sway check`` reads base_model_name_or_path from adapter_config.json."""

    def test_reads_base_from_adapter_config(self, tmp_path: Path) -> None:
        from dlm_sway.cli.commands import _infer_base_from_adapter_config

        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "HuggingFaceTB/SmolLM2-135M-Instruct"}),
            encoding="utf-8",
        )
        assert _infer_base_from_adapter_config(adapter) == "HuggingFaceTB/SmolLM2-135M-Instruct"

    def test_returns_none_when_config_missing(self, tmp_path: Path) -> None:
        from dlm_sway.cli.commands import _infer_base_from_adapter_config

        assert _infer_base_from_adapter_config(tmp_path) is None

    def test_returns_none_when_field_missing(self, tmp_path: Path) -> None:
        from dlm_sway.cli.commands import _infer_base_from_adapter_config

        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(json.dumps({"rank": 8}), encoding="utf-8")
        assert _infer_base_from_adapter_config(adapter) is None

    def test_returns_none_when_config_malformed(self, tmp_path: Path) -> None:
        from dlm_sway.cli.commands import _infer_base_from_adapter_config

        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{ not json", encoding="utf-8")
        assert _infer_base_from_adapter_config(adapter) is None


class TestCheckBanner:
    """D12: ``_check_banner`` maps z-score to the right verdict tier."""

    def _suite_with_z(self, z_value: float | None) -> tuple:
        from datetime import UTC, datetime

        from dlm_sway.core.result import ProbeResult, SuiteResult, SwayScore, Verdict

        now = datetime.now(UTC)
        probes = (
            ProbeResult(
                name="dk",
                kind="delta_kl",
                verdict=Verdict.PASS if z_value and z_value >= 3 else Verdict.FAIL,
                score=0.5,
                z_score=z_value,
            ),
        )
        suite = SuiteResult(
            spec_path="<t>",
            started_at=now,
            finished_at=now,
            base_model_id="b",
            adapter_id="a",
            sway_version="0.0.0",
            probes=probes,
        )
        score = SwayScore(
            overall=0.5,
            components={"adherence": 0.5},
            band="partial",
        )
        return suite, score

    def test_high_z_is_green(self) -> None:
        from dlm_sway.cli.commands import _check_banner

        suite, score = self._suite_with_z(4.5)
        text, style = _check_banner(score, suite)
        assert "✅" in text
        assert "above noise" in text
        assert "green" in style

    def test_marginal_z_is_yellow(self) -> None:
        from dlm_sway.cli.commands import _check_banner

        suite, score = self._suite_with_z(1.5)
        text, style = _check_banner(score, suite)
        assert "⚠️" in text
        assert "yellow" in style

    def test_low_z_is_red(self) -> None:
        from dlm_sway.cli.commands import _check_banner

        suite, score = self._suite_with_z(0.3)
        text, style = _check_banner(score, suite)
        assert "❌" in text
        assert "red" in style

    def test_missing_z_falls_back_to_composite(self) -> None:
        from dlm_sway.cli.commands import _check_banner

        suite, score = self._suite_with_z(None)
        text, _style = _check_banner(score, suite)
        # No "σ above noise" language when we don't have a z-score.
        assert "σ" not in text
