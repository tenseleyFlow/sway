"""S11 prove-the-value: committed history fixture catches a planted regression.

The sprint's audit §F5 experiment: commit a ``sway-history/`` directory
with N dated result JSONs, run ``sway compare --fail-on-regression 0.10``,
watch it flag the planted regression in the newest run.

Fixture layout (``tests/fixtures/sway-history/``):
    01-2026-01-15.json  — baseline (overall=0.78)
    02-2026-01-22.json  — improved (overall=0.81) — adapter v2 retrained
    03-2026-01-29.json  — over-trained (overall=0.66) — adapter v3

The planted story: v3 is an over-trained adapter. ``delta_kl`` rises
(more divergence from base) but ``section_internalization`` and
``calibration_drift`` both drop sharply — the adapter memorized one
section at the cost of the others, and forgot broader knowledge.
``sway compare --fail-on-regression 0.10`` must exit 1 and surface the
two regressions.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dlm_sway.cli.app import app

HISTORY_DIR = Path(__file__).parent.parent / "fixtures" / "sway-history"


def _history_paths() -> list[str]:
    paths = sorted(HISTORY_DIR.glob("*.json"))
    assert len(paths) == 3, f"expected 3 history JSONs, got {len(paths)}: {paths}"
    return [str(p) for p in paths]


def test_history_fixture_exists() -> None:
    """Sanity: the committed history fixture is three JSONs in order."""
    paths = sorted(HISTORY_DIR.glob("*.json"))
    assert [p.name for p in paths] == [
        "01-2026-01-15.json",
        "02-2026-01-22.json",
        "03-2026-01-29.json",
    ]


def test_compare_catches_planted_regression() -> None:
    """The prove-the-value run: --fail-on-regression 0.10 exits 1 + flags both probes."""
    result = CliRunner().invoke(
        app,
        ["compare", *_history_paths(), "--format", "json", "--fail-on-regression", "0.10"],
    )
    assert result.exit_code == 1, (
        "expected exit=1 (planted regression ≥ 0.10 on run 03); "
        f"got exit={result.exit_code}\nstdout:\n{result.stdout}"
    )
    # JSON payload is emitted to stdout even though the gate failed.
    import json as _json

    parsed = _json.loads(result.stdout)
    regressed_names = [entry["probe"] for entry in parsed["latest_regressions"]]
    # Both section_internalization and calibration_drift dropped ≥0.10.
    assert "section_internalization" in regressed_names, regressed_names
    assert "calibration_drift" in regressed_names, regressed_names
    # delta_kl rose, not regressed — must not appear.
    assert "delta_kl" not in regressed_names

    # Shape contract: 4 probes × 3 runs + composite timeline present.
    assert set(parsed["probe_names"]) == {
        "adapter_ablation",
        "calibration_drift",
        "delta_kl",
        "section_internalization",
    }
    assert len(parsed["composite_series"]) == 3
    # Composite drops on the third run.
    assert parsed["composite_series"][2] < parsed["composite_series"][1]


def test_compare_improving_history_exits_0() -> None:
    """Control: the first two runs alone show an improvement — exit 0."""
    first_two = _history_paths()[:2]
    result = CliRunner().invoke(
        app,
        ["compare", *first_two, "--format", "json", "--fail-on-regression", "0.10"],
    )
    assert result.exit_code == 0, result.stdout


def test_terminal_output_surfaces_regressions() -> None:
    """Human-facing terminal render lists the regressed probe names."""
    result = CliRunner().invoke(
        app,
        ["compare", *_history_paths(), "--fail-on-regression", "0.10"],
    )
    assert result.exit_code == 1, result.stdout
    assert "regressions" in result.stdout.lower()
    assert "section_internalization" in result.stdout
    assert "calibration_drift" in result.stdout
