"""C11: snapshot tests for the three report formats.

JSON is the machine-readable contract downstream tools depend on;
markdown is the CI-friendly human report; JUnit is the CI-dashboard
plumbing. Silent schema drift in any of them breaks consumers.

We serialize a deterministic fixture suite + score through each
emitter and byte-compare against checked-in snapshots under
``tests/snapshots/``. Intentional schema bumps update the snapshot in
the same commit (``SWAY_UPDATE_SNAPSHOTS=1 uv run pytest``); anything
else surfaces as a failed test.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dlm_sway.core.result import (
    DeterminismReport,
    ProbeResult,
    SuiteResult,
    SwayScore,
    Verdict,
)
from dlm_sway.suite import report

SNAPSHOT_DIR = Path(__file__).parent.parent / "snapshots"


def _fixture_suite_and_score() -> tuple[SuiteResult, SwayScore]:
    """A hand-crafted SuiteResult whose every field is deterministic."""
    started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    finished = datetime(2026, 1, 1, 12, 0, 2, 500000, tzinfo=UTC)  # 2.5s wall
    probes = (
        ProbeResult(
            name="dk",
            kind="delta_kl",
            verdict=Verdict.PASS,
            score=0.87,
            raw=0.456,
            z_score=5.12,
            evidence={"divergence_kind": "js", "num_prompts": 4, "weight": 1.0},
            message="mean js=0.4560, z=+5.12σ vs null",
            duration_s=0.123,
            ci_95=(0.412, 0.497),
        ),
        ProbeResult(
            name="sis",
            kind="section_internalization",
            verdict=Verdict.FAIL,
            score=0.30,
            raw=0.012,
            z_score=0.5,
            evidence={"num_sections": 4, "passing_frac": 0.25, "weight": 1.0},
            message="1/4 sections cleared effective_sis≥0.05",
            duration_s=0.456,
        ),
        ProbeResult(
            name="lk",
            kind="leakage",
            verdict=Verdict.SKIP,
            score=None,
            message="no PROSE sections to test for leakage",
            duration_s=0.001,
        ),
        ProbeResult(
            name="ablation",
            kind="adapter_ablation",
            verdict=Verdict.ERROR,
            score=None,
            raw=None,
            message="backend does not implement ScalableDifferentialBackend",
            duration_s=0.0,
        ),
    )
    suite = SuiteResult(
        spec_path="/fixture/sway.yaml",
        started_at=started,
        finished_at=finished,
        base_model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        adapter_id="/fixture/runs/adapter/v0003",
        sway_version="0.1.0.dev0",
        probes=probes,
        null_stats={"delta_kl": {"mean": 0.01, "std": 0.005, "n": 3.0}},
        determinism=DeterminismReport(
            class_="best_effort",
            seed=0,
            notes=("CPU-only backend: strict determinism depends on BLAS impl",),
        ),
    )
    score = SwayScore(
        overall=0.65,
        components={
            "adherence": 0.87,
            "attribution": 0.30,
            "calibration": 0.50,
            "ablation": 0.0,
            "baseline": 1.0,
        },
        weights={
            "adherence": 0.30,
            "attribution": 0.35,
            "calibration": 0.20,
            "ablation": 0.15,
            "baseline": 0.0,
        },
        band="healthy",
        findings=(
            "sis (section_internalization) failed: 1/4 sections cleared effective_sis≥0.05",
            "ablation score is 0.00 — below the noise threshold",
        ),
    )
    return suite, score


def _compare_to_snapshot(actual: str, snapshot_name: str) -> None:
    """Byte-compare ``actual`` against the snapshot file, updating when asked."""
    path = SNAPSHOT_DIR / snapshot_name
    if os.environ.get("SWAY_UPDATE_SNAPSHOTS") == "1" or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8")
        pytest.skip(
            f"snapshot {snapshot_name} written — re-run without SWAY_UPDATE_SNAPSHOTS to verify"
        )
    expected = path.read_text(encoding="utf-8")
    assert actual == expected, (
        f"{snapshot_name} drifted from snapshot.\n"
        f"To accept the new output intentionally, run:\n"
        f"    SWAY_UPDATE_SNAPSHOTS=1 uv run pytest tests/unit/test_report_snapshot.py\n"
        f"and commit the updated file.\n"
    )


def test_json_schema_snapshot() -> None:
    suite, score = _fixture_suite_and_score()
    actual = report.to_json(suite, score)
    # Sanity: it's parseable JSON with the expected top-level fields.
    parsed = json.loads(actual)
    assert parsed["schema_version"] == 1
    assert parsed["determinism"] is not None
    assert parsed["determinism"]["seed"] == 0
    _compare_to_snapshot(actual + "\n", "report.json")


def test_markdown_layout_snapshot() -> None:
    suite, score = _fixture_suite_and_score()
    actual = report.to_markdown(suite, score)
    _compare_to_snapshot(actual, "report.md")


def test_junit_layout_snapshot() -> None:
    suite, score = _fixture_suite_and_score()
    actual = report.to_junit(suite, score)
    # ElementTree tostring doesn't include a trailing newline; normalize
    # so diffs don't hinge on platform-dependent whitespace.
    actual = actual.strip() + "\n"
    # Strip the variable ``time`` attribute on <testsuite> — it encodes
    # wall_seconds but all the testcase times are deterministic, so this
    # single attribute is the only moving part we need to mask.
    actual = re.sub(r' time="[\d.]+"', ' time="<wall>"', actual, count=1)
    _compare_to_snapshot(actual, "report.junit.xml")
