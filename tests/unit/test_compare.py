"""Tests for :mod:`dlm_sway.suite.compare` (S11 / F5)."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from rich.console import Console

from dlm_sway.core.result import (
    ProbeResult,
    SuiteResult,
    SwayScore,
    Verdict,
)
from dlm_sway.suite.compare import (
    CompareMatrix,
    build_matrix,
    render_json,
    render_markdown,
    render_terminal,
)

SNAPSHOT_DIR = Path(__file__).parent.parent / "snapshots"


def _probe(name: str, score: float | None, verdict: Verdict = Verdict.PASS) -> ProbeResult:
    return ProbeResult(
        name=name,
        kind=name.split("_")[0] if "_" in name else "delta_kl",
        verdict=verdict,
        score=score,
        raw=score,
        message=f"{name} score={score}",
    )


def _suite(
    probes: list[ProbeResult], *, started_minute: int, overall: float
) -> tuple[SuiteResult, SwayScore]:
    # ``started_minute`` is treated as offset-from-noon in 30-minute
    # increments (run-a=0, run-b=30, run-c=60 maps to 12:00, 12:30, 13:00).
    hours_offset, mins = divmod(started_minute, 60)
    started = datetime(2026, 1, 1, 12 + hours_offset, mins, 0, tzinfo=UTC)
    finished = datetime(2026, 1, 1, 12 + hours_offset, mins, 30, tzinfo=UTC)
    suite = SuiteResult(
        spec_path="fixture.yaml",
        started_at=started,
        finished_at=finished,
        base_model_id="test/base",
        adapter_id="adapter-v1",
        sway_version="0.0.0",
        probes=tuple(probes),
    )
    score = SwayScore(
        overall=overall,
        components={"adherence": overall, "attribution": overall},
        band=SwayScore.band_for(overall),
    )
    return suite, score


def _three_run_history() -> list[tuple[SuiteResult, SwayScore]]:
    """Three synthetic runs with overlapping-but-not-identical probe sets:

    - run-a: {delta_kl, section_internalization}
    - run-b: {delta_kl, section_internalization, leakage} (leakage *added*)
    - run-c: {delta_kl, leakage}                         (section_internalization *removed*)
    """
    run_a = _suite(
        [
            _probe("delta_kl", 0.80),
            _probe("section_internalization", 0.70),
        ],
        started_minute=0,
        overall=0.75,
    )
    run_b = _suite(
        [
            _probe("delta_kl", 0.82),
            _probe("section_internalization", 0.72),
            _probe("leakage", 0.90),
        ],
        started_minute=30,
        overall=0.81,
    )
    run_c = _suite(
        [
            _probe("delta_kl", 0.65),  # dropped 0.17
            _probe("leakage", 0.88),  # dropped 0.02
        ],
        started_minute=60,
        overall=0.72,
    )
    return [run_a, run_b, run_c]


class TestBuildMatrix:
    def test_union_probe_names_sorted(self) -> None:
        matrix = build_matrix(_three_run_history())
        assert matrix.probe_names == ("delta_kl", "leakage", "section_internalization")

    def test_missing_cells_are_none(self) -> None:
        matrix = build_matrix(_three_run_history())
        # leakage did not exist in run-a.
        assert matrix.scores["leakage"][0] is None
        assert matrix.scores["leakage"][1] == pytest.approx(0.90)
        # section_internalization did not exist in run-c.
        assert matrix.scores["section_internalization"][2] is None

    def test_deltas_skip_none_neighbors(self) -> None:
        matrix = build_matrix(_three_run_history())
        # leakage: None → 0.90 → 0.88. First delta None (None neighbor);
        # second delta ≈ -0.02.
        assert matrix.deltas["leakage"][0] is None
        assert matrix.deltas["leakage"][1] == pytest.approx(-0.02)
        # section_internalization: 0.70 → 0.72 → None. First +0.02; second None.
        assert matrix.deltas["section_internalization"][0] == pytest.approx(0.02)
        assert matrix.deltas["section_internalization"][1] is None

    def test_composite_series_parallels_labels(self) -> None:
        matrix = build_matrix(_three_run_history())
        assert matrix.n_runs == 3
        assert matrix.composite_series == pytest.approx([0.75, 0.81, 0.72])

    def test_labels_default_to_timestamps(self) -> None:
        matrix = build_matrix(_three_run_history())
        # Default labels come from finished_at ISO strings.
        assert matrix.labels[0].startswith("2026-01-01T12:00")

    def test_labels_override(self) -> None:
        runs = _three_run_history()
        matrix = build_matrix(runs, labels=["v1", "v2", "v3"])
        assert matrix.labels == ("v1", "v2", "v3")

    def test_labels_length_mismatch_raises(self) -> None:
        runs = _three_run_history()
        with pytest.raises(ValueError, match="labels length"):
            build_matrix(runs, labels=["v1", "v2"])

    def test_empty_results_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one run"):
            build_matrix([])

    def test_non_finite_score_coerced_to_none(self) -> None:
        suite, score = _suite([_probe("delta_kl", float("nan"))], started_minute=0, overall=0.5)
        matrix = build_matrix([(suite, score)])
        assert matrix.scores["delta_kl"] == [None]


class TestLatestRegressions:
    def test_single_run_yields_empty(self) -> None:
        suite, score = _suite([_probe("dk", 0.5)], started_minute=0, overall=0.5)
        matrix = build_matrix([(suite, score)])
        assert matrix.latest_regressions(threshold=0.1) == []

    def test_catches_newest_regression(self) -> None:
        matrix = build_matrix(_three_run_history())
        regs = matrix.latest_regressions(threshold=0.10)
        # delta_kl dropped 0.17; leakage only 0.02 (below threshold).
        assert [name for name, _ in regs] == ["delta_kl"]
        assert regs[0][1] == pytest.approx(-0.17)

    def test_zero_threshold_disables(self) -> None:
        matrix = build_matrix(_three_run_history())
        assert matrix.latest_regressions(threshold=0.0) == []

    def test_sorted_by_severity(self) -> None:
        run_a = _suite(
            [_probe("p1", 0.9), _probe("p2", 0.9), _probe("p3", 0.9)],
            started_minute=0,
            overall=0.9,
        )
        run_b = _suite(
            [_probe("p1", 0.6), _probe("p2", 0.4), _probe("p3", 0.75)],
            started_minute=30,
            overall=0.6,
        )
        matrix = build_matrix([run_a, run_b])
        regs = matrix.latest_regressions(threshold=0.10)
        # p2 dropped 0.5, p1 dropped 0.3, p3 dropped 0.15 → order p2, p1, p3
        assert [name for name, _ in regs] == ["p2", "p1", "p3"]


class TestRenderJson:
    def test_round_trip(self) -> None:
        matrix = build_matrix(_three_run_history(), labels=["v1", "v2", "v3"])
        raw = render_json(matrix, regression_threshold=0.10)
        parsed = json.loads(raw)
        assert parsed["labels"] == ["v1", "v2", "v3"]
        assert parsed["composite_series"] == pytest.approx([0.75, 0.81, 0.72])
        assert parsed["latest_regressions"][0]["probe"] == "delta_kl"
        assert parsed["latest_regressions"][0]["delta"] == pytest.approx(-0.17)
        assert parsed["regression_threshold"] == pytest.approx(0.10)

    def test_no_regression_at_higher_threshold(self) -> None:
        matrix = build_matrix(_three_run_history(), labels=["v1", "v2", "v3"])
        raw = render_json(matrix, regression_threshold=0.50)
        parsed = json.loads(raw)
        assert parsed["latest_regressions"] == []


class TestRenderTerminal:
    def test_renders_without_error(self) -> None:
        matrix = build_matrix(_three_run_history(), labels=["v1", "v2", "v3"])
        console = Console(record=True, width=160)
        render_terminal(matrix, console=console, regression_threshold=0.10)
        out = console.export_text()
        # Smoke: headers + probe names + composite all appear.
        assert "sway compare" in out
        assert "delta_kl" in out
        assert "composite" in out
        assert "regressions" in out  # regression footer emitted

    def test_suppresses_regression_block_when_none(self) -> None:
        matrix = build_matrix(_three_run_history(), labels=["v1", "v2", "v3"])
        console = Console(record=True, width=160)
        render_terminal(matrix, console=console, regression_threshold=0.90)
        out = console.export_text()
        assert "regressions" not in out


class TestRenderMarkdownSnapshot:
    def test_markdown_snapshot(self) -> None:
        """Lock the markdown layout. Run
        ``SWAY_UPDATE_SNAPSHOTS=1 uv run pytest tests/unit/test_compare.py``
        to regenerate after an intentional format change.
        """
        matrix = build_matrix(_three_run_history(), labels=["v1", "v2", "v3"])
        actual = render_markdown(matrix, regression_threshold=0.10)

        path = SNAPSHOT_DIR / "compare.md"
        if os.environ.get("SWAY_UPDATE_SNAPSHOTS") == "1" or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(actual, encoding="utf-8")
            pytest.skip(
                "snapshot compare.md written — re-run without SWAY_UPDATE_SNAPSHOTS to verify"
            )
        expected = path.read_text(encoding="utf-8")
        assert actual == expected, (
            "compare.md drifted from snapshot.\n"
            "To accept the new output intentionally, run:\n"
            "    SWAY_UPDATE_SNAPSHOTS=1 uv run pytest tests/unit/test_compare.py\n"
            "and commit the updated file.\n"
        )


class TestCompareMatrixProperties:
    def test_n_runs_matches_labels(self) -> None:
        matrix = CompareMatrix(
            labels=("a", "b"),
            timestamps=("", ""),
            probe_names=(),
        )
        assert matrix.n_runs == 2
