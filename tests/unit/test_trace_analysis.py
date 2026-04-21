"""Tests for :mod:`dlm_sway.suite.trace_analysis` (S14 / F12)."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

from dlm_sway.cli.app import app
from dlm_sway.suite.trace_analysis import (
    ProbeSummary,
    TraceEvent,
    build_report,
    load,
    per_probe_summary,
    per_view_summary,
    render_json,
    render_markdown,
    render_terminal,
    slowest_events,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "trace_sample.jsonl"


class TestLoad:
    def test_loads_every_event(self) -> None:
        events = load(FIXTURE)
        assert len(events) == 8
        assert all(isinstance(e, TraceEvent) for e in events)

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        path.write_text(
            '{"ts": 1, "probe": "p", "view_id": "v", "prompt_hash": "a",'
            ' "top_k": 0, "op": "op", "wall_ms": 1.0, "hit": false}\n'
            "not json at all\n"
            "\n"
            '{"ts": 2, "probe": "q", "view_id": "w", "prompt_hash": "b",'
            ' "top_k": 0, "op": "op", "wall_ms": 2.0, "hit": true}\n',
            encoding="utf-8",
        )
        events = load(path)
        assert len(events) == 2

    def test_missing_optional_fields_fallback(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        # pre-S07 shape — no probe, no hit.
        path.write_text(
            '{"ts": 0, "view_id": "base", "prompt_hash": "x",'
            ' "top_k": 0, "op": "next_token_dist", "wall_ms": 5.0}\n',
            encoding="utf-8",
        )
        events = load(path)
        assert len(events) == 1
        assert events[0].probe is None
        assert events[0].hit is False


class TestPerProbeSummary:
    def test_buckets_by_probe(self) -> None:
        events = load(FIXTURE)
        summaries = per_probe_summary(events)
        by_name = {s.probe: s for s in summaries}
        assert set(by_name) == {"dk", "sis"}
        assert by_name["dk"].n_events == 4
        assert by_name["sis"].n_events == 4

    def test_wall_ms_accumulates(self) -> None:
        events = load(FIXTURE)
        summaries = per_probe_summary(events)
        by_name = {s.probe: s for s in summaries}
        # dk: 180.5 + 175.2 + 0.1 + 172.8 = 528.6
        assert by_name["dk"].total_ms == round(180.5 + 175.2 + 0.1 + 172.8, 1)

    def test_cache_hit_tracking(self) -> None:
        events = load(FIXTURE)
        summaries = per_probe_summary(events)
        by_name = {s.probe: s for s in summaries}
        # Each probe had 1 hit, 3 misses in the fixture.
        assert by_name["dk"].cache_hits == 1
        assert by_name["dk"].cache_misses == 3
        assert by_name["dk"].hit_rate == 0.25

    def test_sorted_by_wall_ms_descending(self) -> None:
        events = load(FIXTURE)
        summaries = per_probe_summary(events)
        # sis has bigger total (two ~500 ms events vs dk's ~180 ms events)
        assert summaries[0].probe == "sis"


class TestPerViewSummary:
    def test_buckets_by_view(self) -> None:
        events = load(FIXTURE)
        summaries = per_view_summary(events)
        view_ids = {s.view_id for s in summaries}
        assert view_ids == {"base", "ft"}

    def test_sorted_by_wall_ms(self) -> None:
        events = load(FIXTURE)
        summaries = per_view_summary(events)
        assert summaries[0].total_ms >= summaries[1].total_ms


class TestSlowestEvents:
    def test_returns_top_k(self) -> None:
        events = load(FIXTURE)
        slowest = slowest_events(events, k=3)
        assert len(slowest) == 3
        # Descending by wall_ms
        assert slowest[0].wall_ms >= slowest[1].wall_ms >= slowest[2].wall_ms
        # The slowest is the sis/base first rolling_logprob (520.3 ms).
        assert slowest[0].wall_ms == 520.3

    def test_k_larger_than_events_returns_all(self) -> None:
        events = load(FIXTURE)
        slowest = slowest_events(events, k=100)
        assert len(slowest) == len(events)


class TestRenderers:
    def test_json_shape(self) -> None:
        events = load(FIXTURE)
        report = build_report(events, slowest_k=3)
        payload = json.loads(render_json(report))
        assert payload["total_events"] == 8
        assert payload["overall_hit_rate"] == 0.25
        assert len(payload["per_probe"]) == 2
        assert len(payload["slowest"]) == 3

    def test_markdown_nonempty(self) -> None:
        events = load(FIXTURE)
        md = render_markdown(build_report(events))
        assert "# sway trace" in md
        assert "## per-probe" in md
        assert "## per-view" in md
        assert "dk" in md
        assert "sis" in md

    def test_terminal_renders_without_error(self) -> None:
        events = load(FIXTURE)
        console = Console(record=True, width=160)
        render_terminal(build_report(events), console=console)
        text = console.export_text()
        assert "sway trace" in text
        assert "dk" in text
        assert "sis" in text


class TestBuildReport:
    def test_empty_events(self) -> None:
        report = build_report([])
        assert report.total_events == 0
        assert report.total_wall_ms == 0.0
        assert report.overall_hit_rate == 0.0
        assert report.per_probe == []

    def test_probe_summary_matches(self) -> None:
        events = load(FIXTURE)
        report = build_report(events)
        assert isinstance(report.per_probe[0], ProbeSummary)
        assert report.total_events == len(events)


class TestCli:
    def test_terminal_default(self) -> None:
        result = CliRunner().invoke(app, ["trace", str(FIXTURE)])
        assert result.exit_code == 0, result.stdout
        assert "sway trace" in result.stdout
        assert "dk" in result.stdout
        assert "per-probe" in result.stdout

    def test_json_format(self) -> None:
        result = CliRunner().invoke(app, ["trace", str(FIXTURE), "--format", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.stdout)
        assert parsed["total_events"] == 8

    def test_markdown_format(self) -> None:
        result = CliRunner().invoke(app, ["trace", str(FIXTURE), "--format", "md"])
        assert result.exit_code == 0
        assert "# sway trace" in result.stdout

    def test_missing_file_exits_2(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.jsonl"
        result = CliRunner().invoke(app, ["trace", str(missing)])
        # Typer's PATH argument rejects non-existent paths with its own
        # validation (exit 2) before our code runs.
        assert result.exit_code == 2

    def test_empty_file_exits_1(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        result = CliRunner().invoke(app, ["trace", str(empty)])
        assert result.exit_code == 1
        assert "no events" in result.stdout + result.stderr

    def test_slowest_override(self) -> None:
        result = CliRunner().invoke(
            app, ["trace", str(FIXTURE), "--format", "json", "--slowest", "2"]
        )
        assert result.exit_code == 0
        parsed = json.loads(result.stdout)
        assert len(parsed["slowest"]) == 2
