"""Analyze forward-pass trace JSONLs (S14 / F12).

The S07 trace infrastructure writes one line per backend scoring call
into a JSONL file when ``sway run --trace path.jsonl`` is set. Each
event carries ``(probe, view_id, prompt_hash, top_k, op, wall_ms, hit)``.
Raw events are easy for backend debugging; they're not easy for
answering the user-level questions the audit's F12 pitched:

- *"Which probe × view pair dominated suite wall time?"*
- *"Did the S07 cache actually help this run — overall and per probe?"*
- *"Which individual prompts are the slowest? Can I shrink the suite
  by cutting the expensive tail?"*

This module takes the raw event stream and produces three summary
views answering those three questions, plus three renderers
(terminal, markdown, JSON) that match the conventions of
:mod:`dlm_sway.suite.report` and :mod:`dlm_sway.suite.compare`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One forward-pass event from the ``sway run --trace`` JSONL."""

    ts: float
    probe: str | None
    view_id: str
    prompt_hash: str
    top_k: int
    op: str
    wall_ms: float
    hit: bool


@dataclass(frozen=True, slots=True)
class ProbeSummary:
    """Per-probe aggregation across a trace file."""

    probe: str
    n_events: int
    total_ms: float
    cache_hits: int
    cache_misses: int

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return float(self.cache_hits) / total if total else 0.0


@dataclass(frozen=True, slots=True)
class ViewSummary:
    """Per-view (``base``, ``ft``, ``null_42``, …) aggregation."""

    view_id: str
    n_events: int
    total_ms: float


@dataclass(frozen=True, slots=True)
class TraceReport:
    """Full analysis of a trace file — the shape every renderer consumes."""

    total_events: int
    total_wall_ms: float
    overall_hit_rate: float
    per_probe: list[ProbeSummary] = field(default_factory=list)
    per_view: list[ViewSummary] = field(default_factory=list)
    slowest: list[TraceEvent] = field(default_factory=list)


# ----------------------------------------------------------------------
# Load + analyze
# ----------------------------------------------------------------------


def load(path: Path) -> list[TraceEvent]:
    """Parse the JSONL trace at ``path`` into :class:`TraceEvent` objects.

    Malformed lines are skipped with no error — old traces from a
    schema-bumped writer shouldn't crash the analyzer. Missing
    optional fields fall back to sensible defaults (``probe=None`` for
    pre-S07 traces; ``top_k=0`` for ``logprob_of`` which doesn't carry
    one).
    """
    events: list[TraceEvent] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                events.append(
                    TraceEvent(
                        ts=float(payload.get("ts", 0.0)),
                        probe=payload.get("probe"),
                        view_id=str(payload.get("view_id", "")),
                        prompt_hash=str(payload.get("prompt_hash", "")),
                        top_k=int(payload.get("top_k", 0)),
                        op=str(payload.get("op", "")),
                        wall_ms=float(payload.get("wall_ms", 0.0)),
                        hit=bool(payload.get("hit", False)),
                    )
                )
            except (TypeError, ValueError):
                # Partial / corrupted event — skip without halting.
                continue
    return events


def per_probe_summary(events: list[TraceEvent]) -> list[ProbeSummary]:
    """Group events by probe, sorted by total wall time (descending)."""
    buckets: dict[str, dict[str, Any]] = {}
    for e in events:
        key = e.probe or "<unassigned>"
        b = buckets.setdefault(
            key,
            {"n": 0, "ms": 0.0, "hits": 0, "misses": 0},
        )
        b["n"] += 1
        b["ms"] += e.wall_ms
        if e.hit:
            b["hits"] += 1
        else:
            b["misses"] += 1
    out = [
        ProbeSummary(
            probe=name,
            n_events=b["n"],
            total_ms=b["ms"],
            cache_hits=b["hits"],
            cache_misses=b["misses"],
        )
        for name, b in buckets.items()
    ]
    out.sort(key=lambda s: -s.total_ms)
    return out


def per_view_summary(events: list[TraceEvent]) -> list[ViewSummary]:
    """Group events by view_id, sorted by total wall time (descending)."""
    buckets: dict[str, dict[str, Any]] = {}
    for e in events:
        b = buckets.setdefault(e.view_id, {"n": 0, "ms": 0.0})
        b["n"] += 1
        b["ms"] += e.wall_ms
    out = [
        ViewSummary(view_id=name, n_events=b["n"], total_ms=b["ms"]) for name, b in buckets.items()
    ]
    out.sort(key=lambda s: -s.total_ms)
    return out


def slowest_events(events: list[TraceEvent], *, k: int = 10) -> list[TraceEvent]:
    """Return the ``k`` events with the largest ``wall_ms``.

    Cache hits dominate the event count but rarely the wall-time tail —
    a miss that took 2 seconds is far more actionable than ten hits of
    0.1 ms each. We include both hits and misses in the sort and let
    the caller read ``hit`` per event to interpret.
    """
    return sorted(events, key=lambda e: -e.wall_ms)[:k]


def build_report(events: list[TraceEvent], *, slowest_k: int = 10) -> TraceReport:
    """One-call helper for renderers: turn events into a full report."""
    total_ms = sum(e.wall_ms for e in events)
    total_hits = sum(1 for e in events if e.hit)
    overall_hit_rate = total_hits / len(events) if events else 0.0
    return TraceReport(
        total_events=len(events),
        total_wall_ms=total_ms,
        overall_hit_rate=overall_hit_rate,
        per_probe=per_probe_summary(events),
        per_view=per_view_summary(events),
        slowest=slowest_events(events, k=slowest_k),
    )


# ----------------------------------------------------------------------
# Renderers
# ----------------------------------------------------------------------


def render_terminal(report: TraceReport, *, console: Console | None = None) -> None:
    """Rich-formatted three-table breakdown of the trace."""
    c = console or Console()
    c.print(
        Text(
            f"sway trace — {report.total_events} events, "
            f"total {report.total_wall_ms / 1000.0:.2f}s, "
            f"cache hit rate {report.overall_hit_rate:.1%}",
            style="bold",
        )
    )
    c.print()

    probe_table = Table(
        show_header=True, header_style="bold", box=None, padding=(0, 1), title="per-probe"
    )
    probe_table.add_column("probe", style="cyan")
    probe_table.add_column("events", justify="right")
    probe_table.add_column("wall_ms", justify="right")
    probe_table.add_column("hits", justify="right")
    probe_table.add_column("hit_rate", justify="right")
    for p in report.per_probe:
        probe_table.add_row(
            p.probe,
            str(p.n_events),
            f"{p.total_ms:,.1f}",
            str(p.cache_hits),
            f"{p.hit_rate:.1%}",
        )
    c.print(probe_table)
    c.print()

    view_table = Table(
        show_header=True, header_style="bold", box=None, padding=(0, 1), title="per-view"
    )
    view_table.add_column("view_id", style="magenta")
    view_table.add_column("events", justify="right")
    view_table.add_column("wall_ms", justify="right")
    for v in report.per_view:
        view_table.add_row(v.view_id, str(v.n_events), f"{v.total_ms:,.1f}")
    c.print(view_table)
    c.print()

    if report.slowest:
        slow_table = Table(
            show_header=True,
            header_style="bold",
            box=None,
            padding=(0, 1),
            title=f"slowest {len(report.slowest)} events",
        )
        slow_table.add_column("probe")
        slow_table.add_column("view")
        slow_table.add_column("op")
        slow_table.add_column("wall_ms", justify="right")
        slow_table.add_column("hit")
        for e in report.slowest:
            slow_table.add_row(
                e.probe or "—",
                e.view_id,
                e.op,
                f"{e.wall_ms:,.1f}",
                "yes" if e.hit else "no",
            )
        c.print(slow_table)


def render_markdown(report: TraceReport) -> str:
    """Markdown equivalent of :func:`render_terminal`."""
    buf = StringIO()
    buf.write(
        f"# sway trace — {report.total_events} events, "
        f"total {report.total_wall_ms / 1000.0:.2f}s, "
        f"hit rate {report.overall_hit_rate:.1%}\n\n"
    )
    buf.write("## per-probe\n\n")
    buf.write("| probe | events | wall_ms | hits | hit_rate |\n|---|---:|---:|---:|---:|\n")
    for p in report.per_probe:
        buf.write(
            f"| {p.probe} | {p.n_events} | {p.total_ms:,.1f} | "
            f"{p.cache_hits} | {p.hit_rate:.1%} |\n"
        )
    buf.write("\n## per-view\n\n")
    buf.write("| view_id | events | wall_ms |\n|---|---:|---:|\n")
    for v in report.per_view:
        buf.write(f"| {v.view_id} | {v.n_events} | {v.total_ms:,.1f} |\n")
    if report.slowest:
        buf.write(f"\n## slowest {len(report.slowest)} events\n\n")
        buf.write("| probe | view | op | wall_ms | hit |\n|---|---|---|---:|---|\n")
        for e in report.slowest:
            buf.write(
                f"| {e.probe or '—'} | {e.view_id} | {e.op} | "
                f"{e.wall_ms:,.1f} | {'yes' if e.hit else 'no'} |\n"
            )
    return buf.getvalue()


def render_json(report: TraceReport) -> str:
    """Machine-readable JSON. Same field names as the dataclasses."""
    payload: dict[str, Any] = {
        "total_events": report.total_events,
        "total_wall_ms": report.total_wall_ms,
        "overall_hit_rate": report.overall_hit_rate,
        "per_probe": [
            {
                "probe": p.probe,
                "n_events": p.n_events,
                "total_ms": p.total_ms,
                "cache_hits": p.cache_hits,
                "cache_misses": p.cache_misses,
                "hit_rate": p.hit_rate,
            }
            for p in report.per_probe
        ],
        "per_view": [
            {
                "view_id": v.view_id,
                "n_events": v.n_events,
                "total_ms": v.total_ms,
            }
            for v in report.per_view
        ],
        "slowest": [
            {
                "ts": e.ts,
                "probe": e.probe,
                "view_id": e.view_id,
                "prompt_hash": e.prompt_hash,
                "top_k": e.top_k,
                "op": e.op,
                "wall_ms": e.wall_ms,
                "hit": e.hit,
            }
            for e in report.slowest
        ],
    }
    return json.dumps(payload, indent=2)


__all__ = [
    "ProbeSummary",
    "TraceEvent",
    "TraceReport",
    "ViewSummary",
    "build_report",
    "load",
    "per_probe_summary",
    "per_view_summary",
    "render_json",
    "render_markdown",
    "render_terminal",
    "slowest_events",
]
