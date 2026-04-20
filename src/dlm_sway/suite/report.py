"""Report emitters: terminal (rich), JSON, JUnit XML, markdown.

The terminal renderer is the one a user sees; it's the product surface.
It must communicate the verdict *and* the supporting evidence without
forcing the user to open the JSON.

JSON is the machine-readable source of truth — same fields as the
:class:`SuiteResult` dataclass but flattened for easy downstream parsing
(dashboards, diff tools, history tracking).

JUnit XML exists to drop into CI pipelines so ``dlm-sway gate``
integrates with existing test dashboards with no extra glue.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from io import StringIO
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from dlm_sway.core.result import ProbeResult, SuiteResult, SwayScore, Verdict

_VERDICT_STYLE = {
    Verdict.PASS: "bold green",
    Verdict.FAIL: "bold red",
    Verdict.WARN: "bold yellow",
    Verdict.SKIP: "dim",
    Verdict.ERROR: "bold magenta",
}


def to_terminal(suite: SuiteResult, score: SwayScore, *, console: Console | None = None) -> None:
    """Render the report to a rich Console (stdout by default)."""
    c = console or Console()

    header = Text.assemble(
        ("dlm-sway report — ", "bold"),
        (suite.base_model_id, "cyan"),
        ("  vs  ", "dim"),
        (_adapter_label(suite.adapter_id), "cyan"),
    )
    c.print(Panel(header, expand=False, border_style="blue"))

    c.print()
    c.print(
        Text.assemble(
            ("overall: ", "bold"),
            (f"{score.overall:.2f}", _score_style(score.overall)),
            ("  ", ""),
            (f"[ {score.band} ]", _band_style(score.band)),
        )
    )

    # Component breakdown
    comp_table = Table.grid(padding=(0, 2))
    comp_table.add_column(justify="left")
    comp_table.add_column(justify="right")
    comp_table.add_column()
    for cat in ("adherence", "attribution", "calibration", "ablation", "baseline"):
        if cat not in score.components:
            continue
        v = score.components[cat]
        comp_table.add_row(cat, f"{v:.2f}", _bar(v))
    c.print(comp_table)

    c.print()
    # Per-probe detail
    detail = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    detail.add_column("name", style="cyan")
    detail.add_column("kind", style="dim")
    detail.add_column("verdict")
    detail.add_column("score", justify="right")
    detail.add_column("raw", justify="right")
    detail.add_column("z", justify="right")
    detail.add_column("note", style="dim")
    for r in suite.probes:
        detail.add_row(
            r.name,
            r.kind,
            Text(r.verdict.value, style=_VERDICT_STYLE[r.verdict]),
            f"{r.score:.2f}" if r.score is not None else "—",
            f"{r.raw:.3f}" if r.raw is not None else "—",
            f"{r.z_score:+.2f}σ" if r.z_score is not None else "—",
            (r.message[:80] + "…") if len(r.message) > 80 else r.message,
        )
    c.print(detail)

    if score.findings:
        c.print()
        c.print(Text("top findings:", style="bold"))
        for i, f in enumerate(score.findings, start=1):
            c.print(f"  {i}. {f}")

    c.print()
    c.print(Text(f"wall: {suite.wall_seconds:.2f}s  |  sway {suite.sway_version}", style="dim"))


def to_json(suite: SuiteResult, score: SwayScore) -> str:
    """Serialize the suite + composite score as JSON.

    Stable schema; downstream tools rely on it. Breaking changes bump a
    ``schema_version`` field (not yet present — this is v0.1).
    """
    return json.dumps(_to_jsonable(suite, score), indent=2, sort_keys=True)


def _to_jsonable(suite: SuiteResult, score: SwayScore) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sway_version": suite.sway_version,
        "spec_path": suite.spec_path,
        "base_model_id": suite.base_model_id,
        "adapter_id": suite.adapter_id,
        "started_at": suite.started_at.isoformat(),
        "finished_at": suite.finished_at.isoformat(),
        "wall_seconds": suite.wall_seconds,
        "score": {
            "overall": score.overall,
            "band": score.band,
            "components": score.components,
            "weights": score.weights,
            "findings": list(score.findings),
        },
        "null_stats": suite.null_stats,
        "probes": [_probe_to_jsonable(p) for p in suite.probes],
    }


def _probe_to_jsonable(r: ProbeResult) -> dict[str, Any]:
    return {
        "name": r.name,
        "kind": r.kind,
        "verdict": r.verdict.value,
        "score": r.score,
        "raw": r.raw,
        "z_score": r.z_score,
        "base_value": r.base_value,
        "ft_value": r.ft_value,
        "evidence": r.evidence,
        "message": r.message,
        "duration_s": r.duration_s,
    }


def to_junit(suite: SuiteResult, score: SwayScore) -> str:
    """Serialize as JUnit XML. One ``<testcase>`` per probe."""
    testsuite = ET.Element(
        "testsuite",
        {
            "name": "dlm-sway",
            "tests": str(len(suite.probes)),
            "failures": str(sum(1 for p in suite.probes if p.verdict == Verdict.FAIL)),
            "errors": str(sum(1 for p in suite.probes if p.verdict == Verdict.ERROR)),
            "skipped": str(sum(1 for p in suite.probes if p.verdict == Verdict.SKIP)),
            "time": f"{suite.wall_seconds:.3f}",
        },
    )
    # Properties — the composite score and category breakdown.
    props = ET.SubElement(testsuite, "properties")
    ET.SubElement(props, "property", {"name": "overall", "value": f"{score.overall:.4f}"})
    ET.SubElement(props, "property", {"name": "band", "value": score.band})
    for cat, v in score.components.items():
        ET.SubElement(props, "property", {"name": f"component.{cat}", "value": f"{v:.4f}"})

    for r in suite.probes:
        tc = ET.SubElement(
            testsuite,
            "testcase",
            {"classname": r.kind, "name": r.name, "time": f"{r.duration_s:.3f}"},
        )
        if r.verdict == Verdict.FAIL:
            ET.SubElement(tc, "failure", {"message": r.message or "failed"})
        elif r.verdict == Verdict.ERROR:
            ET.SubElement(tc, "error", {"message": r.message or "errored"})
        elif r.verdict == Verdict.SKIP:
            ET.SubElement(tc, "skipped", {"message": r.message or "skipped"})

    return ET.tostring(testsuite, encoding="unicode")


def to_markdown(suite: SuiteResult, score: SwayScore) -> str:
    """A portable, CI-friendly markdown report."""
    buf = StringIO()
    buf.write("# dlm-sway report\n\n")
    buf.write(f"**Overall:** {score.overall:.2f} (`{score.band}`)  \n")
    buf.write(f"**Base:** `{suite.base_model_id}`  \n")
    buf.write(f"**Adapter:** `{_adapter_label(suite.adapter_id)}`  \n")
    buf.write(f"**Wall:** {suite.wall_seconds:.2f}s  \n\n")

    buf.write("## Components\n\n")
    buf.write("| category | score |\n|---|---:|\n")
    for cat, v in score.components.items():
        buf.write(f"| {cat} | {v:.2f} |\n")
    buf.write("\n## Probes\n\n")
    buf.write("| name | kind | verdict | score | note |\n|---|---|---|---:|---|\n")
    for r in suite.probes:
        buf.write(
            f"| {r.name} | `{r.kind}` | {r.verdict.value} | "
            f"{f'{r.score:.2f}' if r.score is not None else '—'} | "
            f"{r.message[:60]} |\n"
        )
    if score.findings:
        buf.write("\n## Top findings\n\n")
        for f in score.findings:
            buf.write(f"- {f}\n")
    return buf.getvalue()


# -- helpers -----------------------------------------------------------


def _adapter_label(adapter_id: str) -> str:
    if not adapter_id:
        return "(base only)"
    # Only the trailing path chunk is useful in the header.
    parts = adapter_id.rstrip("/").split("/")
    return "/".join(parts[-3:]) if len(parts) > 3 else adapter_id


def _score_style(v: float) -> str:
    if v >= 0.6:
        return "bold green"
    if v >= 0.3:
        return "bold yellow"
    return "bold red"


def _band_style(band: str) -> str:
    return {
        "noise": "red",
        "partial": "yellow",
        "healthy": "green",
        "suspicious": "magenta",
    }.get(band, "white")


def _bar(v: float, *, width: int = 10) -> str:
    clamped = max(0.0, min(1.0, v))
    filled = int(round(clamped * width))
    return "█" * filled + "░" * (width - filled)


__all__ = ["to_terminal", "to_json", "to_junit", "to_markdown"]
