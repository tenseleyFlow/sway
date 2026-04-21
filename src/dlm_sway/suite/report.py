"""Report emitters: terminal (rich), JSON, JUnit XML, markdown.

The terminal renderer is the one a user sees; it's the product surface.
It must communicate the verdict *and* the supporting evidence without
forcing the user to open the JSON.

JSON is the machine-readable source of truth — same fields as the
:class:`SuiteResult` dataclass but flattened for easy downstream parsing
(dashboards, diff tools, history tracking).

JUnit XML exists to drop into CI pipelines so ``sway gate``
integrates with existing test dashboards with no extra glue.
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from io import StringIO
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from dlm_sway.core.result import ProbeResult, SuiteResult, SwayScore, Verdict
from dlm_sway.probes._zscore import format_z_profile

_VERDICT_STYLE = {
    Verdict.PASS: "bold green",
    Verdict.FAIL: "bold red",
    Verdict.WARN: "bold yellow",
    Verdict.SKIP: "dim",
    Verdict.ERROR: "bold magenta",
}

#: Sentinel character all renderers use for "no numeric value available."
#: Single source prevents drift between surfaces (terminal vs markdown
#: vs JSON downstream consumers that copy the rendered strings).
_NONE_GLYPH = "—"


# -- unified number formatters (S06.10) --------------------------------
#
# Every surface that prints a number routes through one of these. A
# tests snapshot that locks report output catches any drift before it
# ships. Typing is wide (``float | int | None``) so callers don't have
# to special-case ``None`` at every site.


def format_score(v: float | int | None) -> str:
    """Two-decimal score, ``—`` when missing or non-finite."""
    if v is None or not math.isfinite(float(v)):
        return _NONE_GLYPH
    return f"{float(v):.2f}"


def format_raw(v: float | int | None) -> str:
    """Three-decimal raw metric, ``—`` when missing or non-finite.

    Uses thousands separators at magnitude ≥ 1 000 so half-life outputs
    from ``prompt_collapse`` don't render as ``1945.473`` (hard to eyeball).
    """
    if v is None or not math.isfinite(float(v)):
        return _NONE_GLYPH
    return f"{float(v):,.3f}"


def format_z(v: float | int | None) -> str:
    """Signed z-score with ``σ`` suffix and thousands separator, ``—`` on None."""
    if v is None or not math.isfinite(float(v)):
        return _NONE_GLYPH
    return f"{float(v):+,.2f}σ"


def format_ci(ci: tuple[float, float] | None) -> str:
    """Percentile-bootstrap 95% CI as ``[lo, hi]``; ``—`` on None / non-finite."""
    if ci is None:
        return _NONE_GLYPH
    lo, hi = ci
    if not (math.isfinite(float(lo)) and math.isfinite(float(hi))):
        return _NONE_GLYPH
    return f"[{float(lo):.3f}, {float(hi):.3f}]"


def _message_with_rank_profile(r: ProbeResult) -> str:
    """Append the per-rank z-profile to a probe's message when present.

    Renders as ``"<message> | rank profile: +4.2σ @ 1x / +6.8σ @ 0.5x"``.
    When the probe didn't run under multi-rank calibration (``z_by_rank``
    is ``None`` or has a single rank), returns the message unchanged.
    """
    base = r.message or ""
    z_by_rank = r.evidence.get("z_by_rank")
    if not z_by_rank or len(z_by_rank) < 2:
        return base
    profile = format_z_profile(z_by_rank)
    if not profile:
        return base
    return f"{base} | rank profile: {profile}" if base else f"rank profile: {profile}"


def format_duration_s(v: float | int | None) -> str:
    """Wall-time display. ``1.23s`` for sub-second, ``12.3s`` above 10, ``—`` on None."""
    if v is None or not math.isfinite(float(v)):
        return _NONE_GLYPH
    f = float(v)
    if f < 10.0:
        return f"{f:.2f}s"
    if f < 100.0:
        return f"{f:.1f}s"
    return f"{f:,.0f}s"


# -- extras-rollup helpers (S06.6) -------------------------------------

_MISSING_EXTRA_RE = re.compile(r"install the \[([^\]]+)\] extra", re.IGNORECASE)


def collect_missing_extras(suite: SuiteResult) -> list[str]:
    """Parse SKIP messages for ``install the [X] extra`` hints.

    Returns a deduplicated, sorted list of extra names that would
    unskip probes. ``BackendNotAvailableError`` formats messages with
    ``install the [<extra>] extra`` so we can lift them out without
    wiring a new field through.
    """
    found: set[str] = set()
    for p in suite.probes:
        if p.verdict != Verdict.SKIP or not p.message:
            continue
        for match in _MISSING_EXTRA_RE.finditer(p.message):
            found.add(match.group(1))
    return sorted(found)


def collect_degenerate_null_kinds(suite: SuiteResult) -> list[str]:
    """Probe kinds whose null-calibration stats were flagged degenerate.

    ``null_adapter`` marks a kind's stats with ``degenerate: 1.0`` when
    the calibration ran but the baseline was too narrow for the z-score
    path to fire (``runs: 1``, or a multi-seed run whose raws collapsed
    to an effectively-zero variance — F02 from Audit 03). Unlike
    :func:`collect_null_opt_outs` (which surfaces probes that opted
    out at spec-build time), this surface catches the case where the
    null *did* run but wasn't useful. Both cases fall back to fixed
    thresholds; the report distinguishes them so users can act:
    ``opt_out`` → expected for probes like ``adapter_revert``;
    ``degenerate`` → bump ``runs:`` in the spec.
    """
    found: set[str] = set()
    for probe in suite.probes:
        if probe.kind != "null_adapter":
            continue
        # ``null_adapter`` writes per-kind stats into
        # ``SuiteResult.null_stats``, not the probe's evidence — the
        # suite-level field is the canonical place the runner threads
        # calibration across probes.
        stats_by_kind = suite.null_stats or {}
        for kind, kind_stats in stats_by_kind.items():
            if not isinstance(kind_stats, dict):
                continue
            if kind_stats.get("degenerate", 0.0) >= 0.5:
                found.add(kind)
    return sorted(found)


def collect_null_opt_outs(suite: SuiteResult) -> list[str]:
    """Probe kinds that opted out of null calibration.

    ``null_adapter`` publishes ``evidence["skipped_kinds"]`` with the
    probe kinds whose ``calibrate_spec`` returned ``None`` (e.g.
    ``adapter_revert`` — no embedder on the null proxy;
    ``prompt_collapse`` — noise can't fit an exponential decay).
    Returns a deduplicated, sorted list of those kinds, or an empty
    list when no null_adapter ran in the suite.
    """
    found: set[str] = set()
    for p in suite.probes:
        if p.kind != "null_adapter":
            continue
        skipped = p.evidence.get("skipped_kinds")
        if not skipped:
            continue
        for kind in skipped:
            if isinstance(kind, str):
                found.add(kind)
    return sorted(found)


def to_terminal(suite: SuiteResult, score: SwayScore, *, console: Console | None = None) -> None:
    """Render the report to a rich Console (stdout by default)."""
    c = console or Console()

    header = Text.assemble(
        ("sway report — ", "bold"),
        (suite.base_model_id, "cyan"),
        ("  vs  ", "dim"),
        (_adapter_label(suite.adapter_id), "cyan"),
    )
    c.print(Panel(header, expand=False, border_style="blue"))

    c.print()
    c.print(
        Text.assemble(
            ("overall: ", "bold"),
            (format_score(score.overall), _score_style(score.overall)),
            ("  ", ""),
            (f"[ {score.band} ]", _band_style(score.band)),
        )
    )

    # Component breakdown. Order matches ``DEFAULT_COMPONENT_WEIGHTS``
    # (the extensibility point) and appends any categories present in
    # ``score.components`` but not in the default weights — so a custom
    # Probe subclass with a new category still renders.
    comp_table = Table.grid(padding=(0, 2))
    comp_table.add_column(justify="left")
    comp_table.add_column(justify="right")
    comp_table.add_column()
    comp_table.add_column(style="dim")
    for cat in _category_order(score):
        if cat not in score.components:
            continue
        v = score.components[cat]
        weight = score.weights.get(cat, 0.0)
        # S03 / B18: a zero-weight category contributes nothing to the
        # composite; label explicitly so users don't mistake the visible
        # bar for judgment.
        label = "(informational, weight=0)" if weight == 0.0 else ""
        comp_table.add_row(cat, format_score(v), _bar(v), label)
    c.print(comp_table)

    c.print()
    # Per-probe detail.
    detail = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    detail.add_column("name", style="cyan")
    detail.add_column("kind", style="dim")
    detail.add_column("verdict")
    detail.add_column("score", justify="right")
    detail.add_column("raw", justify="right")
    detail.add_column("ci95", justify="right", style="dim")
    detail.add_column("z", justify="right")
    # D15: let Rich wrap long messages instead of hard-truncating at 80
    # chars with an ellipsis. ``overflow="fold"`` + ``no_wrap=False``
    # preserves the full text across multiple terminal lines.
    detail.add_column("note", style="dim", overflow="fold", no_wrap=False)
    for r in suite.probes:
        detail.add_row(
            r.name,
            r.kind,
            Text(r.verdict.value, style=_VERDICT_STYLE[r.verdict]),
            format_score(r.score),
            format_raw(r.raw),
            format_ci(r.ci_95),
            format_z(r.z_score),
            Text(_message_with_rank_profile(r)),
        )
    c.print(detail)

    if score.findings:
        c.print()
        c.print(Text("top findings:", style="bold"))
        for i, f in enumerate(score.findings, start=1):
            c.print(f"  {i}. {f}")

    # D3: missing-extras rollup. When probes SKIPped because their
    # backend extras aren't installed, collapse the hints into one
    # actionable footer rather than forcing the user to scan per-row.
    extras = collect_missing_extras(suite)
    if extras:
        c.print()
        skipped_ct = sum(1 for p in suite.probes if p.verdict == Verdict.SKIP)
        c.print(
            Text(
                f"{skipped_ct} probe(s) skipped due to missing extras: "
                f"pip install 'dlm-sway[{','.join(extras)}]'",
                style="dim",
            )
        )

    # F15: null-calibration opt-outs rollup. Probes whose
    # ``calibrate_spec`` returns ``None`` fall back to fixed-threshold
    # verdicts. Surface the list in the footer so users understand
    # why those rows read ``(no calibration)`` in the message column.
    opt_outs = collect_null_opt_outs(suite)
    if opt_outs:
        c.print()
        c.print(
            Text(
                f"{len(opt_outs)} probe(s) opted out of null calibration "
                f"(using fixed thresholds): {', '.join(opt_outs)}",
                style="dim",
            )
        )

    # F02 (Audit 03): null-calibration-degenerate rollup. Distinct from
    # opt-outs — the null *did* run, but its baseline was too narrow
    # (``runs: 1`` or coincidentally-identical seeds). Users see this
    # and bump ``runs:`` in the spec; the fix is actionable.
    degenerate = collect_degenerate_null_kinds(suite)
    if degenerate:
        c.print()
        c.print(
            Text(
                f"{len(degenerate)} probe kind(s) had a degenerate null "
                f"baseline (std ≈ 0, insufficient for z-scoring): "
                f"{', '.join(degenerate)} — bump ``runs:`` in null_adapter spec.",
                style="dim",
            )
        )

    c.print()
    footer_parts = [f"wall: {format_duration_s(suite.wall_seconds)}", f"sway {suite.sway_version}"]
    if suite.determinism is not None:
        footer_parts.append(f"det: {suite.determinism.class_} (seed={suite.determinism.seed})")
    cache_line = _cache_line(suite)
    if cache_line is not None:
        footer_parts.append(cache_line)
    c.print(Text("  |  ".join(footer_parts), style="dim"))


def to_json(suite: SuiteResult, score: SwayScore) -> str:
    """Serialize the suite + composite score as JSON.

    Stable schema; downstream tools rely on it. Breaking changes bump a
    ``schema_version`` field (not yet present — this is v0.1).
    """
    return json.dumps(_to_jsonable(suite, score), indent=2, sort_keys=True)


def _to_jsonable(suite: SuiteResult, score: SwayScore) -> dict[str, Any]:
    determinism: dict[str, Any] | None = None
    if suite.determinism is not None:
        determinism = {
            "class": suite.determinism.class_,
            "seed": suite.determinism.seed,
            "notes": list(suite.determinism.notes),
        }
    return {
        "schema_version": 1,
        "sway_version": suite.sway_version,
        "spec_path": suite.spec_path,
        "base_model_id": suite.base_model_id,
        "adapter_id": suite.adapter_id,
        "started_at": suite.started_at.isoformat(),
        "finished_at": suite.finished_at.isoformat(),
        "wall_seconds": suite.wall_seconds,
        "determinism": determinism,
        "backend_stats": dict(suite.backend_stats) if suite.backend_stats else {},
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
        # S14: bootstrap 95% CI on ``raw``. Serialized as a two-list
        # [lo, hi] so JSON stays tuple-free (match numpy convention).
        "ci_95": list(r.ci_95) if r.ci_95 is not None else None,
    }


def from_json(raw: dict[str, Any]) -> tuple[SuiteResult, SwayScore]:
    """Reconstruct a ``(SuiteResult, SwayScore)`` pair from saved JSON.

    Inverse of :func:`to_json` for the fields the renderers consume.
    Missing fields are tolerated — older snapshots predate
    ``determinism`` and ``schema_version`` — so this helper stays
    backward-compatible by default. ``sway report --format X`` uses
    this so all four formats (terminal / md / junit / json) flow
    through the same renderers as a fresh ``sway run`` (B16).
    """
    from datetime import datetime

    from dlm_sway.core.result import (
        DEFAULT_COMPONENT_WEIGHTS,
        DeterminismReport,
        ProbeResult,
        SuiteResult,
        SwayScore,
        Verdict,
    )

    def _ts(s: str | None) -> datetime:
        if s:
            return datetime.fromisoformat(s)
        # Snapshots that predate the field — give the renderer a
        # well-defined zero so wall-time displays as 0.00s.
        return datetime.fromtimestamp(0).astimezone()

    def _ci_95(v: Any) -> tuple[float, float] | None:
        if v is None:
            return None
        try:
            lo, hi = v
            return (float(lo), float(hi))
        except (TypeError, ValueError):
            return None

    probes = tuple(
        ProbeResult(
            name=p["name"],
            kind=p["kind"],
            verdict=Verdict(p["verdict"]),
            score=p.get("score"),
            raw=p.get("raw"),
            z_score=p.get("z_score"),
            base_value=p.get("base_value"),
            ft_value=p.get("ft_value"),
            evidence=dict(p.get("evidence") or {}),
            message=p.get("message", ""),
            duration_s=float(p.get("duration_s", 0.0)),
            ci_95=_ci_95(p.get("ci_95")),
        )
        for p in raw.get("probes", [])
    )

    determinism: DeterminismReport | None = None
    det_raw = raw.get("determinism")
    if isinstance(det_raw, dict):
        determinism = DeterminismReport(
            class_=det_raw.get("class", "best_effort"),
            seed=int(det_raw.get("seed", 0)),
            notes=tuple(det_raw.get("notes") or ()),
        )

    suite = SuiteResult(
        spec_path=raw.get("spec_path", ""),
        started_at=_ts(raw.get("started_at")),
        finished_at=_ts(raw.get("finished_at")),
        base_model_id=raw.get("base_model_id", ""),
        adapter_id=raw.get("adapter_id", ""),
        sway_version=raw.get("sway_version", "?"),
        probes=probes,
        null_stats=dict(raw.get("null_stats") or {}),
        determinism=determinism,
        backend_stats=dict(raw.get("backend_stats") or {}),
    )

    score_raw: dict[str, Any] = raw.get("score") or {}
    score = SwayScore(
        overall=float(score_raw.get("overall", 0.0)),
        components=dict(score_raw.get("components") or {}),
        weights=dict(score_raw.get("weights") or DEFAULT_COMPONENT_WEIGHTS),
        band=score_raw.get("band", ""),
        findings=tuple(score_raw.get("findings") or ()),
    )
    return suite, score


def to_junit(suite: SuiteResult, score: SwayScore) -> str:
    """Serialize as JUnit XML. One ``<testcase>`` per probe."""
    testsuite = ET.Element(
        "testsuite",
        {
            "name": "sway",
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
    """A portable, CI-friendly markdown report.

    The single source of the markdown emit (B16): both
    ``sway run --markdown`` and ``sway report --format md`` route
    through this function. No second ``_render_markdown_from_json``.
    """
    buf = StringIO()
    buf.write("# sway report\n\n")
    buf.write(f"**Overall:** {format_score(score.overall)} (`{score.band}`)  \n")
    buf.write(f"**Base:** `{suite.base_model_id}`  \n")
    buf.write(f"**Adapter:** `{_adapter_label(suite.adapter_id)}`  \n")
    buf.write(f"**Wall:** {format_duration_s(suite.wall_seconds)}  \n")
    if suite.determinism is not None:
        buf.write(
            f"**Determinism:** `{suite.determinism.class_}` (seed={suite.determinism.seed})  \n"
        )
    cache_line = _cache_line(suite)
    if cache_line is not None:
        buf.write(f"**Backend:** {cache_line}  \n")
    buf.write("\n")

    buf.write("## Components\n\n")
    buf.write("| category | score | weight | |\n|---|---:|---:|---|\n")
    for cat in _category_order(score):
        if cat not in score.components:
            continue
        v = score.components[cat]
        weight = score.weights.get(cat, 0.0)
        label = "(informational, weight=0)" if weight == 0.0 else ""
        buf.write(f"| {cat} | {format_score(v)} | {format_score(weight)} | {label} |\n")

    # D9: markdown must reach parity with the terminal table — raw,
    # z_score, duration_s all shown. Findings are appended as a section
    # below so CI log consumers can see them without opening the JSON.
    buf.write("\n## Probes\n\n")
    buf.write(
        "| name | kind | verdict | score | raw | ci95 | z | duration | note |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---|\n"
    )
    for r in suite.probes:
        # Escape pipes in messages so markdown doesn't treat them as
        # column separators. Leading/trailing whitespace collapsed.
        note = _message_with_rank_profile(r).replace("|", "\\|").replace("\n", " ").strip()
        buf.write(
            f"| {r.name} | `{r.kind}` | {r.verdict.value} | "
            f"{format_score(r.score)} | {format_raw(r.raw)} | "
            f"{format_ci(r.ci_95)} | {format_z(r.z_score)} | "
            f"{format_duration_s(r.duration_s)} | {note} |\n"
        )

    if score.findings:
        buf.write("\n## Top findings\n\n")
        for f in score.findings:
            buf.write(f"- {f}\n")

    # D3: missing-extras rollup.
    extras = collect_missing_extras(suite)
    if extras:
        skipped_ct = sum(1 for p in suite.probes if p.verdict == Verdict.SKIP)
        buf.write("\n## Skipped probes\n\n")
        buf.write(f"{skipped_ct} probe(s) skipped due to missing extras. Install with:\n\n")
        buf.write(f"```\npip install 'dlm-sway[{','.join(extras)}]'\n```\n")

    # F15: null-calibration opt-outs rollup.
    opt_outs = collect_null_opt_outs(suite)
    if opt_outs:
        buf.write("\n## Null-calibration opt-outs\n\n")
        buf.write(
            f"{len(opt_outs)} probe(s) fall back to fixed thresholds because "
            f"their `calibrate_spec` returns `None`:\n\n"
        )
        for kind in opt_outs:
            buf.write(f"- `{kind}`\n")

    # F02 (Audit 03) — degenerate null-calibration rollup.
    degenerate = collect_degenerate_null_kinds(suite)
    if degenerate:
        buf.write("\n## Degenerate null calibration\n\n")
        buf.write(
            f"{len(degenerate)} probe kind(s) ran null_adapter but the "
            f"resulting baseline was too narrow for z-scoring "
            f"(std ≈ 0, typically `runs: 1` or coincidentally-matched "
            f"seeds). Fix: bump `runs:` in the `null_adapter` spec "
            f"entry. Affected kinds:\n\n"
        )
        for kind in degenerate:
            buf.write(f"- `{kind}`\n")

    # F07 — cluster_kl sub-line: expand the per-cluster breakdown so
    # the reader can answer "which topic moved?" without cracking open
    # the JSON. The row itself already carries ``k=N, spec=X.XX`` in
    # the message; this section adds the per-cluster mean KL + top
    # exemplars.
    ck_probes = [p for p in suite.probes if p.kind == "cluster_kl" and p.evidence]
    if ck_probes:
        buf.write("\n## Cluster breakdown (cluster_kl)\n\n")
        for p in ck_probes:
            per_cluster = p.evidence.get("per_cluster_mean_kl", [])
            sizes = p.evidence.get("per_cluster_size", [])
            exemplars = p.evidence.get("cluster_exemplars", [])
            buf.write(f"### `{p.name}`\n\n")
            buf.write("| cluster | size | mean KL | exemplars |\n")
            buf.write("|---:|---:|---:|---|\n")
            for i, (mean, size, ex) in enumerate(zip(per_cluster, sizes, exemplars, strict=False)):
                mean_str = "—" if not isinstance(mean, int | float) else f"{mean:.3f}"
                ex_str = "; ".join(e.replace("|", "\\|") for e in (ex or [])) or "—"
                buf.write(f"| {i} | {size} | {mean_str} | {ex_str} |\n")
            buf.write("\n")

    return buf.getvalue()


# -- helpers -----------------------------------------------------------


def _category_order(score: SwayScore) -> list[str]:
    """Unified render order for component categories.

    Falls back through two sources, in priority order:

    1. Keys of :data:`core.result.DEFAULT_COMPONENT_WEIGHTS` — the
       canonical category list every first-party probe slots into.
    2. Any category present in ``score.components`` that isn't in the
       default weights — so a custom :class:`Probe` subclass declaring
       a brand-new category still renders (F16).

    Keeps the renderer loop in terminal + markdown identical so future
    additions flow through both surfaces without a second code path.
    """
    from dlm_sway.core.result import DEFAULT_COMPONENT_WEIGHTS

    order: list[str] = list(DEFAULT_COMPONENT_WEIGHTS.keys())
    order.extend(cat for cat in score.components if cat not in DEFAULT_COMPONENT_WEIGHTS)
    return order


def _cache_line(suite: SuiteResult) -> str | None:
    """Format the cache-hit-rate footer line, or ``None`` when no stats."""
    stats = suite.backend_stats
    if not stats:
        return None
    hits = int(stats.get("cache_hits", 0))
    misses = int(stats.get("cache_misses", 0))
    total = hits + misses
    if total == 0:
        return None
    pct = 100.0 * hits / total
    return f"cache: {hits}/{total} = {pct:.0f}%"


def _adapter_label(adapter_id: str) -> str:
    """Truncate the adapter path for display; quote when whitespace is present.

    D14: a path containing spaces (``/Users/me/My Adapters/v1``) was
    rendering ambiguously in the header. Quote it whenever any
    whitespace appears so the trailing path is unmistakable.
    """
    if not adapter_id:
        return "(base only)"
    parts = adapter_id.rstrip("/").split("/")
    label = "/".join(parts[-3:]) if len(parts) > 3 else adapter_id
    if any(ch.isspace() for ch in label):
        # Use double quotes so the result drops cleanly into a CLI
        # invocation if a user copy-pastes it.
        return f'"{label}"'
    return label


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


__all__ = [
    "collect_degenerate_null_kinds",
    "collect_missing_extras",
    "collect_null_opt_outs",
    "format_duration_s",
    "format_raw",
    "format_score",
    "format_z",
    "from_json",
    "to_json",
    "to_junit",
    "to_markdown",
    "to_terminal",
]
