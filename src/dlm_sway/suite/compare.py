"""Cross-run comparison of saved ``sway run --json`` outputs (S11 / F5).

Answers the CI question: *"did last night's run regress anything vs
the baseline?"* The user hands :func:`build_matrix` a sequence of
``(SuiteResult, SwayScore)`` pairs — typically rehydrated from JSON
files via :func:`dlm_sway.suite.report.from_json` — and gets back a
:class:`CompareMatrix` whose cells line up probe-by-probe across runs.
Probes that were added, renamed, or removed between runs show as
``None`` cells so a renderer can mark them with the em-dash sentinel
every sway surface uses for "no value" (S06.10).

This module deliberately owns no IO: the CLI reads the files and
``report.from_json`` rehydrates them; this module just does the
matrix/delta math and the rendering. That separation keeps
``build_matrix`` unit-testable without touching the filesystem.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from io import StringIO
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from dlm_sway.core.result import SuiteResult, SwayScore

#: Matches the em-dash glyph every sway surface uses for "no value."
#: Imported from :mod:`report` via :func:`format_score` / :func:`format_z`,
#: but we also need the literal when rendering the *deltas* row (where
#: ``None`` means "no prior run to diff against").
_NONE_GLYPH = "—"

#: Default threshold for ``--fail-on-regression``: a probe whose score
#: dropped by ≥ this value between the newest run and the previous one
#: counts as a regression. Same scale as ``format_score`` (0..1).
DEFAULT_REGRESSION_THRESHOLD: float = 0.10


@dataclass(frozen=True, slots=True)
class CompareMatrix:
    """Cross-run score matrix produced by :func:`build_matrix`.

    Attributes
    ----------
    labels:
        Human-readable label per run, in the order they were passed in
        (the CLI forwards filenames or timestamps). Column headers in
        every renderer.
    timestamps:
        ``finished_at`` ISO-8601 strings per run. Shown in the markdown
        and JSON renderings as metadata — terminal output prefers the
        shorter ``labels`` for column headers.
    probe_names:
        Union of probe names across every run, sorted alphabetically so
        the row order is deterministic across runs (it isn't tied to any
        one run's spec order).
    scores:
        ``name → [score_per_run]``. A cell is ``None`` when the probe
        didn't appear in that run (added later, removed, renamed).
    deltas:
        ``name → [delta_i]`` where ``delta_i = scores[i] - scores[i-1]``.
        ``deltas`` has ``len(labels) - 1`` entries per probe; cells are
        ``None`` when either neighbor is ``None`` or the delta is
        non-finite.
    composite_series:
        Per-run overall ``SwayScore.overall``. Parallel to ``labels``.
    """

    labels: tuple[str, ...]
    timestamps: tuple[str, ...]
    probe_names: tuple[str, ...]
    scores: dict[str, list[float | None]] = field(default_factory=dict)
    deltas: dict[str, list[float | None]] = field(default_factory=dict)
    composite_series: list[float | None] = field(default_factory=list)

    @property
    def n_runs(self) -> int:
        return len(self.labels)

    def latest_regressions(self, threshold: float) -> list[tuple[str, float]]:
        """Probes whose newest-run score dropped ≥ ``threshold`` vs the prior run.

        Returns an empty list when there are fewer than 2 runs (no prior
        to compare) or when no probe regressed that hard. Each entry is
        ``(probe_name, delta)`` with ``delta <= -threshold``. Sorted by
        most severe (most negative delta) first.
        """
        if self.n_runs < 2 or threshold <= 0.0:
            return []
        out: list[tuple[str, float]] = []
        for name in self.probe_names:
            series = self.deltas.get(name, [])
            if not series:
                continue
            last = series[-1]
            if last is None:
                continue
            if last <= -threshold:
                out.append((name, last))
        out.sort(key=lambda pair: pair[1])
        return out


def build_matrix(
    results: list[tuple[SuiteResult, SwayScore]],
    *,
    labels: list[str] | None = None,
) -> CompareMatrix:
    """Fold an N-run history into a :class:`CompareMatrix`.

    ``labels`` let the CLI pass filenames or short identifiers; when
    omitted we fall back to ``finished_at`` timestamps. The order of
    ``results`` is the column order — runs are not sorted.
    """
    if not results:
        raise ValueError("build_matrix requires at least one run")

    label_tuple: tuple[str, ...]
    if labels is not None:
        if len(labels) != len(results):
            raise ValueError(f"labels length {len(labels)} != results length {len(results)}")
        label_tuple = tuple(labels)
    else:
        label_tuple = tuple(
            (s.finished_at.isoformat() if s.finished_at else f"run-{i}")
            for i, (s, _) in enumerate(results)
        )

    timestamp_tuple = tuple(
        (s.finished_at.isoformat() if s.finished_at else "") for s, _ in results
    )

    # Per-run map: probe name → score. Missing probe in a run → absent
    # key. The outer matrix fills with None for those.
    per_run_scores: list[dict[str, float | None]] = []
    for suite, _score in results:
        run_map: dict[str, float | None] = {}
        for p in suite.probes:
            run_map[p.name] = (
                float(p.score) if p.score is not None and math.isfinite(p.score) else None
            )
        per_run_scores.append(run_map)

    # Union of probe names across every run — sorted so row order is
    # stable across invocations regardless of which run appears first.
    union_names = sorted({name for run in per_run_scores for name in run})

    scores: dict[str, list[float | None]] = {
        name: [run.get(name) for run in per_run_scores] for name in union_names
    }

    # Delta series: scores[i] - scores[i-1] per probe, guarded against
    # None neighbors. First delta index is 0 (between run 0 and run 1).
    deltas: dict[str, list[float | None]] = {}
    for name, series in scores.items():
        row: list[float | None] = []
        for i in range(1, len(series)):
            prev = series[i - 1]
            cur = series[i]
            if prev is None or cur is None:
                row.append(None)
                continue
            delta = cur - prev
            row.append(delta if math.isfinite(delta) else None)
        deltas[name] = row

    composite_series: list[float | None] = [
        (
            float(score.overall)
            if score is not None and math.isfinite(float(score.overall))
            else None
        )
        for _, score in results
    ]

    return CompareMatrix(
        labels=label_tuple,
        timestamps=timestamp_tuple,
        probe_names=tuple(union_names),
        scores=scores,
        deltas=deltas,
        composite_series=composite_series,
    )


def _format_cell(v: float | None) -> str:
    """Score cell: two decimals with em-dash for None."""
    return _NONE_GLYPH if v is None else f"{v:.2f}"


def _format_delta(v: float | None) -> str:
    """Delta cell: signed two decimals, em-dash for None, explicit sign."""
    if v is None:
        return _NONE_GLYPH
    if abs(v) < 5e-5:
        return "0.00"
    return f"{v:+.2f}"


def _delta_style(v: float | None, threshold: float) -> str:
    """Rich style for a delta cell. Red on regression, green on improvement."""
    if v is None:
        return "dim"
    if v <= -threshold:
        return "bold red"
    if v >= threshold:
        return "bold green"
    return "dim"


def render_terminal(
    matrix: CompareMatrix,
    *,
    console: Console | None = None,
    regression_threshold: float = DEFAULT_REGRESSION_THRESHOLD,
) -> None:
    """Rich-formatted terminal output. One row per probe + composite + deltas."""
    c = console or Console()

    c.print(Text(f"sway compare — {matrix.n_runs} runs", style="bold"))
    c.print()

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("probe", style="cyan")
    for label in matrix.labels:
        table.add_column(label, justify="right")
    for i in range(matrix.n_runs - 1):
        table.add_column(
            f"Δ {matrix.labels[i]}→{matrix.labels[i + 1]}", justify="right", style="dim"
        )

    for name in matrix.probe_names:
        row: list[str | Text] = [name]
        for v in matrix.scores[name]:
            row.append(_format_cell(v))
        for v in matrix.deltas[name]:
            row.append(Text(_format_delta(v), style=_delta_style(v, regression_threshold)))
        table.add_row(*row)

    # Composite row lives below the probes, separated by a divider the
    # terminal draws implicitly when the name style differs.
    composite_row: list[str | Text] = [Text("composite (overall)", style="bold")]
    for v in matrix.composite_series:
        composite_row.append(_format_cell(v))
    composite_deltas: list[float | None] = []
    for i in range(1, len(matrix.composite_series)):
        prev = matrix.composite_series[i - 1]
        cur = matrix.composite_series[i]
        composite_deltas.append(cur - prev if (prev is not None and cur is not None) else None)
    for v in composite_deltas:
        composite_row.append(Text(_format_delta(v), style=_delta_style(v, regression_threshold)))
    table.add_row(*composite_row)

    c.print(table)

    regressions = matrix.latest_regressions(regression_threshold)
    if regressions:
        c.print()
        c.print(
            Text(
                f"regressions (≥{regression_threshold:.2f} drop vs previous run):",
                style="bold red",
            )
        )
        for name, delta in regressions:
            c.print(f"  {name}: {delta:+.3f}")


def render_markdown(
    matrix: CompareMatrix,
    *,
    regression_threshold: float = DEFAULT_REGRESSION_THRESHOLD,
) -> str:
    """Markdown table — same content as the terminal, pipe-friendly."""
    buf = StringIO()
    buf.write(f"# sway compare — {matrix.n_runs} runs\n\n")

    if matrix.timestamps:
        buf.write("## Runs\n\n")
        buf.write("| label | finished_at |\n|---|---|\n")
        for label, ts in zip(matrix.labels, matrix.timestamps, strict=True):
            buf.write(f"| {label} | {ts or _NONE_GLYPH} |\n")
        buf.write("\n")

    buf.write("## Scores\n\n")
    header = ["probe"] + list(matrix.labels)
    for i in range(matrix.n_runs - 1):
        header.append(f"Δ {matrix.labels[i]}→{matrix.labels[i + 1]}")
    buf.write("| " + " | ".join(header) + " |\n")
    buf.write("|" + "|".join(["---"] * len(header)) + "|\n")

    for name in matrix.probe_names:
        cells = [name] + [_format_cell(v) for v in matrix.scores[name]]
        cells += [_format_delta(v) for v in matrix.deltas[name]]
        buf.write("| " + " | ".join(cells) + " |\n")

    # Composite row
    cells = ["**composite**"] + [_format_cell(v) for v in matrix.composite_series]
    composite_deltas: list[float | None] = []
    for i in range(1, len(matrix.composite_series)):
        prev = matrix.composite_series[i - 1]
        cur = matrix.composite_series[i]
        composite_deltas.append(cur - prev if (prev is not None and cur is not None) else None)
    cells += [_format_delta(v) for v in composite_deltas]
    buf.write("| " + " | ".join(cells) + " |\n")

    regressions = matrix.latest_regressions(regression_threshold)
    if regressions:
        buf.write(f"\n## Regressions (≥{regression_threshold:.2f} drop vs previous run)\n\n")
        for name, delta in regressions:
            buf.write(f"- **{name}** — `{delta:+.3f}`\n")

    return buf.getvalue()


def render_json(
    matrix: CompareMatrix,
    *,
    regression_threshold: float = DEFAULT_REGRESSION_THRESHOLD,
) -> str:
    """Machine-readable JSON. Same field names as :class:`CompareMatrix`."""
    payload: dict[str, Any] = {
        "labels": list(matrix.labels),
        "timestamps": list(matrix.timestamps),
        "probe_names": list(matrix.probe_names),
        "scores": {name: list(series) for name, series in matrix.scores.items()},
        "deltas": {name: list(series) for name, series in matrix.deltas.items()},
        "composite_series": list(matrix.composite_series),
        "regression_threshold": regression_threshold,
        "latest_regressions": [
            {"probe": name, "delta": delta}
            for name, delta in matrix.latest_regressions(regression_threshold)
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=False)


__all__ = [
    "CompareMatrix",
    "DEFAULT_REGRESSION_THRESHOLD",
    "build_matrix",
    "render_json",
    "render_markdown",
    "render_terminal",
]
