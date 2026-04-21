"""Interactive single-file HTML report (S12 / F6).

The terminal renderer is right for CI logs; markdown is right for
checked-in PR artifacts. Neither supports *exploration* — clicking
through per-section SIS bars, hovering over the ablation curve to
read exact λ values, zooming into the probe scatter. This module
produces a self-contained HTML page with four interactive Plotly
panels for the research / write-up case:

1. Composite score gauge + per-category breakdown.
2. Per-section SIS bar chart (when ``section_internalization`` ran).
3. Adapter-ablation response curve (when ``adapter_ablation`` ran).
4. All-probe score + z-score scatter with hover tooltips.

Plotly's JS bundle is inlined once in ``<head>``; each panel's
``<div>`` gets a stable id so snapshot tests don't churn on every
render. The output is typically ~3.6 MB (Plotly is ~3 MB of that)
and loads with zero network calls.

``plotly`` is an *optional* dependency shipped via the ``[viz]`` extra.
When it's not importable, :func:`to_html` raises ``RuntimeError``
with an install hint the CLI can surface.
"""

from __future__ import annotations

import html
from typing import Any

from dlm_sway.core.result import ProbeResult, SuiteResult, SwayScore, Verdict

#: Palette used across panels. Matches the terminal verdict colors so a
#: user scanning the HTML mapping back to the `sway run` output sees
#: the same color grammar.
_VERDICT_COLOR: dict[Verdict, str] = {
    Verdict.PASS: "#28a745",
    Verdict.FAIL: "#dc3545",
    Verdict.WARN: "#ffc107",
    Verdict.SKIP: "#6c757d",
    Verdict.ERROR: "#9c27b0",
}

#: Division colors for the category bars — same per-component palette
#: we publish in the README.
_CATEGORY_COLOR: dict[str, str] = {
    "adherence": "#0d6efd",
    "attribution": "#198754",
    "calibration": "#fd7e14",
    "ablation": "#6f42c1",
    "baseline": "#adb5bd",
}

#: Stable div IDs so the emitted HTML is byte-identical for the same
#: inputs. Plotly defaults to a random UUID per figure otherwise.
_DIV_GAUGE = "sway-gauge"
_DIV_CATEGORY = "sway-category"
_DIV_SIS = "sway-sis"
_DIV_ABLATION = "sway-ablation"
_DIV_SCATTER = "sway-scatter"


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def to_html(suite: SuiteResult, score: SwayScore) -> str:
    """Render a ``SuiteResult``/``SwayScore`` pair as a self-contained HTML page.

    Raises
    ------
    RuntimeError
        When ``plotly`` is not importable. The CLI catches this and
        surfaces an install hint (``pip install 'dlm-sway[viz]'``).
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        from plotly.offline import get_plotlyjs
    except ImportError as exc:
        raise RuntimeError(
            "plotly is required for --format html. Install with: pip install 'dlm-sway[viz]'"
        ) from exc

    gauge_fig = _gauge_figure(go, score)
    category_fig = _category_figure(go, score)
    sis_fig = _sis_figure(go, suite)
    ablation_fig = _ablation_figure(go, suite)
    scatter_fig = _scatter_figure(go, suite)

    panels: list[tuple[str, str, str]] = [
        ("Composite score", _DIV_GAUGE, _fig_to_div(pio, gauge_fig, _DIV_GAUGE)),
        ("Category breakdown", _DIV_CATEGORY, _fig_to_div(pio, category_fig, _DIV_CATEGORY)),
    ]
    if sis_fig is not None:
        panels.append(
            ("Per-section internalization", _DIV_SIS, _fig_to_div(pio, sis_fig, _DIV_SIS))
        )
    if ablation_fig is not None:
        panels.append(
            (
                "Adapter-ablation response",
                _DIV_ABLATION,
                _fig_to_div(pio, ablation_fig, _DIV_ABLATION),
            )
        )
    panels.append(
        ("Per-probe score vs. z-score", _DIV_SCATTER, _fig_to_div(pio, scatter_fig, _DIV_SCATTER))
    )

    return _assemble(suite, score, panels, plotly_js=get_plotlyjs())


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------


def _gauge_figure(go: Any, score: SwayScore) -> Any:
    """Indicator gauge for the composite score, 0..1 with banded thresholds."""
    overall = float(score.overall) if score.overall is not None else 0.0
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=overall,
            number={"valueformat": ".2f", "font": {"size": 48}},
            gauge={
                "axis": {"range": [0.0, 1.0]},
                "bar": {"color": _band_color(score.band)},
                "steps": [
                    {"range": [0.00, 0.30], "color": "#f8d7da"},  # noise
                    {"range": [0.30, 0.60], "color": "#fff3cd"},  # partial
                    {"range": [0.60, 0.85], "color": "#d1e7dd"},  # healthy
                    {"range": [0.85, 1.00], "color": "#e2e3ff"},  # suspicious
                ],
            },
            title={"text": f"<b>{score.band or 'unscored'}</b>"},
        ),
        layout=go.Layout(height=320, margin={"l": 20, "r": 20, "t": 60, "b": 20}),
    )


def _category_figure(go: Any, score: SwayScore) -> Any:
    """Horizontal bar chart of per-category contributions."""
    items = [(cat, float(v)) for cat, v in score.components.items()]
    items.sort(key=lambda pair: pair[0])
    labels = [cat for cat, _ in items]
    values = [v for _, v in items]
    colors = [_CATEGORY_COLOR.get(cat, "#888888") for cat in labels]
    return go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": colors},
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        ),
        layout=go.Layout(
            xaxis={"range": [0.0, 1.0], "title": "component score"},
            yaxis={"title": ""},
            height=260,
            margin={"l": 100, "r": 20, "t": 30, "b": 40},
        ),
    )


def _sis_figure(go: Any, suite: SuiteResult) -> Any | None:
    """Per-section internalization bar chart. ``None`` if no data."""
    probe = _first_probe_of_kind(suite, "section_internalization")
    if probe is None:
        return None
    per_section = probe.evidence.get("per_section")
    if not per_section or not isinstance(per_section, list):
        return None
    labels = [str(row.get("section_id") or row.get("tag") or "?") for row in per_section]
    values = [float(row.get("effective_sis", 0.0)) for row in per_section]
    passed = [bool(row.get("passed")) for row in per_section]
    colors = [_VERDICT_COLOR[Verdict.PASS if p else Verdict.FAIL] for p in passed]
    return go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker={"color": colors},
            hovertemplate="<b>%{x}</b><br>effective_sis=%{y:.3f}<extra></extra>",
        ),
        layout=go.Layout(
            xaxis={"title": "section"},
            yaxis={"title": "effective_sis (own - leak)"},
            height=320,
            margin={"l": 60, "r": 20, "t": 30, "b": 80},
        ),
    )


def _ablation_figure(go: Any, suite: SuiteResult) -> Any | None:
    """λ vs. divergence response curve. ``None`` if ablation didn't run."""
    probe = _first_probe_of_kind(suite, "adapter_ablation")
    if probe is None:
        return None
    lambdas = probe.evidence.get("lambdas")
    divs = probe.evidence.get("mean_divergence_per_lambda")
    if not lambdas or not divs:
        return None
    sat = probe.evidence.get("saturation_lambda")
    fig = go.Figure(
        go.Scatter(
            x=list(lambdas),
            y=list(divs),
            mode="lines+markers",
            marker={"color": _CATEGORY_COLOR["ablation"], "size": 10},
            line={"color": _CATEGORY_COLOR["ablation"], "width": 2},
            hovertemplate="λ=%{x}<br>div=%{y:.4f}<extra></extra>",
            name="divergence",
        ),
        layout=go.Layout(
            xaxis={"title": "lambda"},
            yaxis={"title": "mean divergence"},
            height=320,
            margin={"l": 60, "r": 20, "t": 30, "b": 50},
        ),
    )
    if sat is not None:
        fig.add_vline(
            x=float(sat),
            line_dash="dash",
            line_color="#6c757d",
            annotation_text=f"sat_λ={float(sat):.2f}",
            annotation_position="top",
        )
    return fig


def _scatter_figure(go: Any, suite: SuiteResult) -> Any:
    """Score vs. z-score scatter across every probe, colored by verdict."""
    xs: list[float] = []
    ys: list[float] = []
    texts: list[str] = []
    colors: list[str] = []
    for p in suite.probes:
        # Plot only probes with a numeric score; SKIP / ERROR probes
        # without a score are summarized in the per-row annotation instead
        # of cluttering the scatter at (0,0).
        if p.score is None:
            continue
        xs.append(float(p.score))
        ys.append(float(p.z_score) if p.z_score is not None else 0.0)
        texts.append(
            f"<b>{html.escape(p.name)}</b><br>"
            f"kind: {html.escape(p.kind)}<br>"
            f"verdict: {p.verdict.value}<br>"
            f"score: {p.score:.3f}<br>"
            f"z: {'—' if p.z_score is None else f'{p.z_score:+.2f}σ'}"
        )
        colors.append(_VERDICT_COLOR.get(p.verdict, "#888888"))
    return go.Figure(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker={"size": 14, "color": colors, "line": {"color": "#333", "width": 1}},
            text=texts,
            hovertemplate="%{text}<extra></extra>",
        ),
        layout=go.Layout(
            xaxis={"title": "score", "range": [0.0, 1.0]},
            yaxis={"title": "z-score (σ)", "zeroline": True},
            height=360,
            margin={"l": 60, "r": 20, "t": 30, "b": 50},
        ),
    )


# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------


def _fig_to_div(pio: Any, fig: Any, div_id: str) -> str:
    """Render one figure as a div, reusing the JS we embed once in <head>."""
    return str(
        pio.to_html(
            fig,
            include_plotlyjs=False,
            full_html=False,
            div_id=div_id,
            config={"displaylogo": False, "responsive": True},
        )
    )


def _assemble(
    suite: SuiteResult,
    score: SwayScore,
    panels: list[tuple[str, str, str]],
    *,
    plotly_js: str,
) -> str:
    """Stitch the page together: header card, panels, probe table."""
    title = html.escape(f"sway report — {suite.adapter_id or suite.base_model_id}")
    verdict_summary = _verdict_summary(suite)
    header = (
        f"<h1>{title}</h1>"
        f"<p class='meta'>"
        f"base: <code>{html.escape(suite.base_model_id)}</code> · "
        f"adapter: <code>{html.escape(suite.adapter_id or '—')}</code> · "
        f"sway {html.escape(suite.sway_version)} · "
        f"wall: {suite.wall_seconds:.2f}s"
        f"</p>"
        f"<p class='summary'><b>overall</b>: {score.overall:.2f} "
        f"({html.escape(score.band or '—')})"
        f" · {verdict_summary}</p>"
    )

    panel_html_parts: list[str] = []
    for title_, div_id, div_html in panels:
        panel_html_parts.append(
            f"<section class='panel' id='panel-{html.escape(div_id)}'>"
            f"<h2>{html.escape(title_)}</h2>"
            f"{div_html}"
            f"</section>"
        )

    probe_table = _probe_table_html(suite)

    return _TEMPLATE.format(
        title=title,
        plotly_js=plotly_js,
        header=header,
        panels="\n".join(panel_html_parts),
        probe_table=probe_table,
    )


def _verdict_summary(suite: SuiteResult) -> str:
    counts: dict[Verdict, int] = {}
    for p in suite.probes:
        counts[p.verdict] = counts.get(p.verdict, 0) + 1
    parts = []
    for v in (Verdict.PASS, Verdict.FAIL, Verdict.WARN, Verdict.SKIP, Verdict.ERROR):
        if v in counts:
            parts.append(f"<span class='v-{v.value}'>{counts[v]} {html.escape(v.value)}</span>")
    return " · ".join(parts) or "no probes ran"


def _probe_table_html(suite: SuiteResult) -> str:
    """Textual per-probe table under the charts — same columns as markdown."""
    rows: list[str] = []
    for p in suite.probes:
        rows.append(
            "<tr>"
            f"<td>{html.escape(p.name)}</td>"
            f"<td><code>{html.escape(p.kind)}</code></td>"
            f"<td class='v-{p.verdict.value}'>{html.escape(p.verdict.value)}</td>"
            f"<td>{'—' if p.score is None else f'{p.score:.2f}'}</td>"
            f"<td>{'—' if p.raw is None else f'{p.raw:,.3f}'}</td>"
            f"<td>{'—' if p.z_score is None else f'{p.z_score:+.2f}σ'}</td>"
            f"<td class='note'>{html.escape(p.message or '')}</td>"
            "</tr>"
        )
    return (
        "<section class='probe-table'>"
        "<h2>Probes</h2>"
        "<table>"
        "<thead><tr><th>name</th><th>kind</th><th>verdict</th>"
        "<th>score</th><th>raw</th><th>z</th><th>note</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table></section>"
    )


def _first_probe_of_kind(suite: SuiteResult, kind: str) -> ProbeResult | None:
    for p in suite.probes:
        if p.kind == kind and p.score is not None:
            return p
    return None


def _band_color(band: str) -> str:
    return {
        "noise": "#dc3545",
        "partial": "#ffc107",
        "healthy": "#28a745",
        "suspicious": "#9c27b0",
    }.get(band, "#6c757d")


# ----------------------------------------------------------------------
# Static template. Kept inline (no separate template file) because a
# single page has one consumer and a two-file split would be more
# ceremony than it's worth at this scale.
# ----------------------------------------------------------------------

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="generator" content="sway report html">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           margin: 0; padding: 2rem; color: #222; background: #fafafa; max-width: 1100px;
           margin-left: auto; margin-right: auto; }}
    h1 {{ margin-bottom: 0.25rem; }}
    p.meta {{ color: #666; margin-top: 0; }}
    p.summary {{ font-size: 1.1rem; margin-top: 0.5rem; }}
    section.panel {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 6px;
                     padding: 1rem; margin-top: 1rem; }}
    section.panel h2 {{ margin-top: 0; font-size: 1.1rem; color: #333; }}
    section.probe-table {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 6px;
                           padding: 1rem; margin-top: 1rem; }}
    section.probe-table table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
    section.probe-table th, section.probe-table td {{ padding: 0.4rem 0.6rem;
                                                       border-bottom: 1px solid #eee;
                                                       text-align: left; vertical-align: top; }}
    section.probe-table td.note {{ color: #555; }}
    .v-pass {{ color: #28a745; font-weight: bold; }}
    .v-fail {{ color: #dc3545; font-weight: bold; }}
    .v-warn {{ color: #c98a00; font-weight: bold; }}
    .v-skip {{ color: #6c757d; }}
    .v-error {{ color: #9c27b0; font-weight: bold; }}
    code {{ font-family: 'Menlo', 'Consolas', monospace; font-size: 0.9em;
            background: #f0f0f0; padding: 0.05em 0.3em; border-radius: 3px; }}
  </style>
  <script type="text/javascript">{plotly_js}</script>
</head>
<body>
  {header}
  {panels}
  {probe_table}
</body>
</html>
"""


__all__ = ["to_html"]
