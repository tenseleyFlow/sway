"""Command implementations for the ``sway`` CLI.

Each function here is wired to a subcommand in :mod:`dlm_sway.cli.app`.
Commands deliberately do as little as possible themselves — the real
work lives in :mod:`dlm_sway.suite`, :mod:`dlm_sway.backends`, and the
probes package.
"""

from __future__ import annotations

import json
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from dlm_sway import __version__
from dlm_sway.core.errors import SwayError
from dlm_sway.core.result import SuiteResult, SwayScore, Verdict


def run_cmd(
    spec: Annotated[Path, typer.Argument(help="Path to a sway.yaml spec.")],
    json_out: Annotated[
        Path | None,
        typer.Option(
            "--json",
            "-j",
            help="Write the JSON report to this path in addition to the terminal render.",
        ),
    ] = None,
    markdown_out: Annotated[
        Path | None,
        typer.Option("--markdown", "-m", help="Write a markdown report to this path."),
    ] = None,
    weights: Annotated[
        str | None,
        typer.Option(
            "--weights",
            help=(
                "Override composite-score category weights. Format: "
                "'adherence=0.4,attribution=0.3,calibration=0.2,ablation=0.1'. "
                "Unspecified categories keep their defaults."
            ),
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Validate the spec, list the probes that would run with their "
                "category, and exit 0 — no backend is built (D6)."
            ),
        ),
    ] = False,
    trace: Annotated[
        Path | None,
        typer.Option(
            "--trace",
            help=(
                "Write a forward-pass trace (JSONL) to this path — one event "
                "per backend scoring call with probe / view / cache-hit info. "
                "Useful for perf investigation; zero overhead when unset."
            ),
        ),
    ] = None,
) -> None:
    """Execute a suite and render a terminal report."""
    if dry_run:
        _print_dry_run(spec)
        return
    try:
        weights_override = _parse_weights_flag(weights)
        result, score_obj = _execute_spec(spec, weights_override=weights_override, trace_path=trace)
    except SwayError as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc

    from dlm_sway.suite import report

    console = Console()
    report.to_terminal(result, score_obj, console=console)

    if json_out is not None:
        json_out.write_text(report.to_json(result, score_obj), encoding="utf-8")
        console.print(f"\n[dim]wrote JSON → {json_out}[/dim]")
    if markdown_out is not None:
        markdown_out.write_text(report.to_markdown(result, score_obj), encoding="utf-8")
        console.print(f"[dim]wrote markdown → {markdown_out}[/dim]")


def _print_dry_run(spec_path: Path) -> None:
    """D6: load + validate the spec, print the probe table, exit cleanly.

    No backend construction — useful for fast feedback on spec edits
    before paying for a model load.
    """
    from rich.table import Table

    from dlm_sway.probes.base import build_probe, registry, validate_all_probes
    from dlm_sway.suite.loader import load_spec

    try:
        spec = load_spec(spec_path)
        validate_all_probes(spec.suite)
    except SwayError as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc

    console = Console()
    console.print(f"[bold]dry-run for {spec_path}[/bold] — {len(spec.suite)} probe(s)")
    console.print()

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("#", style="dim")
    table.add_column("name", style="cyan")
    table.add_column("kind")
    table.add_column("category", style="dim")
    table.add_column("enabled", style="dim")
    registered = registry()
    for idx, raw in enumerate(spec.suite, start=1):
        probe, probe_spec = build_probe(raw)
        cls = registered.get(probe.kind)
        category = cls.category if cls is not None else "?"
        table.add_row(
            str(idx),
            probe_spec.name,
            probe.kind,
            category,
            "yes" if probe_spec.enabled else "no",
        )
    console.print(table)


def list_probes_cmd() -> None:
    """List every shipped probe kind with its category + one-line summary (D6)."""
    import sys

    from rich.table import Table

    # Make sure every probe module has been imported and registered.
    import dlm_sway.probes  # noqa: F401
    from dlm_sway.probes.base import registry

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("kind", style="cyan")
    table.add_column("category", style="dim")
    table.add_column("summary")
    for kind in sorted(registry()):
        cls = registry()[kind]
        # Prefer the class-level docstring, then fall back to the
        # defining module's module-level docstring. Most probe modules
        # lead with a solid one-liner at the top; the class body often
        # skips a docstring to avoid repeating it.
        summary = _first_doc_line(cls.__doc__)
        if not summary:
            module = sys.modules.get(cls.__module__)
            summary = _first_doc_line(getattr(module, "__doc__", None))
        table.add_row(kind, cls.category, summary)
    Console().print(table)


def _first_doc_line(doc: str | None) -> str:
    """Return the first non-empty line of ``doc``, stripped."""
    if not doc:
        return ""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def gate_cmd(
    spec: Annotated[Path, typer.Argument(help="Path to a sway.yaml spec.")],
    junit_out: Annotated[
        Path | None, typer.Option("--junit", help="Write JUnit XML for CI ingestion.")
    ] = None,
    coverage_threshold: Annotated[
        float | None,
        typer.Option(
            "--threshold",
            help="Override the spec's coverage_threshold. Exit non-zero below it.",
        ),
    ] = None,
    weights: Annotated[
        str | None,
        typer.Option(
            "--weights",
            help=(
                "Override composite-score category weights. Format: "
                "'adherence=0.4,attribution=0.3,calibration=0.2,ablation=0.1'. "
                "Unspecified categories keep their defaults."
            ),
        ),
    ] = None,
) -> None:
    """Execute a suite and exit non-zero on failure (CI gate)."""
    try:
        weights_override = _parse_weights_flag(weights)
        result, score_obj = _execute_spec(spec, weights_override=weights_override)
    except SwayError as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc

    from dlm_sway.suite import report
    from dlm_sway.suite.loader import load_spec as _load_spec

    console = Console()
    report.to_terminal(result, score_obj, console=console)

    if junit_out is not None:
        junit_out.write_text(report.to_junit(result, score_obj), encoding="utf-8")
        console.print(f"[dim]wrote JUnit → {junit_out}[/dim]")

    threshold = (
        coverage_threshold
        if coverage_threshold is not None
        else _load_spec(spec).defaults.coverage_threshold
    )
    has_failures = any(p.verdict == Verdict.FAIL for p in result.probes)
    below_threshold = score_obj.overall < threshold
    if has_failures or below_threshold:
        console.print(
            f"\n[red]gate FAILED[/red] — overall={score_obj.overall:.2f} < {threshold:.2f}"
            if below_threshold
            else "\n[red]gate FAILED[/red] — at least one probe reported FAIL"
        )
        raise typer.Exit(code=1)
    console.print(f"\n[green]gate passed[/green] — overall={score_obj.overall:.2f}")


def _infer_base_from_adapter_config(adapter_dir: Path) -> str | None:
    """Read ``base_model_name_or_path`` from ``adapter_config.json``.

    Returns ``None`` when the file is missing, malformed, or doesn't
    expose the field. Used by ``sway check`` to make ``--base`` optional
    in the common case where PEFT already wrote the base id on training
    (D4).
    """
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        return None
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    base = data.get("base_model_name_or_path")
    if isinstance(base, str) and base:
        return base
    return None


def _check_banner(score_obj: SwayScore, result: SuiteResult) -> tuple[str, str]:
    """Compute the (text, rich-style) check verdict banner (D12).

    Calibrated on the delta_kl z-score: ≥3σ is green ("above noise"),
    ≥1σ is yellow ("marginal"), and below that is red. When no z-score
    is available (no null calibration ran), falls back to the raw
    score band.
    """
    z = next(
        (p.z_score for p in result.probes if p.kind == "delta_kl" and p.z_score is not None),
        None,
    )
    if z is not None:
        if z >= 3.0:
            return f"✅ adapter is {z:+.2f}σ above noise", "bold green"
        if z >= 1.0:
            return f"⚠️ adapter is {z:+.2f}σ above noise — marginal", "bold yellow"
        return f"❌ adapter is {z:+.2f}σ — indistinguishable from noise", "bold red"

    # Fallback: composite score band.
    if score_obj.overall >= 0.6:
        return f"✅ adapter scored {score_obj.overall:.2f} — looks healthy", "bold green"
    if score_obj.overall >= 0.3:
        return f"⚠️ adapter scored {score_obj.overall:.2f} — partial fit", "bold yellow"
    return f"❌ adapter scored {score_obj.overall:.2f} — noise band", "bold red"


def check_cmd(
    adapter: Annotated[Path, typer.Argument(help="Path to a PEFT adapter directory.")],
    base: Annotated[
        str | None,
        typer.Option(
            "--base",
            help=(
                "HuggingFace base model id or local path. Inferred from "
                "the adapter's ``adapter_config.json`` when omitted (D4)."
            ),
        ),
    ] = None,
    prompts: Annotated[
        Path | None,
        typer.Option(
            "--prompts",
            help="File with one prompt per line. Defaults to sway's built-in quick set.",
        ),
    ] = None,
) -> None:
    """<60s smoke test: "is this adapter doing anything at all?".

    Runs A1 DeltaKL + C2 CalibrationDrift on a small prompt set. No
    spec file required.

    **Banner semantics (F20 clarification).** The ``+N.NNσ above noise``
    header appears only when ``null_adapter`` actually calibrated this
    run — i.e., when the backend implements ``NullCalibratedBackend``.
    Without null calibration (non-HF backends like the HTTP API or MLX
    inference), the banner falls back to the composite score band
    ("healthy", "partial fit", "noise band") and the σ wording is
    suppressed to avoid a false-precision claim.
    """
    from dlm_sway.backends import build as build_backend
    from dlm_sway.core.model import ModelSpec
    from dlm_sway.suite import report
    from dlm_sway.suite.runner import run as run_suite
    from dlm_sway.suite.score import compute as compute_score
    from dlm_sway.suite.spec import SuiteDefaults, SuiteModels, SwaySpec

    # D4: try to infer base model from adapter_config.json before
    # erroring out on a missing --base.
    if base is None:
        inferred = _infer_base_from_adapter_config(adapter)
        if inferred is None:
            typer.secho(
                f"error: --base not given and adapter at {adapter} doesn't carry a "
                f"base_model_name_or_path in adapter_config.json. Pass --base "
                f"explicitly.",
                err=True,
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=2)
        base = inferred
        typer.secho(f"(inferred base model: {base})", err=True, fg=typer.colors.CYAN)

    quick_prompts = _load_prompts(prompts) if prompts else _BUILTIN_QUICK_PROMPTS

    base_spec = ModelSpec(base=base, kind="hf")
    ft_spec = ModelSpec(base=base, kind="hf", adapter=adapter)
    spec = SwaySpec(
        version=1,
        models=SuiteModels(base=base_spec, ft=ft_spec),
        defaults=SuiteDefaults(seed=0),
        suite=[
            # Calibrate first so delta_kl can publish a z-score the
            # banner reads off.
            {"name": "quick_null", "kind": "null_adapter", "runs": 3},
            {
                "name": "quick_delta_kl",
                "kind": "delta_kl",
                "prompts": list(quick_prompts),
                "assert_mean_gte": 0.01,
            },
            {
                "name": "quick_calibration",
                "kind": "calibration_drift",
                "items_limit": 10,
            },
        ],
    )
    try:
        backend = build_backend(ft_spec)
    except SwayError as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc

    try:
        result = run_suite(spec, backend, spec_path="<check>")
    finally:
        _close_if_possible(backend)
    score_obj = compute_score(result)

    # D12: top-line banner before the full report so a user looking
    # only at the first line still gets the verdict.
    console = Console()
    banner_text, banner_style = _check_banner(score_obj, result)
    console.print()
    console.print(banner_text, style=banner_style)
    console.print()
    report.to_terminal(result, score_obj, console=console)


def diff_cmd(
    spec: Annotated[Path, typer.Argument(help="Path to a sway.yaml spec.")],
    adapter_a: Annotated[Path, typer.Option("--a", help="First adapter path.")],
    adapter_b: Annotated[Path, typer.Option("--b", help="Second adapter path.")],
) -> None:
    """Run the same suite against two adapters and show per-probe deltas."""
    from dlm_sway.backends import build as build_backend
    from dlm_sway.suite.loader import load_spec
    from dlm_sway.suite.runner import run as run_suite
    from dlm_sway.suite.score import compute as compute_score

    sway_spec = load_spec(spec)
    console = Console()

    def _score_for(adapter_path: Path) -> tuple[float, dict[str, float]]:
        ft_spec = sway_spec.models.ft.model_copy(update={"adapter": adapter_path})
        backend = build_backend(ft_spec)
        try:
            result = run_suite(sway_spec, backend, spec_path=str(spec))
        finally:
            _close_if_possible(backend)
        scored = compute_score(result)
        per_probe = {p.name: (p.score or 0.0) for p in result.probes}
        return scored.overall, per_probe

    try:
        overall_a, per_a = _score_for(adapter_a)
        overall_b, per_b = _score_for(adapter_b)
    except SwayError as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc

    console.print(f"[bold]overall[/bold]  A: {overall_a:.2f}   B: {overall_b:.2f}")
    console.print()
    console.print("[bold]per-probe[/bold] (A → B, Δ):")
    regressed_small = 0  # |Δ| > 0.10 in the wrong direction
    regressed_large = 0  # |Δ| > 0.20 in the wrong direction
    for name in sorted(per_a.keys() | per_b.keys()):
        a = per_a.get(name, 0.0)
        b = per_b.get(name, 0.0)
        delta = b - a
        sign = "+" if delta >= 0 else ""
        console.print(f"  {name:<30}  {a:.2f}  →  {b:.2f}   ({sign}{delta:+.2f})")
        if delta < -0.10:
            regressed_small += 1
        if delta < -0.20:
            regressed_large += 1

    # D13: regression summary line. The audit's example phrasing was
    # "A→B: 3 probes regressed >0.10, 1 regressed >0.20, composite Δ=+0.02".
    # Color cue tracks the composite delta: green for any improvement,
    # red on regression, yellow on flat-with-regressions.
    composite_delta = overall_b - overall_a
    if composite_delta > 0.0:
        summary_style = "bold green"
    elif regressed_small or regressed_large:
        summary_style = "bold red" if composite_delta < 0.0 else "bold yellow"
    else:
        summary_style = "dim"

    console.print()
    console.print(
        f"A→B: {regressed_small} probe(s) regressed >0.10, "
        f"{regressed_large} regressed >0.20, "
        f"composite Δ={composite_delta:+.2f}",
        style=summary_style,
    )


def autogen_cmd(
    dlm_path: Annotated[Path, typer.Argument(help="Path to a .dlm file.")],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Where to write the generated sway.yaml."),
    ] = Path("sway.yaml"),
) -> None:
    """Generate a sway.yaml from a .dlm file (requires the ``dlm-sway[dlm]`` extra)."""
    import importlib

    try:
        autogen_mod = importlib.import_module("dlm_sway.integrations.dlm.autogen")
    except ImportError as exc:
        typer.secho(
            "dlm integration not installed — run: pip install 'dlm-sway[dlm]'",
            err=True,
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2) from exc

    try:
        autogen_mod.write_sway_yaml(dlm_path, out)
    except SwayError as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc

    typer.echo(f"wrote {out}")


_DOCTOR_BACKENDS: dict[str, tuple[str, ...]] = {
    "hf": ("torch", "transformers", "peft"),
    "mlx": ("mlx", "mlx_lm"),
    # ``sklearn`` is S16's cluster_kl dep; shipped under [semsim] so it
    # rides the same 80 MB MiniLM load adapter_revert already pulls.
    "semsim": ("sentence_transformers", "sklearn"),
    "style": ("spacy", "textstat", "nlpaug"),
    "dlm": ("dlm",),
    # ``plotly`` is the load-bearing dep for ``sway report --format html``;
    # S12 docs listed it but doctor never probed it before F04.
    "viz": ("matplotlib", "plotly"),
    # S13 API backend.
    "api": ("httpx", "tenacity"),
    "pytest": ("pytest",),
}


def _doctor_payload() -> dict[str, Any]:
    """Build the JSON-friendly doctor payload (used by both render paths)."""
    extras: dict[str, dict[str, str | None]] = {}
    for extra, modules in _DOCTOR_BACKENDS.items():
        extras[extra] = {mod: _module_version(mod) for mod in modules}
    return {
        "sway_version": __version__,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "extras": extras,
    }


def _module_version(name: str) -> str | None:
    """Return the installed module's ``__version__`` string, or ``None``."""
    import importlib

    try:
        mod = importlib.import_module(name)
    except ImportError:
        return None
    return str(getattr(mod, "__version__", "installed"))


def doctor_cmd(
    json_out: Annotated[
        bool,
        typer.Option(
            "--json",
            help=(
                "Emit a machine-readable JSON payload instead of the rich "
                "terminal layout (D7). CI-grep-friendly."
            ),
        ),
    ] = False,
) -> None:
    """Print backend availability and version info."""
    payload = _doctor_payload()
    if json_out:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    console = Console()
    console.print(f"[bold]sway[/bold] {payload['sway_version']}")
    console.print(f"  python:    {payload['python']}")
    console.print(f"  platform:  {payload['platform']}")
    console.print()
    console.print("[bold]backends[/bold]")
    for extra, modules in payload["extras"].items():
        parts = []
        for mod, ver in modules.items():
            if ver is None:
                parts.append(f"[red]{mod}: missing[/red]")
            else:
                parts.append(f"[green]{mod}: {ver}[/green]")
        console.print(f"  {extra:<8}  {' '.join(parts)}")


class ReportFormat(StrEnum):
    """Allowed values for ``sway report --format`` (D11).

    Typer enforces the enum at parse time, so unknown formats produce
    a clear ``Invalid value`` error instead of silently falling back
    to the terminal renderer.
    """

    TERMINAL = "terminal"
    MARKDOWN = "md"
    MARKDOWN_LONG = "markdown"  # alias kept for muscle memory
    JUNIT = "junit"
    JSON = "json"
    HTML = "html"


def report_cmd(
    result_json: Annotated[Path, typer.Argument(help="Path to a saved result JSON.")],
    format: Annotated[
        ReportFormat,
        typer.Option(
            "--format",
            help="Output format: terminal, md (alias: markdown), junit, json, or html.",
        ),
    ] = ReportFormat.TERMINAL,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            help=(
                "Write the rendered output to this path instead of stdout. "
                "Required for --format html (Plotly's inlined JS is ~3 MB)."
            ),
        ),
    ] = None,
) -> None:
    """Re-render a previously saved run (for history tracking / dashboards).

    The CLI deserializes the JSON back into the canonical
    ``(SuiteResult, SwayScore)`` pair via :func:`report.from_json`,
    then routes through the same renderers as a fresh ``sway run``.
    Single source for every format keeps terminal / md / junit /
    json / html output identical regardless of where they came from (B16).
    """
    from dlm_sway.suite import report

    raw: dict[str, Any] = json.loads(result_json.read_text(encoding="utf-8"))

    if format is ReportFormat.JSON:
        # Pass-through: the saved file *is* the canonical JSON. Re-emit
        # via to_json against the round-tripped pair so any schema
        # additions land consistently.
        suite, score = report.from_json(raw)
        _emit(report.to_json(suite, score), out)
        return

    suite, score = report.from_json(raw)
    if format in (ReportFormat.MARKDOWN, ReportFormat.MARKDOWN_LONG):
        _emit(report.to_markdown(suite, score), out)
        return
    if format is ReportFormat.JUNIT:
        _emit(report.to_junit(suite, score), out)
        return
    if format is ReportFormat.HTML:
        try:
            from dlm_sway.suite import report_html
        except ImportError as exc:  # pragma: no cover — graceful install hint
            typer.echo(f"sway report --format html: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        try:
            html_text = report_html.to_html(suite, score)
        except RuntimeError as exc:
            typer.echo(f"sway report --format html: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        if out is None:
            # Refuse to dump 3 MB of HTML to stdout by default — the
            # user almost always wants a file.
            typer.echo(
                "sway report --format html requires --out PATH "
                "(Plotly JS bundle is ~3 MB; stdout is not an HTML viewer)",
                err=True,
            )
            raise typer.Exit(code=2)
        out.write_text(html_text, encoding="utf-8")
        typer.echo(f"wrote HTML → {out}", err=True)
        return
    # ReportFormat.TERMINAL.
    if out is not None:
        typer.echo(
            "sway report --format terminal does not support --out; "
            "use --format md or --format html for file output.",
            err=True,
        )
        raise typer.Exit(code=2)
    report.to_terminal(suite, score, console=Console())


def _emit(text: str, out: Path | None) -> None:
    """Either write to the target path or ``typer.echo`` to stdout."""
    if out is None:
        typer.echo(text)
    else:
        out.write_text(text, encoding="utf-8")
        typer.echo(f"wrote {out}", err=True)


class CompareFormat(StrEnum):
    """Allowed values for ``sway compare --format``."""

    TERMINAL = "terminal"
    MARKDOWN = "md"
    MARKDOWN_LONG = "markdown"  # alias kept for muscle memory
    JSON = "json"


def compare_cmd(
    result_jsons: Annotated[
        list[Path],
        typer.Argument(help="Two or more saved result JSONs, in chronological order."),
    ],
    format: Annotated[
        CompareFormat,
        typer.Option(
            "--format",
            help="Output format: terminal, md (alias: markdown), or json.",
        ),
    ] = CompareFormat.TERMINAL,
    fail_on_regression: Annotated[
        float,
        typer.Option(
            "--fail-on-regression",
            help=(
                "Exit non-zero when any probe's score in the newest run dropped "
                "by ≥ this threshold vs the previous run. 0 disables the gate."
            ),
        ),
    ] = 0.0,
) -> None:
    """Compare N saved runs side-by-side (regression dashboard).

    Rehydrates each JSON via :func:`report.from_json`, folds the runs
    into a :class:`CompareMatrix`, and renders the score table + delta
    columns + composite timeline. Intended for CI: point at a history
    directory (``sway-history/*.json``) and pipe the output into the
    build's log, or set ``--fail-on-regression`` to make the build red
    on a real drop.
    """
    from dlm_sway.suite import compare, report

    if len(result_jsons) < 2:
        typer.echo("sway compare: need at least two result JSONs", err=True)
        raise typer.Exit(code=2)

    pairs: list[tuple[SuiteResult, SwayScore]] = []
    labels: list[str] = []
    for path in result_jsons:
        try:
            raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            typer.echo(f"sway compare: cannot read {path}: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        pairs.append(report.from_json(raw))
        # Short label — the filename without the ``.json`` suffix.
        labels.append(path.stem)

    matrix = compare.build_matrix(pairs, labels=labels)

    if format is CompareFormat.JSON:
        typer.echo(compare.render_json(matrix, regression_threshold=fail_on_regression))
    elif format in (CompareFormat.MARKDOWN, CompareFormat.MARKDOWN_LONG):
        typer.echo(compare.render_markdown(matrix, regression_threshold=fail_on_regression))
    else:
        compare.render_terminal(
            matrix,
            console=Console(),
            regression_threshold=fail_on_regression,
        )

    # Exit-code gate: any probe whose last-run delta is ≤ -threshold is a
    # regression. ``fail_on_regression=0`` disables the gate entirely.
    if fail_on_regression > 0.0:
        regressions = matrix.latest_regressions(fail_on_regression)
        if regressions:
            raise typer.Exit(code=1)


class TraceFormat(StrEnum):
    """Allowed values for ``sway trace --format``."""

    TERMINAL = "terminal"
    MARKDOWN = "md"
    MARKDOWN_LONG = "markdown"  # alias kept for muscle memory
    JSON = "json"


def trace_cmd(
    trace_file: Annotated[
        Path,
        typer.Argument(
            help=("Path to a forward-pass trace JSONL produced by `sway run --trace <path>`."),
        ),
    ],
    format: Annotated[
        TraceFormat,
        typer.Option(
            "--format",
            help="Output format: terminal, md (alias: markdown), or json.",
        ),
    ] = TraceFormat.TERMINAL,
    slowest: Annotated[
        int,
        typer.Option(
            "--slowest",
            help="How many slowest-events rows to show. 0 hides that table.",
        ),
    ] = 10,
) -> None:
    """Analyze a forward-pass trace JSONL.

    Reads the per-event JSONL `sway run --trace` writes, aggregates
    into per-probe + per-view summaries, and surfaces the top-N
    slowest events. Intended for suite-performance investigation:
    point at a captured trace and see which probe × view pair
    dominated wall time, whether the S07 cache helped, and which
    individual prompts took the longest.
    """
    from dlm_sway.suite import trace_analysis

    try:
        events = trace_analysis.load(trace_file)
    except OSError as exc:
        typer.echo(f"sway trace: cannot read {trace_file}: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if not events:
        typer.echo(f"sway trace: no events in {trace_file}", err=True)
        raise typer.Exit(code=1)

    report = trace_analysis.build_report(events, slowest_k=max(0, slowest))

    if format is TraceFormat.JSON:
        typer.echo(trace_analysis.render_json(report))
    elif format in (TraceFormat.MARKDOWN, TraceFormat.MARKDOWN_LONG):
        typer.echo(trace_analysis.render_markdown(report))
    else:
        trace_analysis.render_terminal(report, console=Console())


# -- helpers -----------------------------------------------------------


_BUILTIN_QUICK_PROMPTS: tuple[str, ...] = (
    "The quick brown fox",
    "Once upon a time",
    "The answer to the question is",
    "One important lesson is",
    "In my opinion,",
    "The first step is to",
    "Remember that",
    "A common mistake is",
)


def _load_prompts(path: Path) -> tuple[str, ...]:
    return tuple(
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def _execute_spec(
    path: Path,
    *,
    weights_override: dict[str, float] | None = None,
    trace_path: Path | None = None,
) -> tuple[SuiteResult, SwayScore]:
    """Load a spec, build a backend, run the suite, fold scores. Shared
    by ``run`` and ``gate``. Picks up .dlm-derived sections when the
    spec's ``dlm_source`` is set.

    ``weights_override`` takes precedence over ``spec.defaults.score_weights``
    (which itself takes precedence over the compile-time defaults). The
    CLI hands through ``--weights k=v,k=v`` via this parameter.
    """
    from dlm_sway.backends import build as build_backend
    from dlm_sway.backends import build_two_separate
    from dlm_sway.probes.base import validate_all_probes
    from dlm_sway.suite.loader import load_spec
    from dlm_sway.suite.runner import run as run_suite
    from dlm_sway.suite.score import compute as compute_score

    spec = load_spec(path)
    # B7: validate every probe entry before paying the cost of loading
    # a backend. A user with a typo in `kind:` shouldn't wait minutes
    # for the model to download just to learn they spelled the probe
    # name wrong.
    validate_all_probes(spec.suite)
    sections = None
    doc_text = None
    if spec.dlm_source is not None:
        import importlib

        try:
            resolver = importlib.import_module("dlm_sway.integrations.dlm.resolver")
            handle = resolver.resolve_dlm(Path(spec.dlm_source))
            sections = handle.sections
            doc_text = handle.doc_text
        except ImportError:
            # D8: don't silently swallow. The user wrote ``dlm_source``
            # in their YAML expecting the bridge to populate sections;
            # warn loudly so they know why downstream attribution
            # probes are SKIPping.
            typer.secho(
                f"warning: spec sets dlm_source={spec.dlm_source!r} but the "
                f"[dlm] extra is not installed — sections not provided "
                f"(pip install 'dlm-sway[dlm]')",
                err=True,
                fg=typer.colors.YELLOW,
            )
            sections = None
        except SwayError as exc:
            # The bridge imported but failed (no adapter, malformed
            # .dlm, etc). Same surface — warn, don't crash the suite.
            typer.secho(
                f"warning: dlm_source={spec.dlm_source!r} did not resolve: {exc}",
                err=True,
                fg=typer.colors.YELLOW,
            )
            sections = None
    if spec.defaults.differential:
        backend: Any = build_backend(spec.models.ft)
    else:
        backend = build_two_separate(spec.models)
    try:
        result = run_suite(
            spec,
            backend,
            spec_path=str(path),
            sections=sections,
            doc_text=doc_text,
            trace_path=trace_path,
        )
    finally:
        _close_if_possible(backend)
    effective_weights = weights_override or spec.defaults.score_weights
    score_obj = compute_score(result, weights=effective_weights)
    return result, score_obj


def _parse_weights_flag(raw: str | None) -> dict[str, float] | None:
    """Parse ``--weights k=v,k=v`` into a dict; pydantic validates on use.

    Returns ``None`` when the flag is empty / unset. Pydantic's
    ``SuiteDefaults._validate_weights`` is re-invoked indirectly via
    ``SwayScore`` — so any unknown category or negative value surfaces
    the same error whether set in YAML or on the command line.
    """
    if not raw:
        return None
    out: dict[str, float] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise typer.BadParameter(
                f"--weights: expected 'key=value' pairs, got {pair!r}. "
                "Example: --weights adherence=0.4,attribution=0.3"
            )
        key, _, value = pair.partition("=")
        key = key.strip()
        try:
            out[key] = float(value.strip())
        except ValueError as exc:
            raise typer.BadParameter(f"--weights: {value!r} for {key!r} is not a number") from exc
    return out or None


def _close_if_possible(backend: object) -> None:
    close = getattr(backend, "close", None)
    if callable(close):
        close()
