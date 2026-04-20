"""Command implementations for the ``dlm-sway`` CLI.

Each function here is wired to a subcommand in :mod:`dlm_sway.cli.app`.
Commands deliberately do as little as possible themselves — the real
work lives in :mod:`dlm_sway.suite`, :mod:`dlm_sway.backends`, and the
probes package.
"""

from __future__ import annotations

import json
import sys
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
) -> None:
    """Execute a suite and render a terminal report."""
    try:
        result, score_obj = _execute_spec(spec)
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
) -> None:
    """Execute a suite and exit non-zero on failure (CI gate)."""
    try:
        result, score_obj = _execute_spec(spec)
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


def check_cmd(
    adapter: Annotated[Path, typer.Argument(help="Path to a PEFT adapter directory.")],
    base: Annotated[str, typer.Option("--base", help="HuggingFace base model id or local path.")],
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
    """
    from dlm_sway.backends import build as build_backend
    from dlm_sway.core.model import ModelSpec
    from dlm_sway.suite import report
    from dlm_sway.suite.runner import run as run_suite
    from dlm_sway.suite.score import compute as compute_score
    from dlm_sway.suite.spec import SuiteDefaults, SuiteModels, SwaySpec

    quick_prompts = _load_prompts(prompts) if prompts else _BUILTIN_QUICK_PROMPTS

    base_spec = ModelSpec(base=base, kind="hf")
    ft_spec = ModelSpec(base=base, kind="hf", adapter=adapter)
    spec = SwaySpec(
        version=1,
        models=SuiteModels(base=base_spec, ft=ft_spec),
        defaults=SuiteDefaults(seed=0),
        suite=[
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
    report.to_terminal(result, score_obj, console=Console())


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
    for name in sorted(per_a.keys() | per_b.keys()):
        a = per_a.get(name, 0.0)
        b = per_b.get(name, 0.0)
        delta = b - a
        sign = "+" if delta >= 0 else ""
        console.print(f"  {name:<30}  {a:.2f}  →  {b:.2f}   ({sign}{delta:+.2f})")


def autogen_cmd(
    dlm_path: Annotated[Path, typer.Argument(help="Path to a .dlm file.")],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Where to write the generated sway.yaml."),
    ] = Path("sway.yaml"),
) -> None:
    """Generate a sway.yaml from a .dlm file (requires dlm-sway[dlm])."""
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


def doctor_cmd() -> None:
    """Print backend availability and version info."""
    console = Console()
    console.print(f"[bold]dlm-sway[/bold] {__version__}")
    console.print(f"  python:    {sys.version.split()[0]}")
    console.print(f"  platform:  {sys.platform}")
    console.print()

    console.print("[bold]backends[/bold]")
    console.print(
        f"  hf:        {_probe_import('torch')} {_probe_import('transformers')} {_probe_import('peft')}"
    )
    console.print(f"  mlx:       {_probe_import('mlx')} {_probe_import('mlx_lm')}")
    console.print(f"  semsim:    {_probe_import('sentence_transformers')}")
    console.print(
        f"  style+:    {_probe_import('spacy')} {_probe_import('textstat')} {_probe_import('nlpaug')}"
    )
    console.print(f"  dlm:       {_probe_import('dlm')}")
    console.print(f"  viz:       {_probe_import('matplotlib')}")


def report_cmd(
    result_json: Annotated[Path, typer.Argument(help="Path to a saved result JSON.")],
    format: Annotated[
        str, typer.Option("--format", help="Output format: terminal, md, junit, json.")
    ] = "terminal",
) -> None:
    """Re-render a previously saved run (for history tracking / dashboards)."""
    raw: dict[str, Any] = json.loads(result_json.read_text(encoding="utf-8"))
    fmt = format.lower()
    if fmt == "json":
        typer.echo(json.dumps(raw, indent=2, sort_keys=True))
        return
    if fmt in {"md", "markdown"}:
        # A file-level re-render needs the dataclasses back; simplest is
        # to synthesize a minimal markdown from the JSON directly.
        typer.echo(_render_markdown_from_json(raw))
        return
    if fmt == "junit":
        typer.echo(_render_junit_from_json(raw))
        return
    # Default: terminal-ish one-liner summary.
    score: dict[str, Any] = raw.get("score", {})
    typer.echo(f"overall: {score.get('overall', 0.0):.2f}  [{score.get('band', '?')}]")
    probes: list[dict[str, Any]] = raw.get("probes", [])
    for p in probes:
        typer.echo(
            f"  {p['name']:<30}  {p['verdict']:<6}  "
            f"{(p.get('score') or 0.0):.2f}  {p.get('message', '')[:60]}"
        )


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


def _execute_spec(path: Path) -> tuple[SuiteResult, SwayScore]:
    """Load a spec, build a backend, run the suite, fold scores. Shared
    by ``run`` and ``gate``. Picks up .dlm-derived sections when the
    spec's ``dlm_source`` is set."""
    from dlm_sway.backends import build as build_backend
    from dlm_sway.suite.loader import load_spec
    from dlm_sway.suite.runner import run as run_suite
    from dlm_sway.suite.score import compute as compute_score

    spec = load_spec(path)
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
            # Honoring dlm_source is best-effort — probes that need
            # sections will SKIP with a pointer at the extra.
            sections = None
    backend = build_backend(spec.models.ft)
    try:
        result = run_suite(spec, backend, spec_path=str(path), sections=sections, doc_text=doc_text)
    finally:
        _close_if_possible(backend)
    score_obj = compute_score(result)
    return result, score_obj


def _close_if_possible(backend: object) -> None:
    close = getattr(backend, "close", None)
    if callable(close):
        close()


def _probe_import(name: str) -> str:
    import importlib

    try:
        mod = importlib.import_module(name)
    except ImportError:
        return f"[red]{name}: missing[/red]"
    ver = getattr(mod, "__version__", "installed")
    return f"[green]{name}: {ver}[/green]"


def _render_markdown_from_json(raw: dict[str, Any]) -> str:
    score: dict[str, Any] = raw.get("score", {})
    lines: list[str] = [
        "# dlm-sway report",
        "",
        f"**Overall:** {score.get('overall', 0.0):.2f} (`{score.get('band', '?')}`)  ",
        f"**Base:** `{raw.get('base_model_id', '?')}`  ",
        f"**Adapter:** `{raw.get('adapter_id', '?')}`  ",
        "",
        "## Probes",
        "",
        "| name | kind | verdict | score |",
        "|---|---|---|---:|",
    ]
    probes: list[dict[str, Any]] = raw.get("probes", [])
    for p in probes:
        lines.append(
            f"| {p['name']} | `{p['kind']}` | {p['verdict']} | {(p.get('score') or 0.0):.2f} |"
        )
    return "\n".join(lines)


def _render_junit_from_json(raw: dict[str, Any]) -> str:
    """Minimal JUnit renderer from a saved JSON (useful for report --format junit)."""
    import xml.etree.ElementTree as ET

    probes: list[dict[str, Any]] = raw.get("probes", [])
    testsuite = ET.Element("testsuite", {"name": "dlm-sway", "tests": str(len(probes))})
    for p in probes:
        tc = ET.SubElement(testsuite, "testcase", {"classname": p["kind"], "name": p["name"]})
        if p["verdict"] == "fail":
            ET.SubElement(tc, "failure", {"message": p.get("message", "")})
        elif p["verdict"] == "error":
            ET.SubElement(tc, "error", {"message": p.get("message", "")})
        elif p["verdict"] == "skip":
            ET.SubElement(tc, "skipped", {"message": p.get("message", "")})
    return ET.tostring(testsuite, encoding="unicode")
