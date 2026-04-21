"""sway CLI entry point.

``pip install dlm-sway`` installs this module's :func:`main` as the
``sway`` console script. Every subcommand is a thin wrapper around a
library-level function so the CLI surface mirrors what programmatic
callers get.
"""

from __future__ import annotations

import typer

from dlm_sway import __version__
from dlm_sway.cli import commands

app = typer.Typer(
    name="sway",
    no_args_is_help=True,
    add_completion=False,
    help="Differential testing for fine-tuned causal language models.",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"sway {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(  # noqa: B008 — typer pattern
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print version and exit.",
    ),
) -> None:
    """Root callback; accepts ``--version``."""
    del version


app.command("run")(commands.run_cmd)
app.command("gate")(commands.gate_cmd)
app.command("check")(commands.check_cmd)
app.command("diff")(commands.diff_cmd)
app.command("autogen")(commands.autogen_cmd)
app.command("doctor")(commands.doctor_cmd)
app.command("report")(commands.report_cmd)
app.command("compare")(commands.compare_cmd)
app.command("trace")(commands.trace_cmd)
app.command("mine")(commands.mine_cmd)
app.command("list-probes")(commands.list_probes_cmd)


def main() -> None:
    """Script entry point registered in :file:`pyproject.toml`."""
    app()


if __name__ == "__main__":
    main()
