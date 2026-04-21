"""C7: positively assert that sway works without ``dlm`` installed.

The architectural promise from the plan: ``dlm-sway`` does not depend
on the ``dlm`` package; the bridge under
``src/dlm_sway/integrations/dlm/`` is the only place that imports it,
and only when actually invoked.

This test simulates the dlm-not-installed environment by patching
``sys.modules['dlm'] = None`` (which makes Python's import machinery
raise ``ModuleNotFoundError`` on subsequent ``import dlm``). It then:

1. Builds a small dummy-backend suite end-to-end and confirms it runs
   to completion with no ImportError.
2. Invokes ``sway autogen`` and confirms a clean error pointing at
   the ``[dlm]`` extra (rather than a stack trace).

The integration suite cannot actually uninstall dlm at test time, so
this is the next-best regression guard — it pins the lazy-import
boundary against future refactors that might pull dlm into a
top-level ``dlm_sway.*`` import path.
"""

from __future__ import annotations

import sys

import pytest
from typer.testing import CliRunner

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.cli.app import app
from dlm_sway.suite.runner import run as run_suite
from dlm_sway.suite.spec import SwaySpec


@pytest.fixture
def dlm_unimportable(monkeypatch: pytest.MonkeyPatch):
    """Make ``import dlm`` raise ModuleNotFoundError for the test."""
    # Block the parent package and any submodule the bridge imports.
    for name in (
        "dlm",
        "dlm.doc",
        "dlm.doc.parser",
        "dlm.base_models",
        "dlm.store",
        "dlm.store.paths",
        "dlm.data",
        "dlm.data.instruction_parser",
        "dlm.data.preference_parser",
    ):
        monkeypatch.setitem(sys.modules, name, None)


def test_dummy_suite_runs_without_dlm(dlm_unimportable) -> None:
    """A normal suite run never touches ``dlm`` and must work without it."""
    backend = DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
    spec = SwaySpec.model_validate(
        {
            "version": 1,
            "models": {
                "base": {"base": "b"},
                "ft": {"base": "b", "adapter": "/tmp/a"},
            },
            "suite": [
                {
                    "name": "dk",
                    "kind": "delta_kl",
                    "prompts": ["q1", "q2"],
                    "assert_mean_gte": 0.0,
                }
            ],
        }
    )
    result = run_suite(spec, backend)
    assert len(result.probes) == 1


def test_autogen_emits_clean_error_without_dlm(dlm_unimportable, tmp_path) -> None:
    """``sway autogen`` is the one path that needs dlm — it must surface
    a clean error pointing at the ``[dlm]`` extra, not a stack trace."""
    fake_dlm = tmp_path / "doesnt-matter.dlm"
    fake_dlm.write_text("# stub\n")
    result = CliRunner().invoke(app, ["autogen", str(fake_dlm)])
    assert result.exit_code == 2, (
        f"expected exit 2; got {result.exit_code}\nstdout: {result.stdout}\nstderr: {result.stderr if result.stderr_bytes else ''}"
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert "dlm-sway[dlm]" in combined, f"expected install hint in error output; got {combined!r}"
