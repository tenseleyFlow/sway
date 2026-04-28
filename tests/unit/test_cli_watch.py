"""CLI tests for ``sway watch``.

Covers argument validation and the "spec without dlm_source" guard.
End-to-end watching is exercised by tests/unit/test_watch_observer.py
(unit) and tests/integration/test_watch_loop.py (integration).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("watchdog")

from typer.testing import CliRunner  # noqa: E402

from dlm_sway.cli.app import app  # noqa: E402

SPEC_WITH_DLM_SOURCE = """\
version: 1
dlm_source: ./mydoc.dlm
models:
  base: { kind: dummy, base: tiny-base }
  ft:   { kind: dummy, base: tiny-base, adapter: ./fake-adapter }
defaults:
  seed: 0
  differential: true
  coverage_threshold: 0.6
suite:
  - { name: smoke, kind: dir,
      prompt: "the cat",
      target: " sat",
      distractor: " ran",
      assert: { delta_logprob_gte: 0.0 } }
"""

SPEC_WITHOUT_DLM_SOURCE = """\
version: 1
models:
  base: { kind: dummy, base: tiny-base }
  ft:   { kind: dummy, base: tiny-base, adapter: ./fake-adapter }
defaults:
  seed: 0
  differential: true
  coverage_threshold: 0.6
suite:
  - { name: smoke, kind: dir,
      prompt: "the cat",
      target: " sat",
      distractor: " ran",
      assert: { delta_logprob_gte: 0.0 } }
"""


def test_watch_in_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "watch" in result.stdout


def test_watch_help_lists_required_flags() -> None:
    result = CliRunner().invoke(app, ["watch", "--help"])
    assert result.exit_code == 0
    out = result.stdout
    assert "--history-dir" in out
    assert "--max-history" in out
    assert "--on-fail" in out
    assert "--serve-url" in out
    assert "SWAY_SERVE_URL" in out
    assert "SWAY_RESULT_PATH" in out


def test_watch_refuses_negative_max_history(tmp_path: Path) -> None:
    spec = tmp_path / "spec.yaml"
    spec.write_text(SPEC_WITH_DLM_SOURCE, encoding="utf-8")
    result = CliRunner().invoke(app, ["watch", str(spec), "--max-history", "-1"])
    assert result.exit_code == 2
    combined = (result.stdout or "") + (result.stderr or "")
    assert "max-history" in combined.lower()


def test_watch_refuses_spec_without_dlm_source(tmp_path: Path) -> None:
    spec = tmp_path / "spec.yaml"
    spec.write_text(SPEC_WITHOUT_DLM_SOURCE, encoding="utf-8")
    result = CliRunner().invoke(app, ["watch", str(spec)])
    assert result.exit_code == 2
    combined = (result.stdout or "") + (result.stderr or "")
    assert "dlm_source" in combined


def test_watch_refuses_missing_dlm_source(tmp_path: Path) -> None:
    spec = tmp_path / "spec.yaml"
    spec.write_text(SPEC_WITH_DLM_SOURCE, encoding="utf-8")
    result = CliRunner().invoke(app, ["watch", str(spec)])
    # The .dlm file referenced doesn't exist → exit 2 with a clear message.
    assert result.exit_code == 2
    combined = (result.stdout or "") + (result.stderr or "")
    assert "dlm_source" in combined or "not found" in combined.lower()
