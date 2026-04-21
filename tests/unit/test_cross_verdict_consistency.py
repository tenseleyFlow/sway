"""Stronger-test #10 — ``run`` / ``gate`` / ``report --format junit`` agree.

B16 pinned the gate's PASS/FAIL accounting against ``sway run``'s
rendered table but didn't pin the ``report --format junit``
re-rendering of saved JSON. Schema drift between the writer and the
JUnit emitter would produce inconsistent PASS/FAIL tallies across the
three surfaces — a CI dashboard reading the JUnit file and a developer
reading the terminal output would see different numbers.

This test runs a controlled spec through all three CLI paths against
a stubbed dummy backend (the same pattern used by
``test_sway_gate_exit_code.py``) and asserts the per-verdict tally
is identical on every surface.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.cli.app import app


@pytest.fixture
def stub_build_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``backends.build`` with a dummy-returning factory so the
    CLI commands run without loading a real HF model. Same shape as
    ``tests/integration/test_sway_gate_exit_code.py``'s fixture."""

    def _factory(*_args: object, **_kwargs: object) -> DummyDifferentialBackend:
        return DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())

    import dlm_sway.backends as backends_mod

    monkeypatch.setattr(backends_mod, "build", _factory)


def _write_spec(path: Path) -> None:
    """Two delta_kl probes with different thresholds — one passes on
    the dummy backend's synthesized divergence, one fails."""
    path.write_text(
        """
version: 1
models:
  base:
    base: stub
    kind: hf
    adapter: /tmp/stub
  ft:
    base: stub
    kind: hf
    adapter: /tmp/stub
defaults:
  seed: 0
  coverage_threshold: 0.0
suite:
  - name: dk_loose
    kind: delta_kl
    prompts: [p1, p2, p3, p4]
    assert_mean_gte: 0.0
  - name: dk_strict
    kind: delta_kl
    prompts: [p1, p2, p3, p4]
    assert_mean_gte: 100.0
""".strip()
    )


def _verdicts_from_json(payload: dict[str, object]) -> Counter[str]:
    probes = payload["probes"]
    assert isinstance(probes, list)
    return Counter(str(p["verdict"]) for p in probes)


def _verdicts_from_junit(xml_text: str) -> Counter[str]:
    """<testcase> without child → PASS; with failure/error/skipped → mapped accordingly."""
    root = ET.fromstring(xml_text)
    counts: Counter[str] = Counter()
    for case in root.iter("testcase"):
        if case.find("error") is not None:
            counts["error"] += 1
        elif case.find("failure") is not None:
            counts["fail"] += 1
        elif case.find("skipped") is not None:
            counts["skip"] += 1
        else:
            counts["pass"] += 1
    return counts


def _verdicts_from_stdout(stdout: str) -> Counter[str]:
    """Lift per-row verdict tokens out of the markdown/terminal
    table rendered to stdout. The verdict column carries a single
    lowercased keyword in the fixed set."""
    return Counter(
        tok for tok in stdout.split() if tok in {"pass", "fail", "skip", "error", "warn"}
    )


def test_gate_run_junit_agree_on_verdicts(
    stub_build_backend: None,  # noqa: ARG001
    tmp_path: Path,
) -> None:
    spec = tmp_path / "sway.yaml"
    _write_spec(spec)
    runner = CliRunner()

    # ``sway run`` writes the JSON we'll re-render as JUnit.
    json_out = tmp_path / "run.json"
    run = runner.invoke(app, ["run", str(spec), "--json", str(json_out)])
    assert run.exit_code == 0, run.stdout
    assert json_out.exists()
    run_payload = json.loads(json_out.read_text(encoding="utf-8"))
    run_verdicts = _verdicts_from_json(run_payload)

    # ``sway gate`` surfaces PASS/FAIL in stdout even when it exits 1.
    gate = runner.invoke(app, ["gate", str(spec)])
    assert gate.exit_code == 1, gate.stdout  # one failing probe
    gate_verdicts = _verdicts_from_stdout(gate.stdout)

    # ``sway report --format junit`` re-renders the JSON file.
    junit_out = tmp_path / "run.junit.xml"
    rpt = runner.invoke(
        app, ["report", str(json_out), "--format", "junit", "--out", str(junit_out)]
    )
    assert rpt.exit_code == 0, rpt.stdout
    junit_verdicts = _verdicts_from_junit(junit_out.read_text(encoding="utf-8"))

    assert run_verdicts == gate_verdicts == junit_verdicts, (
        f"surfaces disagree:\n"
        f"  run:   {dict(run_verdicts)}\n"
        f"  gate:  {dict(gate_verdicts)}\n"
        f"  junit: {dict(junit_verdicts)}"
    )
    assert run_verdicts["pass"] == 1
    assert run_verdicts["fail"] == 1
