"""Integration test: ``sway gate`` exits non-zero on FAIL (C6).

Audit 01 flagged that no test asserts the gate's CI contract — the
whole point of ``sway gate`` is to fail a CI run when probes fail or
the composite score drops below the coverage threshold. This test
spawns the CLI via Typer's ``CliRunner`` (in-process, fast) and pins
the exit-code behavior on three scenarios:

1. Spec that produces only PASS verdicts → exit 0.
2. Spec that produces a FAIL verdict → exit 1.
3. Spec where the composite score is below an explicit ``--threshold``
   override → exit 1.

We swap ``backends.build`` for a dummy-backend factory so the test
doesn't need a real HF model. That's a unit-style shortcut around the
``backends/__init__.py:build`` rejection of ``kind="dummy"``; the
gate's own logic is what's under test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.cli.app import app


@pytest.fixture
def stub_build_backend(monkeypatch: pytest.MonkeyPatch):
    """Replace ``backends.build`` with a dummy-returning factory."""

    def _factory(*_args, **_kwargs):
        return DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())

    # The CLI imports ``build`` lazily inside ``_execute_spec``; patch
    # the module-level attribute on ``dlm_sway.backends``.
    import dlm_sway.backends as backends_mod

    monkeypatch.setattr(backends_mod, "build", _factory)


def _write_spec(path: Path, *, prompts: list[str], threshold: float, assert_mean: float) -> None:
    path.write_text(
        f"""
version: 1
models:
  base:
    base: stub
    kind: hf
    adapter: /tmp/stub-adapter
  ft:
    base: stub
    kind: hf
    adapter: /tmp/stub-adapter
defaults:
  seed: 0
  coverage_threshold: {threshold}
suite:
  - name: dk
    kind: delta_kl
    prompts: {prompts!r}
    assert_mean_gte: {assert_mean}
""".strip()
    )


def test_gate_exits_zero_on_pass(stub_build_backend, tmp_path: Path) -> None:
    """Dummy backend produces zero divergence → assert_mean_gte=0.0 → PASS."""
    spec = tmp_path / "pass.yaml"
    _write_spec(spec, prompts=["q1", "q2"], threshold=0.0, assert_mean=0.0)
    result = CliRunner().invoke(app, ["gate", str(spec)])
    assert result.exit_code == 0, (
        f"expected exit 0; got {result.exit_code}\nstdout: {result.stdout}"
    )


def test_gate_exits_one_on_fail(stub_build_backend, tmp_path: Path) -> None:
    """Dummy backend produces zero divergence → assert_mean_gte=0.5 → FAIL → exit 1."""
    spec = tmp_path / "fail.yaml"
    _write_spec(spec, prompts=["q1", "q2"], threshold=0.0, assert_mean=0.5)
    result = CliRunner().invoke(app, ["gate", str(spec)])
    assert result.exit_code == 1, (
        f"expected exit 1 on FAIL; got {result.exit_code}\nstdout: {result.stdout}"
    )
    assert "FAILED" in result.stdout


def test_gate_exits_one_when_below_threshold(stub_build_backend, tmp_path: Path) -> None:
    """PASS verdict, but composite score < --threshold override → exit 1."""
    spec = tmp_path / "below.yaml"
    _write_spec(spec, prompts=["q1", "q2"], threshold=0.0, assert_mean=0.0)
    # Pass-criterion is satisfied (assert_mean_gte=0.0 always passes),
    # so the gate would exit 0 on verdict alone. Pin the threshold above
    # any reachable composite score (delta_kl's score is bounded by the
    # JS-ln(2)-normalized mean, well under 1.0 for the dummy backend) to
    # exercise the score-only failure path.
    result = CliRunner().invoke(app, ["gate", str(spec), "--threshold", "0.999"])
    assert result.exit_code == 1, (
        f"expected exit 1 below threshold; got {result.exit_code}\nstdout: {result.stdout}"
    )
    assert "FAILED" in result.stdout
