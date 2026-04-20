"""Tests for :mod:`dlm_sway.suite.runner`.

Uses the dummy backend + ad-hoc probe classes so nothing real is loaded.
"""

from __future__ import annotations

from typing import Literal

import pytest

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.errors import ProbeError
from dlm_sway.core.result import ProbeResult, Verdict
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext
from dlm_sway.suite.runner import run
from dlm_sway.suite.spec import SwaySpec


class _PassSpec(ProbeSpec):
    kind: Literal["__runner_pass"] = "__runner_pass"


class _PassProbe(Probe):
    kind = "__runner_pass"
    spec_cls = _PassSpec
    category = "adherence"

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        return ProbeResult(name=spec.name, kind=spec.kind, verdict=Verdict.PASS, score=0.9)


class _FailSpec(ProbeSpec):
    kind: Literal["__runner_fail"] = "__runner_fail"


class _FailProbe(Probe):
    kind = "__runner_fail"
    spec_cls = _FailSpec
    category = "attribution"

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        return ProbeResult(name=spec.name, kind=spec.kind, verdict=Verdict.FAIL, score=0.1)


class _RaiseSpec(ProbeSpec):
    kind: Literal["__runner_raise"] = "__runner_raise"


class _RaiseProbe(Probe):
    kind = "__runner_raise"
    spec_cls = _RaiseSpec

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        raise ProbeError(spec.kind, "kaboom")


class _UnexpectedSpec(ProbeSpec):
    kind: Literal["__runner_unexpected"] = "__runner_unexpected"


class _UnexpectedProbe(Probe):
    kind = "__runner_unexpected"
    spec_cls = _UnexpectedSpec

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        raise ValueError("surprise")


@pytest.fixture
def backend() -> DummyDifferentialBackend:
    return DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())


def _spec(*entries: dict) -> SwaySpec:
    return SwaySpec.model_validate(
        {
            "version": 1,
            "models": {
                "base": {"base": "b"},
                "ft": {"base": "b", "adapter": "/tmp/a"},
            },
            "suite": list(entries),
        }
    )


class TestRunner:
    def test_runs_each_probe_in_order(self, backend: DummyDifferentialBackend) -> None:
        spec = _spec(
            {"name": "p1", "kind": "__runner_pass"},
            {"name": "p2", "kind": "__runner_fail"},
        )
        result = run(spec, backend)
        assert [r.name for r in result.probes] == ["p1", "p2"]
        assert result.probes[0].verdict == Verdict.PASS
        assert result.probes[1].verdict == Verdict.FAIL

    def test_disabled_probe_records_skip(self, backend: DummyDifferentialBackend) -> None:
        spec = _spec({"name": "p1", "kind": "__runner_pass", "enabled": False})
        result = run(spec, backend)
        assert result.probes[0].verdict == Verdict.SKIP
        assert "disabled" in result.probes[0].message

    def test_probeerror_becomes_error_verdict(self, backend: DummyDifferentialBackend) -> None:
        spec = _spec({"name": "oops", "kind": "__runner_raise"})
        result = run(spec, backend)
        assert result.probes[0].verdict == Verdict.ERROR
        assert "kaboom" in result.probes[0].message

    def test_unexpected_exception_becomes_error_verdict(
        self, backend: DummyDifferentialBackend
    ) -> None:
        spec = _spec({"name": "oops", "kind": "__runner_unexpected"})
        result = run(spec, backend)
        assert result.probes[0].verdict == Verdict.ERROR
        assert "ValueError" in result.probes[0].message

    def test_wall_seconds_populated(self, backend: DummyDifferentialBackend) -> None:
        spec = _spec({"name": "p1", "kind": "__runner_pass"})
        result = run(spec, backend)
        assert result.wall_seconds >= 0
        assert result.probes[0].duration_s >= 0

    def test_null_adapter_passes_on_null_calibrated_backend(
        self, backend: DummyDifferentialBackend
    ) -> None:
        # Dummy backend implements NullCalibratedBackend, so calibration runs.
        spec = _spec({"name": "null", "kind": "null_adapter", "runs": 2, "prompts": ["q1"]})
        result = run(spec, backend)
        assert result.probes[0].kind == "null_adapter"
        assert result.probes[0].verdict == Verdict.PASS
        # And the suite's null_stats bubbles up onto the result.
        assert "delta_kl" in result.null_stats
