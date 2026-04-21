"""Tests for :mod:`dlm_sway.suite.runner`.

Uses the dummy backend + ad-hoc probe classes so nothing real is loaded.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
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
        # Explicit calibrate_kinds so it runs even without downstream probes.
        spec = _spec(
            {
                "name": "null",
                "kind": "null_adapter",
                "runs": 2,
                "calibrate_kinds": ["delta_kl"],
            }
        )
        result = run(spec, backend)
        assert result.probes[0].kind == "null_adapter"
        assert result.probes[0].verdict == Verdict.PASS
        # And the suite's null_stats bubbles up onto the result.
        assert "delta_kl" in result.null_stats


class TestPreflightGate:
    """The S01 preflight gate: a NaN-producing backend aborts the suite.

    No probe runs; the SuiteResult contains a single synthetic ERROR
    probe explaining the abort.
    """

    def test_preflight_failure_aborts_suite(self) -> None:
        import math

        from dlm_sway.core.scoring import TokenDist

        # Seed a NaN dist on the ft side under the preflight prompt.
        nan_dist = TokenDist(
            token_ids=np.array([1, 2], dtype=np.int64),
            logprobs=np.array([math.nan, -0.5], dtype=np.float32),
            vocab_size=100,
        )
        ft = DummyResponses(token_dists={"preflight": nan_dist})
        bad_backend = DummyDifferentialBackend(base=DummyResponses(), ft=ft)

        spec = _spec(
            {"name": "p1", "kind": "__runner_pass"},
            {"name": "p2", "kind": "__runner_pass"},
        )
        result = run(spec, bad_backend)

        # Exactly one synthetic ERROR probe; no configured probes ran.
        assert len(result.probes) == 1
        assert result.probes[0].kind == "preflight"
        assert result.probes[0].verdict == Verdict.ERROR
        assert "preflight failed" in result.probes[0].message
        assert "ft view" in result.probes[0].message
        # Configured probe names did not run.
        assert "p1" not in {p.name for p in result.probes}

    def test_skip_preflight_flag_runs_suite_anyway(self) -> None:
        import math

        from dlm_sway.core.scoring import TokenDist

        nan_dist = TokenDist(
            token_ids=np.array([1, 2], dtype=np.int64),
            logprobs=np.array([math.nan, -0.5], dtype=np.float32),
            vocab_size=100,
        )
        ft = DummyResponses(token_dists={"preflight": nan_dist})
        bad_backend = DummyDifferentialBackend(base=DummyResponses(), ft=ft)

        spec = _spec({"name": "p1", "kind": "__runner_pass"})
        result = run(spec, bad_backend, skip_preflight=True)
        # Probe ran (ignoring the unhealthy backend).
        assert len(result.probes) == 1
        assert result.probes[0].name == "p1"

    def test_finite_backend_preflight_passes_through(
        self, backend: DummyDifferentialBackend
    ) -> None:
        spec = _spec({"name": "p1", "kind": "__runner_pass"})
        result = run(spec, backend)
        # No synthetic preflight probe injected; configured probe ran.
        assert len(result.probes) == 1
        assert result.probes[0].name == "p1"
