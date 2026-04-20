"""Tests for :mod:`dlm_sway.suite.score` + :mod:`dlm_sway.suite.report`."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Literal

import pytest

from dlm_sway.core.result import ProbeResult, SuiteResult, Verdict, utcnow
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext
from dlm_sway.suite import report, score
from dlm_sway.suite.spec import SwaySpec


class _AdherenceSpec(ProbeSpec):
    kind: Literal["__score_adherence"] = "__score_adherence"


class _AdherenceProbe(Probe):
    kind = "__score_adherence"
    spec_cls = _AdherenceSpec
    category = "adherence"

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        raise NotImplementedError  # never executed; registered for category lookup


class _AttributionSpec(ProbeSpec):
    kind: Literal["__score_attribution"] = "__score_attribution"


class _AttributionProbe(Probe):
    kind = "__score_attribution"
    spec_cls = _AttributionSpec
    category = "attribution"

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        raise NotImplementedError


def _synth_suite(*probes: ProbeResult) -> SuiteResult:
    started = utcnow()
    return SuiteResult(
        spec_path="sway.yaml",
        started_at=started,
        finished_at=started + timedelta(seconds=1),
        base_model_id="base",
        adapter_id="adapter",
        sway_version="0.1.0.dev0",
        probes=probes,
    )


class TestCompute:
    def test_single_passing_probe(self) -> None:
        suite = _synth_suite(
            ProbeResult(name="a", kind="__score_adherence", verdict=Verdict.PASS, score=0.8)
        )
        s = score.compute(suite)
        assert s.overall == pytest.approx(0.8)
        assert s.components["adherence"] == pytest.approx(0.8)
        assert s.band == "healthy"

    def test_mixed_categories_weighted(self) -> None:
        suite = _synth_suite(
            ProbeResult(name="a", kind="__score_adherence", verdict=Verdict.PASS, score=0.9),
            ProbeResult(name="b", kind="__score_attribution", verdict=Verdict.PASS, score=0.3),
        )
        s = score.compute(suite)
        # Active categories: adherence (0.30) + attribution (0.35). Normalized.
        expected = (0.30 * 0.9 + 0.35 * 0.3) / (0.30 + 0.35)
        assert s.overall == pytest.approx(expected)

    def test_errors_and_skips_excluded(self) -> None:
        suite = _synth_suite(
            ProbeResult(name="a", kind="__score_adherence", verdict=Verdict.PASS, score=0.9),
            ProbeResult(name="b", kind="__score_adherence", verdict=Verdict.SKIP, score=None),
            ProbeResult(name="c", kind="__score_adherence", verdict=Verdict.ERROR, score=None),
        )
        s = score.compute(suite)
        assert s.components["adherence"] == pytest.approx(0.9)

    def test_per_probe_weights_override_uniform(self) -> None:
        suite = _synth_suite(
            ProbeResult(
                name="a",
                kind="__score_adherence",
                verdict=Verdict.PASS,
                score=1.0,
                evidence={"weight": 3.0},
            ),
            ProbeResult(
                name="b",
                kind="__score_adherence",
                verdict=Verdict.PASS,
                score=0.0,
                evidence={"weight": 1.0},
            ),
        )
        s = score.compute(suite)
        # Weighted mean: (3·1 + 1·0) / 4 = 0.75
        assert s.components["adherence"] == pytest.approx(0.75)

    def test_failed_probe_surfaces_in_findings(self) -> None:
        suite = _synth_suite(
            ProbeResult(
                name="bad",
                kind="__score_adherence",
                verdict=Verdict.FAIL,
                score=0.1,
                message="nope",
            )
        )
        s = score.compute(suite)
        assert any("bad" in f for f in s.findings)


class TestJsonReport:
    def test_schema_fields(self) -> None:
        suite = _synth_suite(
            ProbeResult(
                name="p1",
                kind="__score_adherence",
                verdict=Verdict.PASS,
                score=0.75,
                raw=0.12,
                z_score=3.1,
            )
        )
        s = score.compute(suite)
        out = json.loads(report.to_json(suite, s))
        assert out["schema_version"] == 1
        assert out["score"]["overall"] == pytest.approx(0.75)
        assert out["probes"][0]["verdict"] == "pass"
        assert out["probes"][0]["z_score"] == pytest.approx(3.1)


class TestJunit:
    def test_counts_populated(self) -> None:
        suite = _synth_suite(
            ProbeResult(name="p1", kind="__score_adherence", verdict=Verdict.PASS, score=1.0),
            ProbeResult(name="p2", kind="__score_adherence", verdict=Verdict.FAIL, score=0.0),
            ProbeResult(
                name="p3",
                kind="__score_adherence",
                verdict=Verdict.ERROR,
                score=None,
            ),
        )
        s = score.compute(suite)
        xml = report.to_junit(suite, s)
        assert 'tests="3"' in xml
        assert 'failures="1"' in xml
        assert 'errors="1"' in xml
        assert "<failure" in xml
        assert "<error" in xml


class TestMarkdown:
    def test_contains_probe_table(self) -> None:
        suite = _synth_suite(
            ProbeResult(name="p1", kind="__score_adherence", verdict=Verdict.PASS, score=0.8)
        )
        s = score.compute(suite)
        md = report.to_markdown(suite, s)
        assert "dlm-sway report" in md
        assert "| p1 | `__score_adherence`" in md


# Force the SwaySpec model to stay reachable from tests (keeps mypy happy
# on the eventual CLI path that calls into both).
assert SwaySpec is not None
