"""Tests for :mod:`dlm_sway.probes.base`."""

from __future__ import annotations

from typing import Literal

import pytest

from dlm_sway.core.errors import SpecValidationError
from dlm_sway.core.result import ProbeResult, Verdict
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext, build_probe, registry


class _DummySpec(ProbeSpec):
    kind: Literal["__test_dummy"] = "__test_dummy"
    payload: str = "x"


class _DummyProbe(Probe):
    kind = "__test_dummy"
    spec_cls = _DummySpec
    category = "adherence"

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        assert isinstance(spec, _DummySpec)
        return ProbeResult(
            name=spec.name,
            kind=spec.kind,
            verdict=Verdict.PASS,
            score=1.0,
            message=spec.payload,
        )


class TestRegistry:
    def test_autoregister(self) -> None:
        assert "__test_dummy" in registry()
        assert registry()["__test_dummy"] is _DummyProbe

    def test_duplicate_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate probe kind"):

            class _Clash(Probe):
                kind = "__test_dummy"
                spec_cls = _DummySpec

                def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
                    raise NotImplementedError


class TestBuildProbe:
    def test_valid_entry(self) -> None:
        probe, spec = build_probe({"name": "t", "kind": "__test_dummy", "payload": "hi"})
        assert isinstance(probe, _DummyProbe)
        assert isinstance(spec, _DummySpec)
        assert spec.payload == "hi"

    def test_unknown_kind(self) -> None:
        with pytest.raises(SpecValidationError, match="unknown probe kind"):
            build_probe({"name": "t", "kind": "no_such_kind"})

    def test_missing_kind(self) -> None:
        with pytest.raises(SpecValidationError, match="missing string 'kind'"):
            build_probe({"name": "t"})

    def test_extra_field_forbidden(self) -> None:
        with pytest.raises(SpecValidationError) as exc_info:
            build_probe({"name": "t", "kind": "__test_dummy", "bogus": "y"})
        assert "bogus" in str(exc_info.value).lower()
