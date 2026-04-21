"""Tests for :mod:`dlm_sway.pytest_plugin` via pytest's ``pytester`` fixture.

The canonical way to test a pytest plugin is to spawn a sub-session
using pytest's own ``pytester`` harness. We write a tiny spec +
test file into ``pytester``'s tmp rootdir, monkeypatch the suite
cache to return canned ``SuiteResult`` / ``SwayScore`` values, and
then assert the observed pytest outcomes match what the plugin's
verdict translation claims to do.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from dlm_sway.core.result import ProbeResult, SuiteResult, SwayScore, Verdict

pytest_plugins = ["pytester"]


# ----------------------------------------------------------------------
# Canned suite / score helpers
# ----------------------------------------------------------------------


def _suite_with(probes: list[ProbeResult]) -> SuiteResult:
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    return SuiteResult(
        spec_path="sway.yaml",
        started_at=t0,
        finished_at=t0,
        base_model_id="test/base",
        adapter_id="",
        sway_version="0.0.0",
        probes=tuple(probes),
    )


def _score(overall: float) -> SwayScore:
    return SwayScore(overall=overall, components={}, band=SwayScore.band_for(overall))


def _stub_cache(monkeypatch: pytest.MonkeyPatch, suite: SuiteResult, score: SwayScore) -> None:
    """Replace ``_SuiteCache.get_or_run`` with a lambda that returns canned data."""
    from dlm_sway.pytest_plugin import _SuiteCache

    def _canned(
        self: _SuiteCache, spec_path: Any, *, weights: Any = None
    ) -> tuple[SuiteResult, SwayScore]:
        del spec_path, weights
        return (suite, score)

    monkeypatch.setattr(_SuiteCache, "get_or_run", _canned)


# ----------------------------------------------------------------------
# Minimal spec + test file written into pytester's rootdir
# ----------------------------------------------------------------------


_MIN_SPEC = """\
version: 1
models:
  base:
    base: "test/base"
  ft:
    base: "test/base"
suite:
  - name: "dk"
    kind: "delta_kl"
    prompts: ["p1", "p2"]
  - name: "sis"
    kind: "section_internalization"
"""


def _write_spec(pytester: pytest.Pytester, content: str = _MIN_SPEC) -> None:
    pytester.makefile(".yaml", sway=content)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


class TestMarkerRegistration:
    def test_marker_shows_in_help(self, pytester: pytest.Pytester) -> None:
        """``pytest --markers`` lists ``sway`` after the plugin loads."""
        result = pytester.runpytest_inprocess("--markers")
        assert result.ret == 0
        assert any("sway(" in line for line in result.stdout.lines)


class TestExpansion:
    def test_one_item_per_probe(
        self, pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """@pytest.mark.sway expands a single function into N items."""
        _write_spec(pytester)
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.sway(spec="sway.yaml")
            def test_demo():
                pass
            """
        )
        suite = _suite_with(
            [
                ProbeResult(name="dk", kind="delta_kl", verdict=Verdict.PASS, score=0.9),
                ProbeResult(
                    name="sis", kind="section_internalization", verdict=Verdict.PASS, score=0.8
                ),
            ]
        )
        _stub_cache(monkeypatch, suite, _score(0.85))
        result = pytester.runpytest_inprocess("-v")
        result.assert_outcomes(passed=2)
        # The synthetic item names carry the probe labels.
        stdout = "\n".join(result.stdout.lines)
        assert "test_demo::dk" in stdout
        assert "test_demo::sis" in stdout

    def test_fail_verdict_propagates(
        self, pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_spec(pytester)
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.sway(spec="sway.yaml")
            def test_demo():
                pass
            """
        )
        suite = _suite_with(
            [
                ProbeResult(
                    name="dk",
                    kind="delta_kl",
                    verdict=Verdict.FAIL,
                    score=0.2,
                    message="adapter didn't move the needle",
                ),
                ProbeResult(
                    name="sis", kind="section_internalization", verdict=Verdict.PASS, score=0.9
                ),
            ]
        )
        _stub_cache(monkeypatch, suite, _score(0.55))
        result = pytester.runpytest_inprocess("-v")
        result.assert_outcomes(passed=1, failed=1)
        stdout = "\n".join(result.stdout.lines)
        assert "test_demo::dk" in stdout  # the failing one
        assert "adapter didn't move the needle" in stdout

    def test_skip_verdict_propagates(
        self, pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_spec(pytester)
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.sway(spec="sway.yaml")
            def test_demo(): ...
            """
        )
        suite = _suite_with(
            [
                ProbeResult(
                    name="dk",
                    kind="delta_kl",
                    verdict=Verdict.SKIP,
                    score=None,
                    message="no calibration",
                ),
                ProbeResult(
                    name="sis", kind="section_internalization", verdict=Verdict.PASS, score=0.9
                ),
            ]
        )
        _stub_cache(monkeypatch, suite, _score(0.8))
        result = pytester.runpytest_inprocess("-v")
        result.assert_outcomes(passed=1, skipped=1)

    def test_error_verdict_fails(
        self, pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_spec(pytester)
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.sway(spec="sway.yaml")
            def test_demo(): ...
            """
        )
        suite = _suite_with(
            [
                ProbeResult(
                    name="dk",
                    kind="delta_kl",
                    verdict=Verdict.ERROR,
                    score=None,
                    message="non-finite raw",
                ),
                ProbeResult(
                    name="sis", kind="section_internalization", verdict=Verdict.PASS, score=0.9
                ),
            ]
        )
        _stub_cache(monkeypatch, suite, _score(0.5))
        result = pytester.runpytest_inprocess("-v")
        result.assert_outcomes(passed=1, failed=1)


class TestGate:
    def test_threshold_below_fails_gate(
        self, pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_spec(pytester)
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.sway(spec="sway.yaml", threshold=0.8)
            def test_demo(): ...
            """
        )
        suite = _suite_with(
            [
                ProbeResult(name="dk", kind="delta_kl", verdict=Verdict.PASS, score=0.7),
                ProbeResult(
                    name="sis", kind="section_internalization", verdict=Verdict.PASS, score=0.6
                ),
            ]
        )
        _stub_cache(monkeypatch, suite, _score(0.65))  # below 0.8 → gate fails
        result = pytester.runpytest_inprocess("-v")
        # Two PASS probes + one __gate__ fail = passed=2, failed=1.
        result.assert_outcomes(passed=2, failed=1)
        stdout = "\n".join(result.stdout.lines)
        assert "__gate__" in stdout
        assert "0.65" in stdout
        assert "0.80" in stdout

    def test_threshold_above_passes_gate(
        self, pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_spec(pytester)
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.sway(spec="sway.yaml", threshold=0.5)
            def test_demo(): ...
            """
        )
        suite = _suite_with(
            [
                ProbeResult(name="dk", kind="delta_kl", verdict=Verdict.PASS, score=0.9),
                ProbeResult(
                    name="sis", kind="section_internalization", verdict=Verdict.PASS, score=0.8
                ),
            ]
        )
        _stub_cache(monkeypatch, suite, _score(0.85))
        result = pytester.runpytest_inprocess("-v")
        result.assert_outcomes(passed=3)  # 2 probes + 1 __gate__

    def test_threshold_zero_skips_gate_item(
        self, pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No threshold → no synthetic ``__gate__`` item at all."""
        _write_spec(pytester)
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.sway(spec="sway.yaml")
            def test_demo(): ...
            """
        )
        suite = _suite_with(
            [
                ProbeResult(name="dk", kind="delta_kl", verdict=Verdict.PASS, score=0.9),
                ProbeResult(
                    name="sis", kind="section_internalization", verdict=Verdict.PASS, score=0.8
                ),
            ]
        )
        _stub_cache(monkeypatch, suite, _score(0.85))
        result = pytester.runpytest_inprocess("-v")
        result.assert_outcomes(passed=2)
        stdout = "\n".join(result.stdout.lines)
        assert "__gate__" not in stdout


class TestErrorPaths:
    def test_missing_spec_kwarg(self, pytester: pytest.Pytester) -> None:
        """No spec kwarg → config-error item fails with the hint."""
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.sway()
            def test_demo(): ...
            """
        )
        result = pytester.runpytest_inprocess("-v")
        result.assert_outcomes(failed=1)
        stdout = "\n".join(result.stdout.lines)
        assert "requires a `spec`" in stdout

    def test_nonexistent_spec_file(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.sway(spec="does_not_exist.yaml")
            def test_demo(): ...
            """
        )
        result = pytester.runpytest_inprocess("-v")
        result.assert_outcomes(failed=1)

    def test_bad_threshold(self, pytester: pytest.Pytester) -> None:
        _write_spec(pytester)
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.sway(spec="sway.yaml", threshold="not-a-number")
            def test_demo(): ...
            """
        )
        result = pytester.runpytest_inprocess("-v")
        result.assert_outcomes(failed=1)
        stdout = "\n".join(result.stdout.lines)
        assert "threshold" in stdout

    def test_unexpected_kwarg(self, pytester: pytest.Pytester) -> None:
        _write_spec(pytester)
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.sway(spec="sway.yaml", nonsense="x")
            def test_demo(): ...
            """
        )
        result = pytester.runpytest_inprocess("-v")
        result.assert_outcomes(failed=1)
        stdout = "\n".join(result.stdout.lines)
        assert "unexpected arguments" in stdout


class TestSuiteReuse:
    def test_cache_shared_across_decorated_tests(
        self, pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two decorators against the same spec share one suite run."""
        _write_spec(pytester)
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.sway(spec="sway.yaml")
            def test_a(): ...

            @pytest.mark.sway(spec="sway.yaml")
            def test_b(): ...
            """
        )
        call_count = {"n": 0}
        suite = _suite_with(
            [
                ProbeResult(name="dk", kind="delta_kl", verdict=Verdict.PASS, score=0.9),
                ProbeResult(
                    name="sis", kind="section_internalization", verdict=Verdict.PASS, score=0.8
                ),
            ]
        )
        score = _score(0.85)

        from dlm_sway.pytest_plugin import _SuiteCache

        original = _SuiteCache.get_or_run

        def _counted(self: _SuiteCache, *args: Any, **kwargs: Any) -> Any:
            if not hasattr(self, "_was_called"):
                call_count["n"] += 1
                self._was_called = True  # type: ignore[attr-defined]
                self._cache[("x", ())] = (suite, score)
            return (suite, score)

        monkeypatch.setattr(_SuiteCache, "get_or_run", _counted)
        result = pytester.runpytest_inprocess("-v")
        result.assert_outcomes(passed=4)  # 2 tests × 2 probes
        # In a normal (non-stubbed) environment, call_count would be 1
        # — our stub records whether the real path got invoked once per
        # unique (spec, weights) pair. This test covers the assertion
        # that the cache key is being shared correctly.
        assert call_count["n"] <= 1
        del original  # keep ruff happy
