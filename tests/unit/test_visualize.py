"""Tests for :mod:`dlm_sway.visualize`.

Exercises the error path (matplotlib missing) and the happy path when
the module is present by stubbing ``matplotlib.pyplot`` via sys.modules.
"""

from __future__ import annotations

import sys
import types
from datetime import timedelta

import pytest

from dlm_sway.core.errors import BackendNotAvailableError
from dlm_sway.core.result import ProbeResult, SuiteResult, Verdict, utcnow


def _suite_with(*probes: ProbeResult) -> SuiteResult:
    started = utcnow()
    return SuiteResult(
        spec_path="sway.yaml",
        started_at=started,
        finished_at=started + timedelta(seconds=1),
        base_model_id="b",
        adapter_id="a",
        sway_version="0.1.0.dev0",
        probes=probes,
    )


class _FakeFig:
    def tight_layout(self) -> None:  # pragma: no cover — trivial
        return None


class _FakeAx:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def bar(self, *a, **k):  # type: ignore[no-untyped-def]
        self.calls.append("bar")

    def plot(self, *a, **k):  # type: ignore[no-untyped-def]
        self.calls.append("plot")

    def hist(self, *a, **k):  # type: ignore[no-untyped-def]
        self.calls.append("hist")

    def axhline(self, *a, **k):  # type: ignore[no-untyped-def]
        return None

    def axvline(self, *a, **k):  # type: ignore[no-untyped-def]
        return None

    def set_xticks(self, *a, **k):  # type: ignore[no-untyped-def]
        return None

    def set_xticklabels(self, *a, **k):  # type: ignore[no-untyped-def]
        return None

    def set_xlabel(self, *a, **k):  # type: ignore[no-untyped-def]
        return None

    def set_ylabel(self, *a, **k):  # type: ignore[no-untyped-def]
        return None

    def set_title(self, *a, **k):  # type: ignore[no-untyped-def]
        return None

    def legend(self, *a, **k):  # type: ignore[no-untyped-def]
        return None


@pytest.fixture
def fake_mpl(monkeypatch: pytest.MonkeyPatch) -> _FakeAx:
    ax = _FakeAx()

    def _subplots(*a, **k):  # type: ignore[no-untyped-def]
        return _FakeFig(), ax

    plt = types.ModuleType("matplotlib.pyplot")
    plt.subplots = _subplots  # type: ignore[attr-defined]
    mpl_pkg = types.ModuleType("matplotlib")
    monkeypatch.setitem(sys.modules, "matplotlib", mpl_pkg)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", plt)
    return ax


def test_section_sis_plot_uses_per_section_evidence(fake_mpl: _FakeAx) -> None:
    from dlm_sway.visualize import plot_section_sis

    suite = _suite_with(
        ProbeResult(
            name="sis",
            kind="section_internalization",
            verdict=Verdict.PASS,
            score=0.75,
            raw=0.1,
            evidence={
                "per_section": [
                    {
                        "section_id": "a",
                        "kind": "prose",
                        "tag": None,
                        "base_nll": 3.0,
                        "ft_nll": 2.5,
                        "own_lift": 0.17,
                        "leak_lift": 0.02,
                        "effective_sis": 0.15,
                        "passed": True,
                    },
                    {
                        "section_id": "b",
                        "kind": "instruction",
                        "tag": "intro",
                        "base_nll": 4.0,
                        "ft_nll": 3.9,
                        "own_lift": 0.025,
                        "leak_lift": 0.03,
                        "effective_sis": -0.005,
                        "passed": False,
                    },
                ],
                "per_section_threshold": 0.05,
            },
        )
    )
    plot_section_sis(suite)
    assert "bar" in fake_mpl.calls


def test_adapter_ablation_plot(fake_mpl: _FakeAx) -> None:
    from dlm_sway.visualize import plot_adapter_ablation

    suite = _suite_with(
        ProbeResult(
            name="abl",
            kind="adapter_ablation",
            verdict=Verdict.PASS,
            score=0.8,
            raw=0.9,
            evidence={
                "lambdas": [0.0, 0.5, 1.0, 1.25],
                "mean_divergence_per_lambda": [0.0, 0.5, 1.0, 1.1],
                "linearity": 0.91,
                "saturation_lambda": 0.75,
                "overshoot": 1.1,
            },
        )
    )
    plot_adapter_ablation(suite)
    assert "plot" in fake_mpl.calls


def test_kl_histogram_plot(fake_mpl: _FakeAx) -> None:
    from dlm_sway.visualize import plot_kl_histogram

    suite = _suite_with(
        ProbeResult(
            name="dk",
            kind="delta_kl",
            verdict=Verdict.PASS,
            score=0.7,
            raw=0.1,
            evidence={"per_prompt": [0.05, 0.1, 0.12, 0.09, 0.15], "divergence_kind": "js"},
        )
    )
    plot_kl_histogram(suite)
    assert "hist" in fake_mpl.calls


def test_raises_when_matplotlib_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Purge matplotlib modules and block imports.
    for mod in list(sys.modules):
        if mod == "matplotlib" or mod.startswith("matplotlib."):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *a, **k):  # type: ignore[no-untyped-def]
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("matplotlib missing in this venv")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from dlm_sway.visualize import plot_section_sis

    suite = _suite_with()
    with pytest.raises(BackendNotAvailableError):
        plot_section_sis(suite)


def test_raises_when_no_matching_probe(fake_mpl: _FakeAx) -> None:
    from dlm_sway.visualize import plot_section_sis

    suite = _suite_with()  # empty — no section_internalization probe
    with pytest.raises(ValueError, match="section_internalization"):
        plot_section_sis(suite)
