"""Tests for :mod:`dlm_sway.suite.report_html` (S12 / F6)."""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import pytest

from dlm_sway.core.result import (
    ProbeResult,
    SuiteResult,
    SwayScore,
    Verdict,
)
from dlm_sway.suite import report_html

SNAPSHOT_DIR = Path(__file__).parent.parent / "snapshots"

# Plotly is shipped via the optional [viz] extra. Skip the whole module
# when it's not importable — the install hint path is covered by the
# CLI test.
pytest.importorskip("plotly")


def _fixture_suite_and_score() -> tuple[SuiteResult, SwayScore]:
    """Suite exercising every panel: section_internalization (SIS bars)
    and adapter_ablation (response curve) both present."""
    started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    finished = datetime(2026, 1, 1, 12, 0, 5, tzinfo=UTC)
    probes = (
        ProbeResult(
            name="dk",
            kind="delta_kl",
            verdict=Verdict.PASS,
            score=0.87,
            raw=0.456,
            z_score=5.12,
            evidence={},
            message="mean js=0.4560, z=+5.12σ vs null",
            duration_s=0.1,
        ),
        ProbeResult(
            name="sis",
            kind="section_internalization",
            verdict=Verdict.PASS,
            score=0.70,
            raw=0.14,
            z_score=3.8,
            evidence={
                "per_section": [
                    {"section_id": "sec01", "effective_sis": 0.18, "passed": True},
                    {"section_id": "sec02", "effective_sis": 0.21, "passed": True},
                    {"section_id": "sec03", "effective_sis": 0.03, "passed": False},
                    {"section_id": "sec04", "effective_sis": 0.10, "passed": True},
                ],
                "num_sections": 4,
                "passing_frac": 0.75,
            },
            message="3/4 sections cleared",
            duration_s=0.3,
        ),
        ProbeResult(
            name="abl",
            kind="adapter_ablation",
            verdict=Verdict.PASS,
            score=0.75,
            raw=0.92,
            z_score=3.5,
            evidence={
                "lambdas": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
                "mean_divergence_per_lambda": [0.0, 0.05, 0.11, 0.16, 0.19, 0.20],
                "linearity": 0.92,
                "saturation_lambda": 0.75,
                "saturation_reason": "found",
                "overshoot": 1.05,
            },
            message="R²=0.92, sat_λ=0.75 (in band), overshoot=1.05",
            duration_s=0.5,
        ),
        ProbeResult(
            name="lk",
            kind="leakage",
            verdict=Verdict.SKIP,
            score=None,
            message="no PROSE sections to test for leakage",
            duration_s=0.0,
        ),
    )
    suite = SuiteResult(
        spec_path="fixture.yaml",
        started_at=started,
        finished_at=finished,
        base_model_id="HuggingFaceTB/SmolLM2-135M",
        adapter_id="adapters/test/v1",
        sway_version="0.1.0",
        probes=probes,
    )
    score = SwayScore(
        overall=0.77,
        components={"adherence": 0.87, "attribution": 0.70, "calibration": 0.0, "ablation": 0.75},
        weights={"adherence": 0.30, "attribution": 0.35, "calibration": 0.20, "ablation": 0.15},
        band="healthy",
    )
    return suite, score


class _WellFormednessChecker(HTMLParser):
    """Trivial subclass: we only use HTMLParser to *not raise*.

    The stdlib parser is tolerant; the test is 'it doesn't blow up.'
    Strict XHTML well-formedness isn't what the browser enforces.
    """

    def error(self, message: str) -> None:  # pragma: no cover — never called with HTMLParser
        raise AssertionError(f"HTMLParser rejected the output: {message}")


def _parse_ok(html_text: str) -> None:
    parser = _WellFormednessChecker(convert_charrefs=True)
    parser.feed(html_text)
    parser.close()


class TestToHtml:
    def test_parses_as_html(self) -> None:
        suite, score = _fixture_suite_and_score()
        out = report_html.to_html(suite, score)
        _parse_ok(out)

    def test_contains_all_probe_names(self) -> None:
        suite, score = _fixture_suite_and_score()
        out = report_html.to_html(suite, score)
        for name in ("dk", "sis", "abl", "lk"):
            assert name in out, f"probe {name!r} not in HTML"

    def test_contains_all_five_panel_divs(self) -> None:
        suite, score = _fixture_suite_and_score()
        out = report_html.to_html(suite, score)
        for div_id in ("sway-gauge", "sway-category", "sway-sis", "sway-ablation", "sway-scatter"):
            assert f'id="{div_id}"' in out, f"panel div {div_id!r} missing"

    def test_plotly_js_inlined_once(self) -> None:
        """The ~3 MB Plotly bundle is embedded, not linked externally.

        Guard: no ``<script src="http..."`` tags exist — everything
        loads from the inline bundle so the page works offline.
        Plotly's bundle body *does* carry the string ``cdn.plot.ly`` as
        an internal default for mapbox config; that's data, not a fetch,
        so we only care about ``<script src=...>`` tags.
        """
        suite, score = _fixture_suite_and_score()
        out = report_html.to_html(suite, score)
        external_scripts = re.findall(r'<script\s+[^>]*src\s*=\s*["\'](https?:[^"\']+)["\']', out)
        assert external_scripts == [], (
            f"HTML pulls in external scripts (should all be inlined): {external_scripts}"
        )
        # Sanity: output is >1 MB (JS bundle is ~3-5 MB — gives us room
        # if Plotly slims down a bit between releases).
        assert len(out) > 1_000_000, f"HTML output suspiciously small: {len(out)} bytes"

    def test_no_sis_panel_when_probe_absent(self) -> None:
        """A suite without section_internalization skips the SIS panel but
        still renders the other four."""
        suite, score = _fixture_suite_and_score()
        pruned_probes = tuple(p for p in suite.probes if p.kind != "section_internalization")
        suite = SuiteResult(
            spec_path=suite.spec_path,
            started_at=suite.started_at,
            finished_at=suite.finished_at,
            base_model_id=suite.base_model_id,
            adapter_id=suite.adapter_id,
            sway_version=suite.sway_version,
            probes=pruned_probes,
        )
        out = report_html.to_html(suite, score)
        assert 'id="sway-sis"' not in out
        assert 'id="sway-ablation"' in out
        assert 'id="sway-scatter"' in out

    def test_zero_probe_suite_still_renders(self) -> None:
        """Empty probes — gauge/category/scatter still emit; no crashes."""
        started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        suite = SuiteResult(
            spec_path="empty.yaml",
            started_at=started,
            finished_at=started,
            base_model_id="base",
            adapter_id="",
            sway_version="0.1.0",
            probes=(),
        )
        score = SwayScore(overall=0.0, components={}, band="noise")
        out = report_html.to_html(suite, score)
        _parse_ok(out)
        assert 'id="sway-gauge"' in out
        assert "no probes ran" in out

    def test_raises_when_plotly_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Simulated ImportError surfaces the install hint."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name.startswith("plotly"):
                raise ImportError("simulated missing plotly")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        suite, score = _fixture_suite_and_score()
        with pytest.raises(RuntimeError, match=r"plotly.*\[viz\]"):
            report_html.to_html(suite, score)


class TestWrapperSnapshot:
    """Snapshot the Sway-owned wrapper, strip the Plotly bundle JS so the
    snapshot doesn't churn on Plotly point releases.
    """

    #: Matches the single ``<script>...plotly_bundle...</script>`` we emit
    #: in ``<head>``. Plotly's per-figure scripts live in the body and
    #: carry the stable chart data — those we *do* want in the snapshot.
    _HEAD_SCRIPT_RE = re.compile(
        r'<script type="text/javascript">\s*/\*\*.*?</script>',
        re.DOTALL,
    )

    def test_snapshot(self) -> None:
        """Run
        ``SWAY_UPDATE_SNAPSHOTS=1 uv run pytest tests/unit/test_report_html.py``
        to regenerate after an intentional wrapper change. Plotly JS
        bundle bumps should NOT drift this — it's stripped before compare.
        """
        suite, score = _fixture_suite_and_score()
        raw = report_html.to_html(suite, score)

        # Strip the Plotly JS bundle; confirm we actually removed it.
        stripped = self._HEAD_SCRIPT_RE.sub(
            '<script type="text/javascript">/* plotly bundle — stripped for snapshot */</script>',
            raw,
            count=1,
        )
        assert stripped != raw, (
            "failed to strip the Plotly JS bundle from the head — regex didn't match"
        )
        # Further shrink: replace per-figure config UUIDs (Plotly sprinkles
        # `"uuid": "..."` in some payloads) to keep snapshot stable across
        # minor Plotly versions.
        stripped = re.sub(r'"uid": ?"[^"]*"', '"uid": "<stripped>"', stripped)

        path = SNAPSHOT_DIR / "report.html"
        if os.environ.get("SWAY_UPDATE_SNAPSHOTS") == "1" or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(stripped, encoding="utf-8")
            pytest.skip(
                "snapshot report.html written — re-run without SWAY_UPDATE_SNAPSHOTS to verify"
            )
        expected = path.read_text(encoding="utf-8")
        assert stripped == expected, (
            "report.html drifted from snapshot.\n"
            "To accept the new output intentionally, run:\n"
            "    SWAY_UPDATE_SNAPSHOTS=1 uv run pytest tests/unit/test_report_html.py\n"
            "and commit the updated file.\n"
        )
