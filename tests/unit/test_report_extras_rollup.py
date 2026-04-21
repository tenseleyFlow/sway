"""Tests for the D3 extras-rollup surface.

Covers ``report.collect_missing_extras`` (pure extraction) and the
terminal/markdown renderers' handling of the resulting footer.
"""

from __future__ import annotations

from datetime import UTC, datetime

from dlm_sway.core.result import ProbeResult, SuiteResult, SwayScore, Verdict
from dlm_sway.suite import report


def _suite_with_messages(messages: list[str]) -> SuiteResult:
    now = datetime.now(UTC)
    probes = tuple(
        ProbeResult(
            name=f"p{i}",
            kind="delta_kl",
            verdict=Verdict.SKIP,
            score=None,
            message=msg,
        )
        for i, msg in enumerate(messages)
    )
    return SuiteResult(
        spec_path="<test>",
        started_at=now,
        finished_at=now,
        base_model_id="b",
        adapter_id="a",
        sway_version="0.0.0",
        probes=probes,
    )


class TestCollectMissingExtras:
    def test_single_extra_single_probe(self) -> None:
        suite = _suite_with_messages(
            ["adapter_revert: install the [semsim] extra for sentence embeddings"]
        )
        assert report.collect_missing_extras(suite) == ["semsim"]

    def test_multiple_probes_deduplicated(self) -> None:
        suite = _suite_with_messages(
            [
                "install the [semsim] extra",
                "install the [semsim] extra",
                "install the [style] extra",
            ]
        )
        assert report.collect_missing_extras(suite) == ["semsim", "style"]

    def test_non_skip_messages_ignored(self) -> None:
        now = datetime.now(UTC)
        probes = (
            ProbeResult(
                name="p1",
                kind="delta_kl",
                verdict=Verdict.PASS,
                score=1.0,
                message="install the [semsim] extra",
            ),
        )
        suite = SuiteResult(
            spec_path="<test>",
            started_at=now,
            finished_at=now,
            base_model_id="b",
            adapter_id="a",
            sway_version="0.0.0",
            probes=probes,
        )
        # A PASS probe mentioning install hints in passing must not
        # pollute the rollup.
        assert report.collect_missing_extras(suite) == []

    def test_empty_suite_no_extras(self) -> None:
        now = datetime.now(UTC)
        suite = SuiteResult(
            spec_path="<test>",
            started_at=now,
            finished_at=now,
            base_model_id="b",
            adapter_id="a",
            sway_version="0.0.0",
        )
        assert report.collect_missing_extras(suite) == []


class TestExtrasFooterInMarkdown:
    def test_footer_includes_pip_command(self) -> None:
        suite = _suite_with_messages(
            [
                "adapter_revert: install the [semsim] extra",
                "style_fingerprint: install the [style] extra",
            ]
        )
        score = SwayScore(overall=0.0, components={}, band="noise")
        md = report.to_markdown(suite, score)
        assert "pip install 'dlm-sway[semsim,style]'" in md
        assert "Skipped probes" in md

    def test_no_footer_when_no_skips(self) -> None:
        now = datetime.now(UTC)
        probes = (
            ProbeResult(name="p1", kind="delta_kl", verdict=Verdict.PASS, score=0.9, message="ok"),
        )
        suite = SuiteResult(
            spec_path="<test>",
            started_at=now,
            finished_at=now,
            base_model_id="b",
            adapter_id="a",
            sway_version="0.0.0",
            probes=probes,
        )
        score = SwayScore(overall=0.9, components={}, band="healthy")
        md = report.to_markdown(suite, score)
        assert "Skipped probes" not in md
