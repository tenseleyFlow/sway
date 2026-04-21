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


class TestNullOptOutsRollup:
    """F15 — surface ``null_adapter.evidence["skipped_kinds"]`` in the report."""

    def _suite_with_null_opt_outs(self, skipped: list[str]) -> SuiteResult:
        now = datetime.now(UTC)
        probes = (
            ProbeResult(
                name="null",
                kind="null_adapter",
                verdict=Verdict.PASS,
                score=1.0,
                evidence={"skipped_kinds": skipped},
            ),
            ProbeResult(name="dk", kind="delta_kl", verdict=Verdict.PASS, score=0.9, message="ok"),
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

    def test_collect_deduplicates_and_sorts(self) -> None:
        suite = self._suite_with_null_opt_outs(
            ["adapter_revert", "prompt_collapse", "adapter_revert"]
        )
        assert report.collect_null_opt_outs(suite) == ["adapter_revert", "prompt_collapse"]

    def test_empty_when_no_null_adapter(self) -> None:
        now = datetime.now(UTC)
        probes = (
            ProbeResult(name="dk", kind="delta_kl", verdict=Verdict.PASS, score=0.9, message="ok"),
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
        assert report.collect_null_opt_outs(suite) == []

    def test_markdown_section_appears(self) -> None:
        suite = self._suite_with_null_opt_outs(["adapter_revert", "prompt_collapse"])
        score = SwayScore(overall=0.9, components={}, band="healthy")
        md = report.to_markdown(suite, score)
        assert "Null-calibration opt-outs" in md
        assert "`adapter_revert`" in md
        assert "`prompt_collapse`" in md

    def test_markdown_omits_section_when_none(self) -> None:
        suite = self._suite_with_null_opt_outs([])
        score = SwayScore(overall=0.9, components={}, band="healthy")
        md = report.to_markdown(suite, score)
        assert "Null-calibration opt-outs" not in md


class TestDegenerateNullRollup:
    """F02 (Audit 03) — probes whose null-calibration ran but produced
    a degenerate baseline (std ≈ 0, typically ``runs: 1``) surface in
    a separate footer rollup so the user sees the actionable fix."""

    def _suite(self, null_stats: dict[str, dict[str, float]]) -> SuiteResult:
        now = datetime.now(UTC)
        probes = (
            ProbeResult(name="null", kind="null_adapter", verdict=Verdict.PASS, score=1.0),
            ProbeResult(name="dk", kind="delta_kl", verdict=Verdict.PASS, score=0.5, message="ok"),
        )
        return SuiteResult(
            spec_path="<test>",
            started_at=now,
            finished_at=now,
            base_model_id="b",
            adapter_id="a",
            sway_version="0.0.0",
            probes=probes,
            null_stats=null_stats,
        )

    def test_degenerate_flag_surfaces_in_rollup(self) -> None:
        suite = self._suite(
            {
                "delta_kl": {"mean": 0.01, "std": 1e-6, "n": 1.0, "degenerate": 1.0},
                "leakage": {"mean": 0.0, "std": 1e-6, "n": 1.0, "degenerate": 1.0},
            }
        )
        assert report.collect_degenerate_null_kinds(suite) == ["delta_kl", "leakage"]

    def test_non_degenerate_stats_excluded(self) -> None:
        suite = self._suite(
            {
                "delta_kl": {"mean": 0.01, "std": 0.005, "n": 3.0, "degenerate": 0.0},
            }
        )
        assert report.collect_degenerate_null_kinds(suite) == []

    def test_no_null_adapter_probe_returns_empty(self) -> None:
        now = datetime.now(UTC)
        suite = SuiteResult(
            spec_path="<test>",
            started_at=now,
            finished_at=now,
            base_model_id="b",
            adapter_id="a",
            sway_version="0.0.0",
            probes=(ProbeResult(name="dk", kind="delta_kl", verdict=Verdict.PASS, score=0.9),),
        )
        assert report.collect_degenerate_null_kinds(suite) == []

    def test_markdown_section_appears_when_degenerate(self) -> None:
        suite = self._suite({"leakage": {"mean": 0.0, "std": 1e-6, "n": 1.0, "degenerate": 1.0}})
        score = SwayScore(overall=0.9, components={}, band="healthy")
        md = report.to_markdown(suite, score)
        assert "Degenerate null calibration" in md
        assert "`leakage`" in md
        assert "bump `runs:`" in md

    def test_markdown_omits_section_when_none_degenerate(self) -> None:
        suite = self._suite({"delta_kl": {"mean": 0.0, "std": 0.01, "n": 3.0, "degenerate": 0.0}})
        score = SwayScore(overall=0.9, components={}, band="healthy")
        md = report.to_markdown(suite, score)
        assert "Degenerate null calibration" not in md
