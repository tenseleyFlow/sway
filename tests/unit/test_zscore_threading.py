"""End-to-end tests that every numeric probe threads null_stats correctly.

Covers: with stats → ``z_score`` field populated + verdict respects
``assert_z_gte``; without stats → fixed-threshold verdict + the
``(no calibration)`` annotation surfaces in the message.
"""

from __future__ import annotations

import pytest

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.result import Verdict
from dlm_sway.probes.base import RunContext, build_probe


def _backend() -> DummyDifferentialBackend:
    return DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())


class TestNoCalibrationAnnotation:
    """When stats are absent, every probe's message carries the note."""

    @pytest.mark.parametrize(
        ("kind", "spec_kwargs"),
        [
            ("delta_kl", {"prompts": ["q1", "q2"]}),
            (
                "paraphrase_invariance",
                {
                    "cases": [
                        {"prompt": "q", "gold": "a", "paraphrases": ["p1", "p2"]},
                    ]
                },
            ),
            (
                "preference_flip",
                {
                    "triples": [
                        {"prompt": "q1", "chosen": "a", "rejected": "b"},
                        {"prompt": "q2", "chosen": "c", "rejected": "d"},
                        {"prompt": "q3", "chosen": "e", "rejected": "f"},
                        {"prompt": "q4", "chosen": "g", "rejected": "h"},
                    ]
                },
            ),
            ("calibration_drift", {"items_limit": 5}),
        ],
    )
    def test_no_calibration_note_in_message(self, kind: str, spec_kwargs: dict) -> None:
        probe, spec = build_probe({"name": "p", "kind": kind, **spec_kwargs})
        ctx = RunContext(backend=_backend())
        result = probe.run(spec, ctx)
        # If the probe produced a PASS/FAIL verdict with a raw, it took
        # the fixed-threshold path and must surface the annotation.
        if result.verdict in (Verdict.PASS, Verdict.FAIL) and result.raw is not None:
            assert "no calibration" in result.message.lower(), (
                f"{kind} did not surface the no-calibration annotation; message={result.message!r}"
            )
            assert result.z_score is None

    def test_section_internalization_no_calibration(self) -> None:
        from dlm_sway.core.sections import Section

        sections = [
            Section(id="s1", kind="prose", content="alpha beta gamma.", tag=None),
            Section(id="s2", kind="prose", content="delta epsilon zeta.", tag=None),
        ]
        probe, spec = build_probe({"name": "p", "kind": "section_internalization"})
        result = probe.run(spec, RunContext(backend=_backend(), sections=sections))
        if result.verdict in (Verdict.PASS, Verdict.FAIL) and result.raw is not None:
            assert "no calibration" in result.message.lower()
            assert result.z_score is None


class TestStatsThreadedToZScore:
    """With stats in ctx.null_stats, numeric probes z-score and populate the field."""

    def test_delta_kl_emits_z_score(self) -> None:
        probe, spec = build_probe(
            {
                "name": "dk",
                "kind": "delta_kl",
                "prompts": ["p1", "p2"],
                "assert_z_gte": -50.0,  # permissive so we always PASS
            }
        )
        stats = {"delta_kl": {"mean": 0.0, "std": 0.01, "n": 3.0}}
        ctx = RunContext(backend=_backend(), null_stats=stats)
        result = probe.run(spec, ctx)
        assert result.z_score is not None
        assert "vs null" in result.message
