"""Tests for :mod:`dlm_sway.probes.training_drift`."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from dlm_sway.core.result import Verdict
from dlm_sway.probes.base import RunContext, build_probe
from dlm_sway.probes.training_drift import (
    TrainingDriftError,
    _collect_steps,
    _compute_metrics,
    _count_spikes,
    _downsampled_curve,
    _verdict_from_metrics,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_jsonl(
    path: Path, *, banner: bool = True, steps: list[tuple[int, float]] | None = None
) -> Path:
    """Build a dlm-shaped train-*.jsonl fixture file.

    Mirrors the real format:
    - Optional banner line (type=banner).
    - Per-step lines (type=step) with step + loss + lr + grad_norm.
    - No closing line — real runs may crash mid-step; the probe
      should tolerate truncated files.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if banner:
        lines.append(json.dumps({"type": "banner", "run_id": 1, "seed": 42}))
    for step, loss in steps or []:
        lines.append(
            json.dumps({"type": "step", "step": step, "loss": loss, "lr": 1e-4, "grad_norm": 0.5})
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _smooth_decay(num_steps: int = 50) -> list[tuple[int, float]]:
    """Clean exponential decay — should pass every threshold."""
    return [(i, 5.0 * math.exp(-i / 20.0) + 0.1) for i in range(num_steps)]


def _spiky_curve(num_steps: int = 50, *, spike_at: int = 25) -> list[tuple[int, float]]:
    """Clean decay with one obvious loss-increase spike injected.

    The spike is a loss *increase* relative to the prior step (the
    semantically meaningful "training instability" signal — fast
    convergence steps are not instabilities).
    """
    base = _smooth_decay(num_steps)
    out = list(base)
    # Inject a sharp upward spike: loss jumps from ~base value to 5x
    # of itself. The next step's delta back down doesn't count as a
    # spike (loss decreases aren't instabilities by design).
    out[spike_at] = (spike_at, base[spike_at][1] * 5.0)
    return out


# ---------------------------------------------------------------------------
# End-to-end probe behavior
# ---------------------------------------------------------------------------


class TestProbeBehavior:
    def test_pass_on_smooth_curve(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        _write_jsonl(
            store / "logs" / "train-000001-20260101T000000.jsonl",
            steps=_smooth_decay(60),
        )
        probe, spec = build_probe(
            {
                "name": "td",
                "kind": "training_drift",
                "store_path": str(store),
            }
        )
        result = probe.run(spec, RunContext())
        assert result.verdict == Verdict.PASS, result.message
        assert result.evidence["instability_events"] == 0
        assert result.evidence["smoothness"] >= 0.7
        # Initial loss is the curve's first sample, final is the last.
        assert result.evidence["initial_loss"] == pytest.approx(5.1, abs=0.1)
        assert result.evidence["final_loss"] < 1.0

    def test_warn_on_spiky_curve(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        _write_jsonl(store / "logs" / "train-000001-20260101T000000.jsonl", steps=_spiky_curve(60))
        probe, spec = build_probe(
            {
                "name": "td",
                "kind": "training_drift",
                "store_path": str(store),
            }
        )
        result = probe.run(spec, RunContext())
        assert result.verdict == Verdict.WARN
        assert result.evidence["instability_events"] >= 1
        assert "instability_events" in result.message

    def test_skip_when_no_store_path(self) -> None:
        probe, spec = build_probe({"name": "td", "kind": "training_drift"})
        result = probe.run(spec, RunContext())
        assert result.verdict == Verdict.SKIP
        assert "no store_path" in result.message

    def test_skip_when_logs_dir_missing(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        store.mkdir()
        probe, spec = build_probe(
            {
                "name": "td",
                "kind": "training_drift",
                "store_path": str(store),
            }
        )
        result = probe.run(spec, RunContext())
        assert result.verdict == Verdict.SKIP
        assert "no logs/" in result.message

    def test_skip_when_no_jsonl(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        (store / "logs").mkdir(parents=True)
        probe, spec = build_probe(
            {
                "name": "td",
                "kind": "training_drift",
                "store_path": str(store),
            }
        )
        result = probe.run(spec, RunContext())
        assert result.verdict == Verdict.SKIP
        assert "no train-*.jsonl" in result.message

    def test_skip_when_too_few_steps(self, tmp_path: Path) -> None:
        """Default min_steps=10 — a 3-step curve must SKIP, not produce
        a misleading verdict."""
        store = tmp_path / "store"
        _write_jsonl(
            store / "logs" / "train-000001-20260101T000000.jsonl",
            steps=[(0, 5.0), (1, 4.5), (2, 4.0)],
        )
        probe, spec = build_probe(
            {
                "name": "td",
                "kind": "training_drift",
                "store_path": str(store),
            }
        )
        result = probe.run(spec, RunContext())
        assert result.verdict == Verdict.SKIP
        assert "too short" in result.message

    def test_resumed_runs_dedupe_keep_latest(self, tmp_path: Path) -> None:
        """Two log files with overlapping step numbers — second one wins
        (mirrors dlm metrics' resume semantics)."""
        store = tmp_path / "store"
        # First run: steps 0..9 with high losses
        _write_jsonl(
            store / "logs" / "train-000001-20260101T000000.jsonl",
            steps=[(i, 10.0 + i) for i in range(10)],
        )
        # Resumed run: steps 5..14 with low losses (resume picked up,
        # values for 5..9 should overwrite the originals)
        _write_jsonl(
            store / "logs" / "train-000001-20260101T010000.jsonl",
            steps=[(i, 1.0) for i in range(5, 15)],
        )
        probe, spec = build_probe(
            {
                "name": "td",
                "kind": "training_drift",
                "store_path": str(store),
            }
        )
        result = probe.run(spec, RunContext())
        # Final loss should be from the resumed run (1.0), not the
        # first run's step 14 (which doesn't exist).
        assert result.evidence["final_loss"] == 1.0
        # Step 5 should carry the resumed value, not the original.
        curve = result.evidence["curve_sampled"]
        step_5 = next((loss for s, loss in curve if s == 5), None)
        assert step_5 == 1.0, f"step 5 should be from resumed run; got {step_5}"

    def test_curve_downsampled_when_long(self, tmp_path: Path) -> None:
        """A 1500-step run should land in evidence with curve_sampled <= 512."""
        store = tmp_path / "store"
        _write_jsonl(
            store / "logs" / "train-000001-20260101T000000.jsonl",
            steps=[(i, 5.0 * math.exp(-i / 500.0)) for i in range(1500)],
        )
        probe, spec = build_probe(
            {
                "name": "td",
                "kind": "training_drift",
                "store_path": str(store),
            }
        )
        result = probe.run(spec, RunContext())
        assert result.evidence["num_steps"] == 1500
        curve = result.evidence["curve_sampled"]
        assert len(curve) <= 512
        # Endpoints preserved.
        assert curve[0][0] == 0
        assert curve[-1][0] == 1499

    def test_corrupt_first_line_errors(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        log_dir = store / "logs"
        log_dir.mkdir(parents=True)
        (log_dir / "train-000001-20260101T000000.jsonl").write_text(
            "not even json\n", encoding="utf-8"
        )
        probe, spec = build_probe(
            {
                "name": "td",
                "kind": "training_drift",
                "store_path": str(store),
            }
        )
        result = probe.run(spec, RunContext())
        assert result.verdict == Verdict.ERROR
        assert "not valid JSON" in result.message

    def test_truncated_trailing_line_tolerated(self, tmp_path: Path) -> None:
        """A crashed-mid-line trainer leaves a partial JSON tail. The
        probe should consume the good lines and skip the bad one."""
        store = tmp_path / "store"
        log = store / "logs" / "train-000001-20260101T000000.jsonl"
        log.parent.mkdir(parents=True)
        good_lines = [
            json.dumps({"type": "banner", "run_id": 1}),
        ] + [
            json.dumps({"type": "step", "step": i, "loss": 5.0 - i * 0.05, "lr": 1e-4})
            for i in range(60)
        ]
        # Trailing partial line a crashed trainer might emit.
        log.write_text(
            "\n".join(good_lines) + '\n{"type": "step", "step": 60, "lo', encoding="utf-8"
        )
        probe, spec = build_probe(
            {
                "name": "td",
                "kind": "training_drift",
                "store_path": str(store),
            }
        )
        result = probe.run(spec, RunContext())
        # Verdict can be PASS or WARN depending on the curve, but it
        # must not be ERROR — the partial line shouldn't break the run.
        assert result.verdict in {Verdict.PASS, Verdict.WARN}
        assert result.evidence["num_steps"] == 60


# ---------------------------------------------------------------------------
# Pure-math metric helpers
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_smooth_decay_metrics(self) -> None:
        losses = np.array([5.0 * math.exp(-i / 20.0) + 0.1 for i in range(50)])
        m = _compute_metrics(losses, rolling_window=10, spike_sigma=3.0)
        assert m["instability_events"] == 0
        assert m["smoothness"] > 0.95
        assert 0.0 < m["convergence_ratio"] < 0.2

    def test_constant_loss_marked_unsmooth(self) -> None:
        """A perfectly flat curve is NOT 'smooth' — it's a stuck run."""
        losses = np.full(50, 5.0)
        m = _compute_metrics(losses, rolling_window=10, spike_sigma=3.0)
        assert m["smoothness"] == 0.0
        assert m["convergence_ratio"] == 1.0

    def test_nan_loss_counts_as_instability(self) -> None:
        """A NaN in the curve should count as an instability event but
        not crash the metric computation (NaN propagation breaks
        everything otherwise)."""
        losses = np.array([5.0 - i * 0.1 for i in range(50)])
        losses[20] = float("nan")
        m = _compute_metrics(losses, rolling_window=10, spike_sigma=3.0)
        assert m["instability_events"] >= 1
        # The forward-fill kept downstream stats finite.
        assert math.isfinite(m["smoothness"])
        assert math.isfinite(m["final_loss"])

    def test_all_nan_returns_zero_smoothness(self) -> None:
        losses = np.array([float("nan")] * 10)
        m = _compute_metrics(losses, rolling_window=10, spike_sigma=3.0)
        assert m["smoothness"] == 0.0
        assert m["instability_events"] == 10

    def test_zero_initial_loss_returns_inf_ratio(self) -> None:
        """Initial loss of 0 (degenerate) → convergence ratio is inf
        rather than ZeroDivisionError."""
        losses = np.array([0.0, 0.5, 1.0, 1.5])
        m = _compute_metrics(losses, rolling_window=2, spike_sigma=3.0)
        assert m["convergence_ratio"] == float("inf")


class TestCountSpikes:
    def test_no_spikes_in_smoothly_decaying_curve(self) -> None:
        """Loss going down — not an instability, regardless of |Δ|."""
        deltas = np.array([-0.05] * 50)
        assert _count_spikes(deltas, window=10, sigma=3.0) == 0

    def test_no_spikes_in_constant_curve(self) -> None:
        deltas = np.zeros(50)
        assert _count_spikes(deltas, window=10, sigma=3.0) == 0

    def test_loss_increase_outlier_caught(self) -> None:
        """A genuine training spike: loss-up event much larger than typical."""
        deltas = np.array([-0.05] * 50)
        deltas[25] = 1.5  # loss jumped UP
        assert _count_spikes(deltas, window=10, sigma=3.0) == 1

    def test_loss_decrease_outlier_ignored(self) -> None:
        """A 'fast convergence' step (loss going down hard) is NOT an
        instability — only loss-up events count."""
        deltas = np.array([-0.05] * 50)
        deltas[25] = -2.0  # huge negative delta — fast convergence
        assert _count_spikes(deltas, window=10, sigma=3.0) == 0

    def test_short_curve_uses_global_baseline(self) -> None:
        # 5 deltas, window=10 → falls back to global MAD.
        deltas = np.array([-0.01, -0.01, 1.0, -0.01, -0.01])
        spikes = _count_spikes(deltas, window=10, sigma=2.0)
        assert spikes == 1

    def test_empty_deltas_returns_zero(self) -> None:
        assert _count_spikes(np.array([]), window=10, sigma=3.0) == 0


class TestDownsampledCurve:
    def test_short_curve_unchanged(self) -> None:
        steps = np.array([0, 1, 2, 3])
        losses = np.array([5.0, 4.0, 3.0, 2.0])
        out = _downsampled_curve(steps, losses, cap=10)
        assert len(out) == 4
        assert out == [(0, 5.0), (1, 4.0), (2, 3.0), (3, 2.0)]

    def test_long_curve_capped_with_endpoints_preserved(self) -> None:
        steps = np.arange(2000)
        losses = np.linspace(5.0, 0.5, 2000)
        out = _downsampled_curve(steps, losses, cap=100)
        assert len(out) <= 110  # cap with the +1 endpoint allowance
        assert out[0][0] == 0
        assert out[-1][0] == 1999


class TestVerdictFromMetrics:
    def _spec(self, **kwargs: object) -> object:
        from dlm_sway.probes.training_drift import TrainingDriftSpec

        return TrainingDriftSpec(name="td", kind="training_drift", **kwargs)  # type: ignore[arg-type]

    def test_pass_when_all_thresholds_clear(self) -> None:
        spec = self._spec()
        v, _, msg = _verdict_from_metrics(
            {
                "smoothness": 0.9,
                "convergence_ratio": 0.4,
                "instability_events": 0,
                "final_loss": 0.5,
            },
            spec,  # type: ignore[arg-type]
        )
        assert v == Verdict.PASS
        assert "smoothness=0.90" in msg
        assert "warnings:" not in msg

    def test_warn_lists_each_failed_threshold(self) -> None:
        spec = self._spec()
        v, _, msg = _verdict_from_metrics(
            {
                "smoothness": 0.5,
                "convergence_ratio": 0.9,
                "instability_events": 3,
                "final_loss": 4.5,
            },
            spec,  # type: ignore[arg-type]
        )
        assert v == Verdict.WARN
        assert "smoothness=0.50" in msg
        assert "convergence_ratio=0.90" in msg
        assert "instability_events=3" in msg


# ---------------------------------------------------------------------------
# Collect steps (JSONL parsing edge cases)
# ---------------------------------------------------------------------------


class TestRealDlmFixture:
    """Validate the probe against a JSONL captured from a real dlm run.

    The fixture under ``tests/fixtures/dlm_train_log_fixture.jsonl`` is
    a captured-from-disk shape: leading banner, an interleaved
    ``type=delta`` (doc-change record), 30 ``type=step`` records, and
    a closing ``type=run_complete``. If this test breaks, dlm's log
    format has shifted and the probe needs an update — that's
    exactly the regression signal we want.
    """

    def test_parses_real_fixture_to_pass_verdict(self, tmp_path: Path) -> None:
        fixture = (
            Path(__file__).resolve().parent.parent / "fixtures" / "dlm_train_log_fixture.jsonl"
        )
        store = tmp_path / "store"
        store.mkdir()
        (store / "logs").mkdir()
        (store / "logs" / "train-000001-20260426T062514.jsonl").write_bytes(fixture.read_bytes())

        probe, spec = build_probe(
            {
                "name": "td",
                "kind": "training_drift",
                "store_path": str(store),
                # The fixture's tail flattens out (loss converges) so
                # the curve has a stable plateau. Permissive convergence
                # threshold to focus the assertion on format compat.
                "assert_convergence_ratio_lte": 0.5,
            }
        )
        result = probe.run(spec, RunContext())
        assert result.verdict == Verdict.PASS, result.message
        assert result.evidence["num_steps"] == 30
        assert result.evidence["instability_events"] == 0
        # Final loss was 1.911 in the fixture; just check the right
        # ballpark so future fixture tweaks don't spuriously fail.
        assert 1.8 < result.evidence["final_loss"] < 2.0
        assert result.evidence["initial_loss"] > 5.0


class TestCollectSteps:
    def test_filters_non_step_records(self, tmp_path: Path) -> None:
        log = tmp_path / "train-000001.jsonl"
        log.write_text(
            "\n".join(
                [
                    json.dumps({"type": "banner", "run_id": 1}),
                    json.dumps({"type": "step", "step": 0, "loss": 5.0}),
                    json.dumps({"type": "delta", "new": [], "removed": []}),
                    json.dumps({"type": "step", "step": 1, "loss": 4.0}),
                    json.dumps({"type": "run_complete", "elapsed_seconds": 10.0}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        out = _collect_steps([log])
        assert out == {0: 5.0, 1: 4.0}

    def test_missing_step_key_skipped(self, tmp_path: Path) -> None:
        """A 'step' record missing required fields is dropped — the
        parser doesn't crash the run on a single bad record."""
        log = tmp_path / "train.jsonl"
        log.write_text(
            "\n".join(
                [
                    json.dumps({"type": "step", "loss": 5.0}),  # no `step`
                    json.dumps({"type": "step", "step": 1, "loss": 4.0}),
                    json.dumps({"type": "step", "step": 2}),  # no `loss`
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        out = _collect_steps([log])
        assert out == {1: 4.0}

    def test_nan_loss_recorded_as_inf(self, tmp_path: Path) -> None:
        """NaN loss should land as +inf in the curve so the spike
        detector flags it as instability without numpy NaN poisoning."""
        log = tmp_path / "train.jsonl"
        log.write_text(
            "\n".join(
                [
                    json.dumps({"type": "step", "step": 0, "loss": 5.0}),
                    # Real dlm logs encode NaN as the literal NaN; json
                    # itself doesn't permit it, so simulate via Infinity
                    # which json.loads accepts in non-strict mode.
                    '{"type": "step", "step": 1, "loss": NaN}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        out = _collect_steps([log])
        assert out[0] == 5.0
        assert math.isinf(out[1])

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TrainingDriftError, match="failed to read"):
            _collect_steps([tmp_path / "nonexistent.jsonl"])
