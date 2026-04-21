"""Tests for the multi-rank null-adapter calibration path (S10 / F4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.result import Verdict
from dlm_sway.probes._zscore import format_z_profile, z_scores_by_rank
from dlm_sway.probes.base import RunContext, build_probe
from dlm_sway.suite.runner import run as run_suite
from dlm_sway.suite.spec import SwaySpec


def _backend() -> DummyDifferentialBackend:
    return DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())


class TestDummyBackendRankScale:
    def test_rank_scale_scales_noise_std(self) -> None:
        """sqrt(rank_scale) scales the null-view perturbation std."""
        backend = _backend()
        prompt = "hello"

        # Collect 40 samples at each rank_scale to estimate std.
        def _std_at(rank_scale: float) -> float:
            lps = []
            for seed in range(40):
                with backend.as_null_adapter(seed=seed, rank_scale=rank_scale) as view:
                    d = view.next_token_dist(prompt, top_k=8)
                lps.append(d.logprobs)
            arr = np.asarray(lps)
            # Variance across seeds at each position, averaged.
            return float(np.mean(np.std(arr, axis=0)))

        std_1 = _std_at(1.0)
        std_half = _std_at(0.5)
        std_2 = _std_at(2.0)
        # std ∝ sqrt(rank_scale). Tolerance loose because of the
        # top-k renorm and seed discretion.
        assert std_half < std_1 < std_2
        # Ratio should be roughly sqrt(2) with 20% tolerance.
        ratio_up = std_2 / std_1
        ratio_down = std_1 / std_half
        assert 1.15 < ratio_up < 1.75, f"2x ratio={ratio_up}"
        assert 1.15 < ratio_down < 1.75, f"0.5x ratio={ratio_down}"

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_non_positive_rank_scale(self, bad: float) -> None:
        backend = _backend()
        with pytest.raises(ValueError, match="rank_scale"):
            with backend.as_null_adapter(seed=0, rank_scale=bad):
                pass

    def test_rank_scale_1_preserves_pre_s10_behavior(self) -> None:
        """rank_scale=1.0 → identical output to calling without the kwarg."""
        backend = _backend()
        with backend.as_null_adapter(seed=7) as v1:
            d1 = v1.next_token_dist("hello", top_k=8)
        with backend.as_null_adapter(seed=7, rank_scale=1.0) as v2:
            d2 = v2.next_token_dist("hello", top_k=8)
        np.testing.assert_array_equal(d1.logprobs, d2.logprobs)


class TestNullProbeMultiRank:
    def test_single_rank_default_matches_pre_s10(self) -> None:
        """Default rank_multipliers=[1.0] produces the same shape of
        evidence as pre-S10 + a null_stats_by_rank with one entry."""
        backend = _backend()
        probe, spec = build_probe(
            {
                "name": "null",
                "kind": "null_adapter",
                "runs": 3,
                "calibrate_kinds": ["delta_kl"],
            }
        )
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.PASS
        stats = result.evidence["null_stats"]
        by_rank = result.evidence["null_stats_by_rank"]
        assert "delta_kl" in stats
        assert set(by_rank) == {"rank_1.00"}
        assert by_rank["rank_1.00"] == stats

    def test_three_ranks_produce_three_groups(self) -> None:
        backend = _backend()
        probe, spec = build_probe(
            {
                "name": "null",
                "kind": "null_adapter",
                "runs": 3,
                "rank_multipliers": [0.5, 1.0, 2.0],
                "calibrate_kinds": ["delta_kl"],
            }
        )
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.PASS
        by_rank = result.evidence["null_stats_by_rank"]
        assert set(by_rank) == {"rank_0.50", "rank_1.00", "rank_2.00"}
        for rkey, kind_stats in by_rank.items():
            assert "delta_kl" in kind_stats, f"{rkey} missing delta_kl"
            assert kind_stats["delta_kl"]["std"] > 0.0

    def test_rank_0_and_negative_rejected(self) -> None:
        backend = _backend()
        probe, spec = build_probe(
            {
                "name": "null",
                "kind": "null_adapter",
                "runs": 2,
                "rank_multipliers": [1.0, -0.5],
                "calibrate_kinds": ["delta_kl"],
            }
        )
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.ERROR
        assert "rank_multipliers" in (result.message or "")

    def test_higher_rank_has_larger_null_std(self) -> None:
        """A 2x rank null should show more delta_kl variance than a 0.5x one."""
        backend = _backend()
        probe, spec = build_probe(
            {
                "name": "null",
                "kind": "null_adapter",
                "runs": 5,
                "rank_multipliers": [0.5, 2.0],
                "calibrate_kinds": ["delta_kl"],
                "cache": False,
            }
        )
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        by_rank = result.evidence["null_stats_by_rank"]
        std_half = by_rank["rank_0.50"]["delta_kl"]["std"]
        std_2 = by_rank["rank_2.00"]["delta_kl"]["std"]
        assert std_2 > std_half, f"2x std={std_2} not > 0.5x std={std_half}"


class TestRunnerThreadsNullStatsByRank:
    def test_delta_kl_emits_z_by_rank(self) -> None:
        """null_adapter → delta_kl: evidence carries z_by_rank with three entries."""
        backend = _backend()
        raw_spec = SwaySpec.model_validate(
            {
                "version": 1,
                "models": {
                    "base": {"base": "b"},
                    "ft": {"base": "b", "adapter": "/tmp/a"},
                },
                "suite": [
                    {
                        "name": "null",
                        "kind": "null_adapter",
                        "runs": 3,
                        "rank_multipliers": [0.5, 1.0, 2.0],
                        "cache": False,
                    },
                    {
                        "name": "dk",
                        "kind": "delta_kl",
                        "prompts": ["p1", "p2"],
                        "assert_z_gte": -100.0,  # permissive
                    },
                ],
            }
        )
        result = run_suite(raw_spec, backend)
        assert len(result.probes) == 2
        dk = result.probes[1]
        z_by_rank = dk.evidence.get("z_by_rank")
        assert z_by_rank is not None
        assert set(z_by_rank) == {"rank_0.50", "rank_1.00", "rank_2.00"}
        # Each z is finite.
        for z in z_by_rank.values():
            assert math.isfinite(z)


class TestZScoreHelpers:
    def test_z_scores_by_rank_positive_sign(self) -> None:
        raw = 1.0
        stats_by_rank = {
            "rank_1.00": {"mean": 0.5, "std": 0.1, "n": 3.0},
            "rank_0.50": {"mean": 0.3, "std": 0.1, "n": 3.0},
        }
        z = z_scores_by_rank(raw, stats_by_rank, sign=+1)
        assert z is not None
        assert abs(z["rank_1.00"] - 5.0) < 1e-9
        assert abs(z["rank_0.50"] - 7.0) < 1e-9

    def test_z_scores_by_rank_negative_sign(self) -> None:
        """Lower-is-better probes invert the sign."""
        raw = 0.1
        stats_by_rank = {"rank_1.00": {"mean": 0.5, "std": 0.1, "n": 3.0}}
        z = z_scores_by_rank(raw, stats_by_rank, sign=-1)
        assert z is not None
        assert abs(z["rank_1.00"] - 4.0) < 1e-9  # -((0.1 - 0.5)/0.1) = 4

    def test_z_scores_by_rank_none_on_empty(self) -> None:
        assert z_scores_by_rank(0.0, None) is None
        assert z_scores_by_rank(0.0, {}) is None

    def test_z_scores_by_rank_drops_degenerate_ranks(self) -> None:
        """Ranks with std < MIN_STD silently drop out."""
        stats_by_rank = {
            "rank_1.00": {"mean": 0.0, "std": 0.1, "n": 3.0},
            "rank_0.50": {"mean": 0.0, "std": 1e-9, "n": 3.0},  # degenerate
        }
        z = z_scores_by_rank(1.0, stats_by_rank, sign=+1)
        assert z is not None
        assert set(z) == {"rank_1.00"}

    def test_format_z_profile_readable_labels(self) -> None:
        s = format_z_profile(
            {"rank_1.00": 4.2, "rank_0.50": 6.8, "rank_2.00": 2.1},
        )
        assert "+4.20σ @ 1x" in s
        assert "+6.80σ @ 0.5x" in s
        assert "+2.10σ @ 2x" in s
        assert " / " in s

    def test_format_z_profile_empty(self) -> None:
        assert format_z_profile(None) == ""
        assert format_z_profile({}) == ""


class TestProveTheValueRankSaturation:
    """S10 prove-the-value (§F4): rank profile reveals adapter saturation.

    The dummy backend's null view injects noise scaled by
    ``sqrt(rank_scale)`` into ``next_token_dist``. That scales the
    null distribution of ``delta_kl``'s raw metric (mean JS divergence
    across prompts) so that smaller ``rank_scale`` → tighter null →
    larger z at the same adapter divergence.

    Test: hold the adapter fixed (ft responses that produce a known
    divergence from base), vary rank_scale across {0.5, 1.0, 2.0}, and
    assert z_0.5 > z_1 > z_2 — exactly the signature of a rank-sized
    adapter: stronger signal vs a smaller-rank null, weaker signal vs
    a larger-rank null.
    """

    def test_rank_profile_monotone_in_inverse_rank(self) -> None:
        backend = _backend()
        raw_spec = SwaySpec.model_validate(
            {
                "version": 1,
                "models": {
                    "base": {"base": "b"},
                    "ft": {"base": "b", "adapter": "/tmp/a"},
                },
                "suite": [
                    {
                        "name": "null",
                        "kind": "null_adapter",
                        "runs": 5,
                        "rank_multipliers": [0.5, 1.0, 2.0],
                        "cache": False,
                    },
                    {
                        "name": "dk",
                        "kind": "delta_kl",
                        "prompts": ["p1", "p2", "p3", "p4"],
                        "assert_z_gte": -100.0,  # permissive
                    },
                ],
            }
        )
        result = run_suite(raw_spec, backend)
        dk = result.probes[1]
        z_by_rank = dk.evidence["z_by_rank"]
        # Smaller rank → tighter null → larger (more positive) z.
        z_half = z_by_rank["rank_0.50"]
        z_1 = z_by_rank["rank_1.00"]
        z_2 = z_by_rank["rank_2.00"]
        assert z_half > z_1 > z_2, (
            f"expected z monotone-decreasing in rank; got "
            f"0.5x={z_half:.2f}, 1x={z_1:.2f}, 2x={z_2:.2f}"
        )
