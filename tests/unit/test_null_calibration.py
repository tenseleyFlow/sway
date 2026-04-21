"""Tests for null-adapter calibration.

Covers: dummy backend ``as_null_adapter`` yields a plausibly noisy
view; ``NullAdapterProbe`` populates ``ctx.null_stats`` in a way
downstream probes pick up end-to-end; missing-capability SKIP path.
"""

from __future__ import annotations

import numpy as np
import pytest

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.result import Verdict
from dlm_sway.core.scoring import NullCalibratedBackend
from dlm_sway.probes.base import RunContext, build_probe
from dlm_sway.suite.runner import run as run_suite
from dlm_sway.suite.spec import SwaySpec


def _diverging_backend() -> DummyDifferentialBackend:
    base = DummyResponses()
    ft = DummyResponses()
    return DummyDifferentialBackend(base=base, ft=ft)


class TestProtocolConformance:
    def test_dummy_is_null_calibrated(self) -> None:
        assert isinstance(_diverging_backend(), NullCalibratedBackend)


class TestAsNullAdapter:
    def test_yields_perturbed_view(self) -> None:
        backend = _diverging_backend()
        with backend.as_base() as base:
            base_dist = base.next_token_dist("hello")
        with backend.as_null_adapter(seed=0) as null:
            null_dist = null.next_token_dist("hello")
        # Some perturbation, but bounded.
        assert not np.allclose(base_dist.logprobs, null_dist.logprobs)

    def test_different_seeds_yield_different_views(self) -> None:
        backend = _diverging_backend()
        with backend.as_null_adapter(seed=1) as v1:
            d1 = v1.next_token_dist("hello")
        with backend.as_null_adapter(seed=2) as v2:
            d2 = v2.next_token_dist("hello")
        assert not np.allclose(d1.logprobs, d2.logprobs)

    def test_view_exclusion_enforced(self) -> None:
        import pytest

        backend = _diverging_backend()
        with backend.as_null_adapter(seed=0), pytest.raises(RuntimeError):
            with backend.as_base():
                pass


class TestProbe:
    def test_populates_null_stats(self) -> None:
        """Explicit `calibrate_kinds` calibrates regardless of suite order."""
        backend = _diverging_backend()
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
        assert "delta_kl" in stats
        assert stats["delta_kl"]["n"] == 3.0
        assert stats["delta_kl"]["std"] > 0.0  # seeded perturbations produce variance

    def test_auto_populates_from_downstream_kinds(self) -> None:
        """When `calibrate_kinds` is empty, falls back to `ctx.downstream_kinds`."""
        backend = _diverging_backend()
        probe, spec = build_probe({"name": "null", "kind": "null_adapter", "runs": 2})
        ctx = RunContext(
            backend=backend,
            # paraphrase_invariance opts in (stable mean_verb under null);
            # prompt_collapse opts out (half_life is undefined under null).
            downstream_kinds=("delta_kl", "prompt_collapse", "paraphrase_invariance"),
        )
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.PASS
        stats = result.evidence["null_stats"]
        # Downstream numeric kinds that opt in get stats.
        assert "delta_kl" in stats
        assert "paraphrase_invariance" in stats
        # prompt_collapse opts out — show up in skipped_kinds instead.
        assert "prompt_collapse" not in stats
        skipped = {s["kind"] for s in result.evidence["skipped_kinds"]}
        assert "prompt_collapse" in skipped

    def test_empty_calibrate_kinds_with_no_downstream_is_noop(self) -> None:
        """No kinds, no calibration — probe still PASSes with empty stats."""
        backend = _diverging_backend()
        probe, spec = build_probe({"name": "null", "kind": "null_adapter", "runs": 2})
        ctx = RunContext(backend=backend)  # no downstream_kinds
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.PASS
        assert result.evidence["null_stats"] == {}
        assert result.evidence["calibrated_kinds"] == []

    def test_unregistered_kind_is_silently_skipped(self) -> None:
        backend = _diverging_backend()
        probe, spec = build_probe(
            {
                "name": "null",
                "kind": "null_adapter",
                "runs": 2,
                "calibrate_kinds": ["delta_kl", "nonexistent_kind"],
            }
        )
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        assert "delta_kl" in result.evidence["null_stats"]
        assert "nonexistent_kind" not in result.evidence["null_stats"]

    def test_opt_out_probe_is_reported_as_skipped(self) -> None:
        """A kind whose calibrate_spec returns None surfaces in skipped_kinds."""
        backend = _diverging_backend()
        probe, spec = build_probe(
            {
                "name": "null",
                "kind": "null_adapter",
                "runs": 2,
                # adapter_revert.calibrate_spec returns None by default
                # (inherits from base), so we expect it to opt out.
                "calibrate_kinds": ["adapter_revert", "delta_kl"],
            }
        )
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        assert "delta_kl" in result.evidence["null_stats"]
        skipped = [s["kind"] for s in result.evidence["skipped_kinds"]]
        assert "adapter_revert" in skipped

    def test_runner_threads_null_stats_to_subsequent_probes(self) -> None:
        """End-to-end: null_adapter first → delta_kl picks up z-score path."""
        backend = _diverging_backend()
        raw_spec = SwaySpec.model_validate(
            {
                "version": 1,
                "models": {"base": {"base": "b"}, "ft": {"base": "b", "adapter": "/tmp/a"}},
                "suite": [
                    {
                        "name": "null",
                        "kind": "null_adapter",
                        "runs": 3,
                    },
                    {
                        "name": "dk",
                        "kind": "delta_kl",
                        "prompts": ["p1", "p2"],
                        "assert_z_gte": -10.0,  # permissive so we pass regardless
                    },
                ],
            }
        )
        result = run_suite(raw_spec, backend)
        assert len(result.probes) == 2
        null_result = result.probes[0]
        dk_result = result.probes[1]
        assert null_result.verdict == Verdict.PASS
        # The delta_kl probe should have computed a z_score because null_stats was present.
        assert dk_result.z_score is not None, (
            "delta_kl should have z-scored against null baseline, got "
            f"evidence={dk_result.evidence}, message={dk_result.message}"
        )

    def test_runner_threaded_null_stats_are_immutable(self) -> None:
        """B21: a probe shouldn't be able to mutate the stats other probes consume."""
        from types import MappingProxyType

        backend = _diverging_backend()
        raw_spec = SwaySpec.model_validate(
            {
                "version": 1,
                "models": {"base": {"base": "b"}, "ft": {"base": "b", "adapter": "/tmp/a"}},
                "suite": [
                    {"name": "null", "kind": "null_adapter", "runs": 2},
                    {
                        "name": "dk",
                        "kind": "delta_kl",
                        "prompts": ["q1"],
                        "assert_z_gte": -100.0,
                    },
                ],
            }
        )
        from dlm_sway.probes import delta_kl as dk_mod

        captured: dict[str, object] = {}
        original_run = dk_mod.DeltaKLProbe.run

        def _capturing_run(self, spec, ctx):
            captured["null_stats"] = ctx.null_stats
            return original_run(self, spec, ctx)

        mp = pytest.MonkeyPatch()
        mp.setattr(dk_mod.DeltaKLProbe, "run", _capturing_run)
        try:
            run_suite(raw_spec, backend)
        finally:
            mp.undo()

        stats = captured["null_stats"]
        assert isinstance(stats, MappingProxyType), (
            f"expected MappingProxyType, got {type(stats).__name__}"
        )
        with pytest.raises(TypeError):
            stats["bogus"] = {"mean": 0.0, "std": 1.0, "n": 1.0}  # type: ignore[index]

    def test_cache_hit_short_circuits_calibration(self, tmp_path, monkeypatch) -> None:
        """A cached stats blob is loaded without re-running any probes."""
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

        class _IdBackend(DummyDifferentialBackend):
            def cache_identity(self) -> str:
                return "test:id-backend"

        backend = _IdBackend(base=DummyResponses(), ft=DummyResponses())

        # First call: populates the cache.
        probe, spec = build_probe(
            {
                "name": "null",
                "kind": "null_adapter",
                "runs": 2,
                "calibrate_kinds": ["delta_kl"],
            }
        )
        ctx = RunContext(backend=backend)
        r1 = probe.run(spec, ctx)
        assert r1.evidence["from_cache"] is False

        # Second call: same params, same identity → cache hit.
        r2 = probe.run(spec, ctx)
        assert r2.evidence["from_cache"] is True
        assert "delta_kl" in r2.evidence["null_stats"]

    def test_cache_disabled_forces_recompute(self, tmp_path, monkeypatch) -> None:
        """``cache=false`` bypasses the cache even if a prior run populated it."""
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

        class _IdBackend(DummyDifferentialBackend):
            def cache_identity(self) -> str:
                return "test:id-backend-2"

        backend = _IdBackend(base=DummyResponses(), ft=DummyResponses())
        probe, populating_spec = build_probe(
            {
                "name": "null",
                "kind": "null_adapter",
                "runs": 2,
                "calibrate_kinds": ["delta_kl"],
            }
        )
        probe.run(populating_spec, RunContext(backend=backend))

        _, fresh_spec = build_probe(
            {
                "name": "null",
                "kind": "null_adapter",
                "runs": 2,
                "calibrate_kinds": ["delta_kl"],
                "cache": False,
            }
        )
        r = probe.run(fresh_spec, RunContext(backend=backend))
        assert r.evidence["from_cache"] is False

    def test_degenerate_calibration_flagged_and_refused(self) -> None:
        """F02 (Audit 03): identical raws or runs≤1 → ``degenerate: 1.0``
        in the stats dict, and the downstream z-score computation
        refuses instead of firing on a 1e-6 floor.

        Pre-F02 this test asserted ``std ≥ 1e-6`` + ``z is not None``,
        which is exactly the contract that produced the audit's
        +290,766σ observation on a leakage probe under ``runs: 1``.
        The fix flips both assertions.
        """
        backend = _diverging_backend()
        probe, spec = build_probe(
            {
                "name": "null",
                "kind": "null_adapter",
                "runs": 1,  # single seed → degenerate by construction
                "calibrate_kinds": ["delta_kl"],
            }
        )
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.PASS
        stats = result.evidence["null_stats"]["delta_kl"]
        # Std floor is still 1e-6 (preserved for valid-but-tight
        # multi-seed nulls). What changed is the explicit
        # ``degenerate`` flag on the stats dict — ``runs: 1`` → True.
        assert stats["std"] == 1e-6
        assert stats["degenerate"] >= 0.5
        # Downstream z_score now refuses rather than emit runaway values.
        from dlm_sway.probes._zscore import z_score

        assert z_score(0.5, stats) is None

    def test_per_kind_stats_published(self) -> None:
        """Every calibrating kind gets its own (mean, std, n) triple."""
        backend = _diverging_backend()
        probe, spec = build_probe(
            {
                "name": "null",
                "kind": "null_adapter",
                "runs": 3,
                "calibrate_kinds": ["delta_kl", "paraphrase_invariance"],
            }
        )
        ctx = RunContext(backend=backend)
        result = probe.run(spec, ctx)
        stats = result.evidence["null_stats"]
        for kind in ("delta_kl", "paraphrase_invariance"):
            assert kind in stats, f"missing {kind} in published stats"
            s = stats[kind]
            assert "mean" in s
            assert "std" in s
            assert "n" in s
            assert s["std"] >= 1e-6

    def test_skip_when_backend_not_null_calibrated(self) -> None:
        class _Bare:
            def as_base(self):  # noqa: ANN202
                raise NotImplementedError

            def as_finetuned(self):  # noqa: ANN202
                raise NotImplementedError

        probe, spec = build_probe({"name": "null", "kind": "null_adapter"})
        ctx = RunContext(backend=_Bare())  # type: ignore[arg-type]
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.SKIP
        assert "NullCalibratedBackend" in result.message
