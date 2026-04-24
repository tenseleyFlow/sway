"""S23 — batched backend execution regression tests.

Pin the three invariants the sprint depends on:

1. A ``batch_score=True`` probe routes its scoring through
   ``next_token_dist_batch`` (not the single-prompt path), and the
   instrumentation counters reflect that.
2. The dummy backend's batched path produces results identical to the
   single-prompt path — protocol default-loop correctness.
3. The report footer surfaces the batch counters alongside cache stats
   when any batched forward fires.
"""

from __future__ import annotations

import numpy as np

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.scoring import TokenDist
from dlm_sway.probes.base import RunContext, build_probe
from dlm_sway.probes.delta_kl import DeltaKLProbe
from dlm_sway.suite.report import _cache_line


def _planted_backend() -> DummyDifferentialBackend:
    """Two prompts with distinguishable base vs ft distributions."""
    base = DummyResponses(
        token_dists={
            "q1": TokenDist(
                token_ids=np.array([1, 2, 3], dtype=np.int64),
                logprobs=np.log(np.array([0.9, 0.05, 0.05], dtype=np.float32)),
                vocab_size=100,
            ),
            "q2": TokenDist(
                token_ids=np.array([5, 6], dtype=np.int64),
                logprobs=np.log(np.array([0.8, 0.2], dtype=np.float32)),
                vocab_size=100,
            ),
        }
    )
    ft = DummyResponses(
        token_dists={
            "q1": TokenDist(
                token_ids=np.array([1, 2, 3], dtype=np.int64),
                logprobs=np.log(np.array([0.3, 0.35, 0.35], dtype=np.float32)),
                vocab_size=100,
            ),
            "q2": TokenDist(
                token_ids=np.array([5, 6], dtype=np.int64),
                logprobs=np.log(np.array([0.4, 0.6], dtype=np.float32)),
                vocab_size=100,
            ),
        }
    )
    return DummyDifferentialBackend(base=base, ft=ft)


def test_delta_kl_opt_in_flag_is_set() -> None:
    """Guard against a future refactor accidentally unsetting the flag."""
    assert DeltaKLProbe.batch_score is True


def test_batched_probe_routes_through_next_token_dist_batch() -> None:
    """Running a batch_score=True probe must call the batched method
    on the view — not fall back to the per-prompt path.

    The dummy backend has no real forward to amortize, so we spy on
    the batched method directly rather than assert on
    ``batches_sent`` counters (those fire only when HF's real
    batched compute hits ``cached_batch``)."""
    backend = _planted_backend()
    calls: list[tuple[str, tuple[str, ...]]] = []

    original = backend.__class__.as_base

    from contextlib import contextmanager

    @contextmanager
    def tracking_as_base(self):  # type: ignore[no-untyped-def]
        with original(self) as view:
            orig_batch = view.next_token_dist_batch

            def tracked(prompts, **kwargs):  # type: ignore[no-untyped-def]
                calls.append(("base", tuple(prompts)))
                return orig_batch(prompts, **kwargs)

            view.next_token_dist_batch = tracked  # type: ignore[method-assign]
            yield view

    backend.__class__.as_base = tracking_as_base  # type: ignore[method-assign]
    try:
        probe, spec = build_probe(
            {
                "name": "dk",
                "kind": "delta_kl",
                "prompts": ["q1", "q2"],
                "assert_mean_gte": 0.01,
            }
        )
        ctx = RunContext(backend=backend, seed=0, top_k=256)
        probe.run(spec, ctx)
    finally:
        backend.__class__.as_base = original  # type: ignore[method-assign]

    assert calls == [("base", ("q1", "q2"))], (
        f"expected one batched base call covering both prompts, got {calls!r}"
    )


def test_batched_results_equal_serial_results() -> None:
    """Dummy default-loop: batched path is serial internally so the
    divergences must match a hand-computed single-prompt iteration."""
    backend = _planted_backend()
    with backend.as_base() as base_view:
        batched = base_view.next_token_dist_batch(["q1", "q2"], top_k=10)
        # Note: same view call twice so the cache hits on the second pass
        # — but the TokenDists returned must be byte-identical.
        serial_q1 = base_view.next_token_dist("q1", top_k=10)
        serial_q2 = base_view.next_token_dist("q2", top_k=10)
    np.testing.assert_array_equal(batched[0].token_ids, serial_q1.token_ids)
    np.testing.assert_array_equal(batched[0].logprobs, serial_q1.logprobs)
    np.testing.assert_array_equal(batched[1].token_ids, serial_q2.token_ids)
    np.testing.assert_array_equal(batched[1].logprobs, serial_q2.logprobs)


def test_report_footer_surfaces_batches_when_nonzero() -> None:
    """The cache_line footer includes the batches segment iff
    batches_sent > 0. Runs without batching show cache line alone."""
    from datetime import UTC, datetime

    from dlm_sway.core.result import SuiteResult

    now = datetime.now(tz=UTC)

    def _suite(stats: dict[str, float | int]) -> SuiteResult:
        return SuiteResult(
            spec_path="x.yaml",
            started_at=now,
            finished_at=now,
            base_model_id="stub",
            adapter_id="stub",
            sway_version="0.1.0",
            backend_stats=stats,
        )

    # With batching.
    line = _cache_line(
        _suite(
            {
                "cache_hits": 5,
                "cache_misses": 10,
                "batches_sent": 3,
                "batched_prompts": 18,
                "avg_batch_size": 6.0,
                "max_batch_size": 8,
            }
        )
    )
    assert line is not None
    assert "cache: 5/15" in line
    assert "batches: 3" in line
    assert "avg=6.0" in line

    # Without batching — pre-S23 footer shape preserved.
    line_no_batch = _cache_line(_suite({"cache_hits": 5, "cache_misses": 10, "batches_sent": 0}))
    assert line_no_batch is not None
    assert "batches" not in line_no_batch


def test_empty_prompts_short_circuit() -> None:
    """Empty prompt list on the batched path returns an empty list
    without any forward work."""
    backend = _planted_backend()
    with backend.as_base() as base_view:
        out = base_view.next_token_dist_batch([], top_k=10)
    assert out == []
