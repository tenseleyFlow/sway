"""Tests for :mod:`dlm_sway.backends._instrumentation` (Sprint 07).

Covers the three invariants the cache + trace + stats plumbing must
hold:

1. **LRU correctness**: hit / miss / eviction at capacity.
2. **View-id isolation**: the same ``(prompt, top_k)`` on a different
   view_id is a miss — the Sprint 07 cache must *not* cross-pollute
   base and ft results (that would silently poison every probe).
3. **Trace writer**: ``path=None`` is a zero-overhead no-op; with a
   path, the file is JSONL-parseable and carries the expected fields.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dlm_sway.backends._instrumentation import (
    BackendInstrumentation,
    BackendStats,
    ForwardCache,
    TraceWriter,
)


class TestForwardCache:
    def test_hit_after_put(self) -> None:
        cache = ForwardCache(maxsize=4)
        key = ("next_token_dist", "base", "abc", 32)
        cache.put(key, "value")
        assert cache.get(key) == "value"

    def test_miss_returns_sentinel(self) -> None:
        from dlm_sway.backends._instrumentation import _MISS

        cache = ForwardCache(maxsize=4)
        assert cache.get(("missing",)) is _MISS

    def test_lru_eviction(self) -> None:
        cache = ForwardCache(maxsize=2)
        cache.put(("a",), 1)
        cache.put(("b",), 2)
        cache.put(("c",), 3)  # evicts "a" (LRU)
        from dlm_sway.backends._instrumentation import _MISS

        assert cache.get(("a",)) is _MISS
        assert cache.get(("b",)) == 2
        assert cache.get(("c",)) == 3

    def test_get_promotes_to_mru(self) -> None:
        """Re-accessing an entry bumps it to the front of the LRU."""
        cache = ForwardCache(maxsize=2)
        cache.put(("a",), 1)
        cache.put(("b",), 2)
        _ = cache.get(("a",))  # promote a to MRU
        cache.put(("c",), 3)  # should evict b, not a
        from dlm_sway.backends._instrumentation import _MISS

        assert cache.get(("a",)) == 1
        assert cache.get(("b",)) is _MISS

    def test_invalid_maxsize_rejected(self) -> None:
        with pytest.raises(ValueError, match="maxsize must be positive"):
            ForwardCache(maxsize=0)

    def test_clear(self) -> None:
        cache = ForwardCache(maxsize=4)
        cache.put(("a",), 1)
        cache.clear()
        assert len(cache) == 0


class TestBackendInstrumentationCached:
    def test_miss_then_hit(self) -> None:
        inst = BackendInstrumentation()
        call_count = {"n": 0}

        def compute() -> str:
            call_count["n"] += 1
            return "computed"

        v1 = inst.cached("next_token_dist", "base", "prompt", 32, compute)
        v2 = inst.cached("next_token_dist", "base", "prompt", 32, compute)
        assert v1 == v2 == "computed"
        assert call_count["n"] == 1
        assert inst.stats.cache_hits == 1
        assert inst.stats.cache_misses == 1
        assert inst.stats.forward_passes == 1

    def test_view_id_isolates_cache_entries(self) -> None:
        """Base and ft views on the same prompt must *not* collide."""
        inst = BackendInstrumentation()
        hits: list[str] = []

        def compute_base() -> str:
            hits.append("base")
            return "base_value"

        def compute_ft() -> str:
            hits.append("ft")
            return "ft_value"

        v_base = inst.cached("next_token_dist", "base", "p", 32, compute_base)
        v_ft = inst.cached("next_token_dist", "ft", "p", 32, compute_ft)
        assert v_base == "base_value"
        assert v_ft == "ft_value"
        assert hits == ["base", "ft"]  # neither side short-circuited

    def test_top_k_isolates_cache_entries(self) -> None:
        """top_k=8 vs top_k=32 are different cache entries."""
        inst = BackendInstrumentation()
        hits = 0

        def compute() -> int:
            nonlocal hits
            hits += 1
            return hits

        a = inst.cached("next_token_dist", "base", "p", 8, compute)
        b = inst.cached("next_token_dist", "base", "p", 32, compute)
        assert a != b
        assert hits == 2

    def test_op_isolates_cache_entries(self) -> None:
        """Same prompt via logprob_of vs next_token_dist → distinct keys."""
        inst = BackendInstrumentation()
        calls: list[str] = []

        def c_lp() -> float:
            calls.append("lp")
            return -3.14

        def c_dist() -> str:
            calls.append("dist")
            return "d"

        inst.cached("logprob_of", "base", "p", 0, c_lp)
        inst.cached("next_token_dist", "base", "p", 0, c_dist)
        assert calls == ["lp", "dist"]


class TestBackendStats:
    def test_hit_rate_zero_when_empty(self) -> None:
        s = BackendStats()
        assert s.hit_rate == 0.0

    def test_to_dict_shape(self) -> None:
        s = BackendStats(cache_hits=3, cache_misses=7, forward_passes=7, scoring_wall_s=1.5)
        d = s.to_dict()
        assert d["cache_hits"] == 3
        assert d["cache_misses"] == 7
        assert d["forward_passes"] == 7
        assert d["scoring_wall_s"] == pytest.approx(1.5)
        assert d["hit_rate"] == pytest.approx(0.3)


class TestTraceWriter:
    def test_disabled_is_noop(self, tmp_path: Path) -> None:
        """``path=None`` creates no file and writes nothing."""
        from dlm_sway.backends._instrumentation import _TraceEvent

        writer = TraceWriter(None)
        writer.write(
            _TraceEvent(
                ts=0.0,
                probe="p",
                view_id="base",
                prompt_hash="x",
                top_k=0,
                op="o",
                wall_ms=1.0,
                hit=True,
            )
        )
        writer.close()
        # Nothing written anywhere — no assertion needed beyond the
        # non-crash, but confirm the tmp_path is empty.
        assert not any(tmp_path.iterdir())

    def test_enabled_writes_jsonl(self, tmp_path: Path) -> None:
        from dlm_sway.backends._instrumentation import _TraceEvent

        trace_file = tmp_path / "trace.jsonl"
        writer = TraceWriter(trace_file)
        writer.write(
            _TraceEvent(
                ts=1.23,
                probe="dk",
                view_id="base",
                prompt_hash="abc",
                top_k=32,
                op="next_token_dist",
                wall_ms=0.75,
                hit=False,
            )
        )
        writer.write(
            _TraceEvent(
                ts=1.24,
                probe="dk",
                view_id="base",
                prompt_hash="abc",
                top_k=32,
                op="next_token_dist",
                wall_ms=0.01,
                hit=True,
            )
        )
        writer.close()

        lines = trace_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["probe"] == "dk"
        assert first["view_id"] == "base"
        assert first["op"] == "next_token_dist"
        assert first["hit"] is False
        assert json.loads(lines[1])["hit"] is True

    def test_instrumentation_end_to_end_trace(self, tmp_path: Path) -> None:
        """Full path: BackendInstrumentation → cached() → trace file."""
        trace_file = tmp_path / "trace.jsonl"
        inst = BackendInstrumentation()
        inst.trace = TraceWriter(trace_file)
        inst.set_current_probe("my_probe")

        inst.cached("next_token_dist", "base", "the capital", 16, lambda: "computed")
        inst.cached("next_token_dist", "base", "the capital", 16, lambda: "computed")
        inst.close()

        lines = trace_file.read_text(encoding="utf-8").strip().splitlines()
        events = [json.loads(line) for line in lines]
        assert len(events) == 2
        assert events[0]["probe"] == "my_probe"
        assert events[0]["hit"] is False
        assert events[1]["hit"] is True
