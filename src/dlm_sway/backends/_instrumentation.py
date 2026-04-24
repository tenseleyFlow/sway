"""Shared forward-pass cache + tracing + stats (Sprint 07).

The audit observed that probes re-compute base distributions earlier
probes already produced — ``delta_kl``, ``adapter_ablation``,
``prompt_collapse``, and the null-adapter calibration matrix all hit
overlapping prompts on the base view, and nothing was sharing the
result. This module is the shared plumbing:

- :class:`ForwardCache` — a bounded LRU keyed on
  ``(op, view_id, prompt_hash, top_k)`` that backend views consult
  before computing a forward pass.
- :class:`TraceWriter` — appends one JSONL line per forward pass when
  enabled; total no-op when disabled.
- :class:`BackendInstrumentation` — what a backend instance owns (one
  cache + one tracer). Views call :meth:`cached` to route their
  scoring methods through both.

Every scoring method on every backend view is exactly three lines of
glue:

    def next_token_dist(self, prompt, *, top_k=256):
        return self._inst.cached(
            "next_token_dist", self.id, prompt, top_k,
            lambda: self._compute_next_token_dist(prompt, top_k=top_k),
        )

Thread-safety scope is intentionally narrow — the HF backend's
``_active`` toggle is still single-threaded (B19 design note); the
cache itself is protected by a lock so concurrent view entries
don't corrupt the LRU ordering even in the experimental concurrency
mode.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

#: The operations the cache keys on. Kept tight — each op has a
#: distinct return shape (TokenDist vs RollingLogprob vs float) and
#: the backend view is the one that decides which to call.
CacheOp = str  # one of "next_token_dist" | "rolling_logprob" | "logprob_of"


@dataclass(slots=True)
class BackendStats:
    """Counters a backend publishes to the suite result for a run.

    Fields are cumulative across every view transition; the runner
    snapshots them after the suite finishes and attaches the snapshot
    to :class:`~dlm_sway.core.result.SuiteResult.backend_stats`.
    """

    cache_hits: int = 0
    cache_misses: int = 0
    forward_passes: int = 0
    #: Total wall seconds spent *inside* backend scoring methods
    #: (forward-pass compute + cache hits; excludes probe-side work).
    scoring_wall_s: float = 0.0
    #: S23 batched-backend-exec counters. ``batches_sent`` is the number
    #: of batched forward calls the backend actually issued (i.e. misses
    #: routed through the batched path; single-prompt calls and cache
    #: hits don't increment). ``batched_prompts`` is the sum of batch
    #: sizes, so ``batched_prompts / batches_sent`` = mean batch size.
    #: A run with zero batches (older probes / opt-out probes only)
    #: reports the avg as 0 via :attr:`avg_batch_size`.
    batches_sent: int = 0
    batched_prompts: int = 0
    max_batch_size: int = 0

    @property
    def total_lookups(self) -> int:
        return self.cache_hits + self.cache_misses

    @property
    def hit_rate(self) -> float:
        total = self.total_lookups
        return (self.cache_hits / total) if total > 0 else 0.0

    @property
    def avg_batch_size(self) -> float:
        return (self.batched_prompts / self.batches_sent) if self.batches_sent > 0 else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "forward_passes": self.forward_passes,
            "scoring_wall_s": self.scoring_wall_s,
            "hit_rate": self.hit_rate,
            "batches_sent": self.batches_sent,
            "batched_prompts": self.batched_prompts,
            "max_batch_size": self.max_batch_size,
            "avg_batch_size": self.avg_batch_size,
        }


class ForwardCache:
    """Bounded LRU for backend forward passes, keyed on view + prompt.

    We don't use :func:`functools.lru_cache` because the cacheable
    unit isn't a function — it's a ``(op, view_id, prompt, top_k)``
    tuple that spans multiple scoring methods. An explicit
    :class:`~collections.OrderedDict` keeps insertion order and lets
    us bump entries to the end on hit.

    The key includes ``view_id`` (e.g. ``"base"``, ``"ft"``,
    ``"scaled_1.25"``, ``"null_42"``) so a view transition is a
    natural cache miss — no explicit invalidation needed. If a future
    fix to the HF toggle corrupts base/ft weights in place, a new
    view-id string would surface the drift immediately.
    """

    def __init__(self, maxsize: int = 256) -> None:
        if maxsize <= 0:
            raise ValueError(f"maxsize must be positive; got {maxsize}")
        self._maxsize = maxsize
        self._store: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple[Any, ...]) -> Any:
        with self._lock:
            try:
                value = self._store[key]
            except KeyError:
                return _MISS
            self._store.move_to_end(key)
            return value

    def put(self, key: tuple[Any, ...], value: Any) -> None:
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


#: Sentinel used to tell ``ForwardCache.get`` misses apart from legit
#: ``None`` values. Downstream callers check ``result is _MISS``.
_MISS: Any = object()


@dataclass(slots=True)
class _TraceEvent:
    ts: float
    probe: str | None
    view_id: str
    prompt_hash: str
    top_k: int
    op: str
    wall_ms: float
    hit: bool


class TraceWriter:
    """Append-only JSONL writer for forward-pass traces.

    Disabled mode (``path=None``) is a zero-overhead no-op: the
    :meth:`write` method short-circuits on the first branch and the
    backend pays one ``if`` per forward pass. Enabled mode opens the
    file once and appends per event; thread-safe via a simple lock.
    """

    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._fh: Any = None
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8")

    def write(self, event: _TraceEvent) -> None:
        if self._fh is None:
            return
        payload = {
            "ts": event.ts,
            "probe": event.probe,
            "view_id": event.view_id,
            "prompt_hash": event.prompt_hash,
            "top_k": event.top_k,
            "op": event.op,
            "wall_ms": event.wall_ms,
            "hit": event.hit,
        }
        with self._lock:
            self._fh.write(json.dumps(payload) + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class BackendInstrumentation:
    """What a backend instance owns: one cache, one tracer, one stats.

    Backend ``__init__`` constructs this; every view holds a ref.
    The runner peeks at ``.stats`` at suite-end and ships the snapshot
    in :class:`SuiteResult.backend_stats`.

    Context about ``current_probe``: a backend view has no idea which
    probe is calling it, but the runner does. The runner sets
    :meth:`set_current_probe` before each probe runs so trace events
    carry the right label; views don't care.
    """

    def __init__(
        self,
        *,
        cache_maxsize: int = 256,
        trace_path: Path | None = None,
    ) -> None:
        self.cache = ForwardCache(maxsize=cache_maxsize)
        self.trace = TraceWriter(trace_path)
        self.stats = BackendStats()
        self._current_probe: str | None = None

    def set_current_probe(self, name: str | None) -> None:
        """Called by the runner between probes. Label for trace events."""
        self._current_probe = name

    def close(self) -> None:
        self.trace.close()

    def cached(
        self,
        op: CacheOp,
        view_id: str,
        prompt: str,
        top_k: int,
        compute: Callable[[], T],
    ) -> T:
        """Route a backend scoring call through the cache + tracer.

        On cache hit, returns the cached value and increments
        ``cache_hits``. On miss, runs ``compute()``, stores the
        result, increments ``cache_misses`` + ``forward_passes``,
        and (when tracing is enabled) appends a JSONL event.
        """
        prompt_hash = _prompt_hash(prompt)
        key = (op, view_id, prompt_hash, top_k)
        hit_start = time.perf_counter()
        cached_value = self.cache.get(key)
        if cached_value is not _MISS:
            self.stats.cache_hits += 1
            wall = time.perf_counter() - hit_start
            self.stats.scoring_wall_s += wall
            self.trace.write(
                _TraceEvent(
                    ts=time.time(),
                    probe=self._current_probe,
                    view_id=view_id,
                    prompt_hash=prompt_hash,
                    top_k=top_k,
                    op=op,
                    wall_ms=wall * 1000.0,
                    hit=True,
                )
            )
            return cached_value  # type: ignore[no-any-return]

        # Miss: compute, store, record.
        compute_start = time.perf_counter()
        value = compute()
        wall = time.perf_counter() - compute_start
        self.cache.put(key, value)
        self.stats.cache_misses += 1
        self.stats.forward_passes += 1
        self.stats.scoring_wall_s += wall
        self.trace.write(
            _TraceEvent(
                ts=time.time(),
                probe=self._current_probe,
                view_id=view_id,
                prompt_hash=prompt_hash,
                top_k=top_k,
                op=op,
                wall_ms=wall * 1000.0,
                hit=False,
            )
        )
        return value

    def cached_batch(
        self,
        op: CacheOp,
        view_id: str,
        prompts: Sequence[str],
        top_k: int,
        compute_misses: Callable[[list[int]], list[Any]],
    ) -> list[Any]:
        """Route a batched scoring call through the cache + tracer.

        Per-prompt cache lookup happens first; entries already in the
        cache are served from it without ever entering the batch. The
        ``compute_misses`` callback receives the list of indices into
        ``prompts`` that missed, and is expected to return a list of
        results *in the same order* — the backend is free to pad, call
        ``model.forward`` once, and split the logits per row.

        Counters incremented:
          - ``cache_hits`` per cached prompt
          - ``cache_misses`` + ``forward_passes`` per missed prompt
          - ``batches_sent`` once per actual forward (only when
            ``compute_misses`` is called, i.e. at least one miss)
          - ``batched_prompts`` by the miss count
          - ``max_batch_size`` updated to ``max(prev, miss_count)``

        Trace events are emitted per prompt so the JSONL trace keeps
        its per-prompt granularity regardless of how many rows the
        backend packed into one GPU call.
        """
        results: list[Any] = [None] * len(prompts)
        miss_indices: list[int] = []
        prompt_hashes: list[str] = [_prompt_hash(p) for p in prompts]

        # Pass 1: cache lookups.
        for i, prompt_hash in enumerate(prompt_hashes):
            key = (op, view_id, prompt_hash, top_k)
            hit_start = time.perf_counter()
            cached_value = self.cache.get(key)
            if cached_value is not _MISS:
                self.stats.cache_hits += 1
                wall = time.perf_counter() - hit_start
                self.stats.scoring_wall_s += wall
                self.trace.write(
                    _TraceEvent(
                        ts=time.time(),
                        probe=self._current_probe,
                        view_id=view_id,
                        prompt_hash=prompt_hash,
                        top_k=top_k,
                        op=op,
                        wall_ms=wall * 1000.0,
                        hit=True,
                    )
                )
                results[i] = cached_value
            else:
                miss_indices.append(i)

        # Pass 2: one forward call for the miss subset.
        if miss_indices:
            compute_start = time.perf_counter()
            miss_values = compute_misses(miss_indices)
            if len(miss_values) != len(miss_indices):
                raise RuntimeError(
                    f"batched compute returned {len(miss_values)} values for "
                    f"{len(miss_indices)} misses — backend bug"
                )
            wall = time.perf_counter() - compute_start
            # Divide wall time evenly across misses for
            # scoring_wall_s bookkeeping; batched callers don't have a
            # per-prompt attribution.
            per_miss_wall = wall / len(miss_indices)
            self.stats.scoring_wall_s += wall
            self.stats.batches_sent += 1
            self.stats.batched_prompts += len(miss_indices)
            if len(miss_indices) > self.stats.max_batch_size:
                self.stats.max_batch_size = len(miss_indices)
            for miss_pos, idx in enumerate(miss_indices):
                value = miss_values[miss_pos]
                key = (op, view_id, prompt_hashes[idx], top_k)
                self.cache.put(key, value)
                self.stats.cache_misses += 1
                self.stats.forward_passes += 1
                self.trace.write(
                    _TraceEvent(
                        ts=time.time(),
                        probe=self._current_probe,
                        view_id=view_id,
                        prompt_hash=prompt_hashes[idx],
                        top_k=top_k,
                        op=op,
                        wall_ms=per_miss_wall * 1000.0,
                        hit=False,
                    )
                )
                results[idx] = value

        return results


def _prompt_hash(prompt: str) -> str:
    """Stable short hash for cache keys + trace logging.

    SHA-1 truncated to 12 hex chars: enough to avoid collisions at
    suite scale (thousands of distinct prompts per run), small enough
    to keep cache memory down and JSONL lines readable.
    """
    import hashlib

    return hashlib.sha1(prompt.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


__all__ = [
    "BackendInstrumentation",
    "BackendStats",
    "ForwardCache",
    "TraceWriter",
]
