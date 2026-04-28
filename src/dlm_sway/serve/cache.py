"""LRU cache for warm differential backends.

The point of the daemon is keeping backends loaded across requests.
This module owns the cache: keyed by an immutable identity tuple over
the ModelSpec fields that determine backend identity, capped at a
configurable size, evicts least-recently-used on overflow with proper
``close()`` so weights actually get freed.

The cache is **process-local** — there's no on-disk component. Restart
the daemon and the cache resets cold. That's intentional: warm-backend
caching is a memory tradeoff, and persisting weights to disk would
duplicate what HuggingFace's own cache already does at the file level.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path

from dlm_sway.core.errors import SwayError
from dlm_sway.core.model import ModelSpec
from dlm_sway.core.scoring import DifferentialBackend

_LOG = logging.getLogger(__name__)


def cache_key_for(spec: ModelSpec) -> tuple[Hashable, ...]:
    """Identity tuple for a ModelSpec.

    Two ModelSpecs that differ only in fields that don't affect the
    loaded backend (e.g. ``trust_remote_code`` on the same already-
    cached model) hash to the same key. We're conservative — any field
    that touches model identity (``base``, ``adapter``, ``dtype``,
    ``device``, ``kind``) goes into the key. Path normalization
    happens upstream in ModelSpec's validator.
    """
    return (
        spec.kind,
        spec.base,
        str(spec.adapter) if spec.adapter is not None else None,
        spec.dtype,
        spec.device,
    )


@dataclass(frozen=True, slots=True)
class CachedBackend:
    """One entry in the cache. Frozen — fields don't mutate after load.

    ``key`` is the identity tuple; ``backend`` is the live object;
    ``model_spec`` is kept for introspection (``GET /health`` lists
    the loaded models so users can see what's warm).
    """

    key: tuple[Hashable, ...]
    backend: DifferentialBackend
    model_spec: ModelSpec
    load_seconds: float


class BackendCache:
    """LRU cache of differential backends.

    Thread-safe via a single internal lock. The cache contract:

    1. ``get_or_load(spec)`` returns a backend; on miss, builds it
       (paying the load cost) and admits to the cache.
    2. On overflow (``len > max_size``), evict LRU. Eviction calls
       ``backend.close()`` if the backend implements it; otherwise
       drops the reference and lets GC handle it.
    3. ``get_or_load`` is single-flight per key — concurrent requests
       for the same model wait on the loader thread instead of
       building the backend twice.
    """

    def __init__(self, max_size: int = 2) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1; got {max_size}")
        self._max = int(max_size)
        self._entries: OrderedDict[tuple[Hashable, ...], CachedBackend] = OrderedDict()
        self._lock = threading.RLock()
        # Per-key load locks so concurrent requests for the same model
        # serialize at the loader instead of building twice.
        self._key_locks: dict[tuple[Hashable, ...], threading.Lock] = {}

    @property
    def max_size(self) -> int:
        return self._max

    def loaded_keys(self) -> list[tuple[Hashable, ...]]:
        """Snapshot of currently-cached keys, MRU last. Used by /health."""
        with self._lock:
            return list(self._entries.keys())

    def loaded_specs(self) -> list[ModelSpec]:
        """Snapshot of currently-cached model specs, MRU last."""
        with self._lock:
            return [entry.model_spec for entry in self._entries.values()]

    def get_or_load(self, spec: ModelSpec, *, adapter_path: Path | None = None) -> CachedBackend:
        """Return a cached backend for ``spec`` or build + admit one.

        ``adapter_path`` overrides ``spec.adapter`` for the build call —
        mirrors the upstream :func:`dlm_sway.backends.build` contract
        so callers handing in a separately-resolved adapter (e.g. via
        the dlm bridge) don't have to construct a copy of the spec.
        Cache key uses ``spec.adapter``, NOT the override; if you want
        a different adapter to cache distinctly, pass a spec that
        encodes it.
        """
        key = cache_key_for(spec)

        # Fast path — spec already cached.
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                # Touch LRU position.
                self._entries.move_to_end(key)
                return entry
            key_lock = self._key_locks.setdefault(key, threading.Lock())

        # Slow path — single-flight load.
        with key_lock:
            with self._lock:
                # Recheck after acquiring the load lock — another thread
                # may have completed the load while we waited.
                entry = self._entries.get(key)
                if entry is not None:
                    self._entries.move_to_end(key)
                    return entry

            entry = _build_entry(spec, key=key, adapter_path=adapter_path)

            with self._lock:
                # Evict to fit before admitting; ensures we never spike
                # over max_size + 1 between admission and eviction.
                while len(self._entries) >= self._max:
                    self._evict_lru_locked()
                self._entries[key] = entry
                self._entries.move_to_end(key)
            return entry

    def evict_all(self) -> None:
        """Close every backend. Called on daemon shutdown."""
        with self._lock:
            keys = list(self._entries.keys())
            for k in keys:
                self._evict_locked(k)

    # -- internals -----------------------------------------------------

    def _evict_lru_locked(self) -> None:
        # Caller holds self._lock. ``OrderedDict.__iter__`` yields
        # insertion order; LRU is the first key.
        if not self._entries:
            return
        lru_key = next(iter(self._entries))
        self._evict_locked(lru_key)

    def _evict_locked(self, key: tuple[Hashable, ...]) -> None:
        # Caller holds self._lock.
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        # Backends carry a ``close()`` when they own GPU memory or
        # network connections (HF, MLX, API). Dummy doesn't —
        # so don't require it. Failure during close is logged and
        # swallowed: a daemon stays up even if one backend's close()
        # raises.
        close = getattr(entry.backend, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("backend close raised on eviction: %s", exc)
        self._key_locks.pop(key, None)


def _build_entry(
    spec: ModelSpec,
    *,
    key: tuple[Hashable, ...],
    adapter_path: Path | None,
) -> CachedBackend:
    """Materialize a backend from a spec, timing the load."""
    import time

    from dlm_sway.backends import build as build_backend

    started = time.monotonic()
    try:
        backend = build_backend(spec, adapter_path=adapter_path)
    except SwayError:
        raise
    except Exception as exc:  # noqa: BLE001 — surface load failures as SwayError
        raise SwayError(
            f"backend load failed for kind={spec.kind} base={spec.base!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    elapsed = time.monotonic() - started
    return CachedBackend(key=key, backend=backend, model_spec=spec, load_seconds=elapsed)
