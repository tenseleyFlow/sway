"""Unit tests for :class:`dlm_sway.serve.cache.BackendCache`.

The cache is the daemon's heart: an LRU of warm differential backends
keyed by the identity tuple over ``ModelSpec``. These tests exercise:

* hit / miss / LRU eviction order
* concurrent ``get_or_load`` for the same key only loads once
* ``close()`` runs on eviction (and tolerates close failures)
* ``cache_key_for`` is stable across spec field permutations that
  don't change identity
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from dlm_sway.core.model import ModelSpec
from dlm_sway.serve.cache import BackendCache, CachedBackend, cache_key_for


class _StubBackend:
    """Minimal stand-in implementing the loader contract.

    Tracks how many times :meth:`close` was called so eviction tests
    can assert the call happened. Doesn't need to satisfy the full
    ``DifferentialBackend`` Protocol — the cache only ever calls
    ``.close()`` on it, and tests look up ``.tag`` to identify entries.
    """

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


def _spec(base: str, *, adapter: Path | None = None) -> ModelSpec:
    return ModelSpec(base=base, kind="dummy", adapter=adapter)


def _seed(cache: BackendCache, spec: ModelSpec, tag: str) -> _StubBackend:
    """Helper: insert a stub entry under ``spec``'s identity key."""
    backend = _StubBackend(tag)
    key = cache_key_for(spec)
    entry = CachedBackend(
        key=key,
        backend=backend,  # type: ignore[arg-type]
        model_spec=spec,
        load_seconds=0.0,
    )
    # Use the cache's internal lock + dict directly so we don't trip
    # the build path. Accessing _entries is fine in unit tests; the
    # production path goes through get_or_load.
    with cache._lock:  # noqa: SLF001
        cache._entries[key] = entry  # noqa: SLF001
    return backend


class TestCacheKey:
    def test_key_is_stable_across_equivalent_specs(self) -> None:
        a = _spec("modelA")
        b = _spec("modelA")
        assert cache_key_for(a) == cache_key_for(b)

    def test_key_differs_on_base(self) -> None:
        assert cache_key_for(_spec("modelA")) != cache_key_for(_spec("modelB"))

    def test_key_differs_on_adapter(self, tmp_path: Path) -> None:
        adapter = tmp_path / "ad"
        adapter.mkdir()
        with_adapter = _spec("modelA", adapter=adapter)
        without_adapter = _spec("modelA")
        assert cache_key_for(with_adapter) != cache_key_for(without_adapter)

    def test_key_ignores_trust_remote_code(self) -> None:
        """Two specs differing only in non-identity fields hash equal."""
        plain = ModelSpec(base="modelA", kind="dummy")
        trust = ModelSpec(base="modelA", kind="dummy", trust_remote_code=True)
        assert cache_key_for(plain) == cache_key_for(trust)


class TestCacheLRU:
    def test_max_size_validation(self) -> None:
        with pytest.raises(ValueError, match="max_size must be >= 1"):
            BackendCache(max_size=0)

    def test_hit_promotes_to_mru(self) -> None:
        cache = BackendCache(max_size=2)
        spec_a = _spec("A")
        spec_b = _spec("B")
        backend_a = _seed(cache, spec_a, "A")
        _seed(cache, spec_b, "B")

        # Touch A so it becomes MRU. get_or_load goes through the
        # hit path and moves the entry to the end.
        result = cache.get_or_load(spec_a)
        assert result.backend is backend_a

        keys = cache.loaded_keys()
        assert keys[-1] == cache_key_for(spec_a)
        assert keys[0] == cache_key_for(spec_b)

    def test_eviction_closes_lru_backend(self) -> None:
        """Insert 2 with cap=2, then load a 3rd via get_or_load and
        confirm the LRU's close() fires."""
        cache = BackendCache(max_size=2)
        spec_a = _spec("A")
        spec_b = _spec("B")
        backend_a = _seed(cache, spec_a, "A")
        _seed(cache, spec_b, "B")

        # Stub the loader so we don't need a real backend build.
        third_backend = _StubBackend("C")

        def _fake_build(spec: ModelSpec, *, adapter_path: Path | None) -> Any:
            del spec, adapter_path
            return third_backend

        import dlm_sway.serve.cache as cache_mod

        original = cache_mod._build_entry  # noqa: SLF001

        def _fake_build_entry(spec: ModelSpec, *, key: Any, adapter_path: Any) -> CachedBackend:
            return CachedBackend(
                key=key,
                backend=_fake_build(spec, adapter_path=adapter_path),
                model_spec=spec,
                load_seconds=0.0,
            )

        cache_mod._build_entry = _fake_build_entry  # type: ignore[assignment]
        try:
            cache.get_or_load(_spec("C"))
        finally:
            cache_mod._build_entry = original  # type: ignore[assignment]

        assert backend_a.close_count == 1, "LRU eviction should call close()"
        keys = cache.loaded_keys()
        assert cache_key_for(spec_a) not in keys
        assert cache_key_for(spec_b) in keys
        assert cache_key_for(_spec("C")) in keys

    def test_evict_all_closes_every_backend(self) -> None:
        cache = BackendCache(max_size=3)
        backends = [_seed(cache, _spec(f"M{i}"), f"M{i}") for i in range(3)]
        cache.evict_all()
        assert cache.loaded_keys() == []
        for b in backends:
            assert b.close_count == 1

    def test_close_failure_is_swallowed(self, caplog: pytest.LogCaptureFixture) -> None:
        """A backend whose close() raises should not crash the daemon."""
        cache = BackendCache(max_size=1)
        spec = _spec("boom")
        backend = _seed(cache, spec, "boom")

        def _raising_close() -> None:
            raise RuntimeError("close failed")

        backend.close = _raising_close  # type: ignore[method-assign]

        with caplog.at_level("WARNING"):
            cache.evict_all()

        # The error was logged but didn't propagate.
        assert any("close raised" in r.message for r in caplog.records)
        assert cache.loaded_keys() == []


class TestSingleFlight:
    def test_concurrent_get_or_load_loads_once(self) -> None:
        """Two threads asking for the same spec must result in exactly
        one underlying build, not two."""
        cache = BackendCache(max_size=2)

        build_count = 0
        build_lock = threading.Lock()
        backend = _StubBackend("solo")

        import dlm_sway.serve.cache as cache_mod

        original = cache_mod._build_entry  # noqa: SLF001

        def _slow_build_entry(spec: ModelSpec, *, key: Any, adapter_path: Any) -> CachedBackend:
            nonlocal build_count
            with build_lock:
                build_count += 1
            # Sleep to widen the window for both threads to see a miss.
            time.sleep(0.05)
            return CachedBackend(
                key=key,
                backend=backend,  # type: ignore[arg-type]
                model_spec=spec,
                load_seconds=0.0,
            )

        cache_mod._build_entry = _slow_build_entry  # type: ignore[assignment]

        spec = _spec("solo")
        results: list[CachedBackend] = []
        errs: list[BaseException] = []

        def _worker() -> None:
            try:
                results.append(cache.get_or_load(spec))
            except BaseException as exc:  # noqa: BLE001
                errs.append(exc)

        try:
            threads = [threading.Thread(target=_worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5.0)
        finally:
            cache_mod._build_entry = original  # type: ignore[assignment]

        assert errs == []
        assert build_count == 1, f"single-flight broken: built {build_count} times"
        assert all(r.backend is backend for r in results)
