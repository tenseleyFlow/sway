"""Tests for the on-disk null-calibration cache."""

from __future__ import annotations

import pytest

from dlm_sway.probes._null_cache import compute_key, load, save


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Redirect the cache root into a per-test tmp dir."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    return tmp_path


class TestComputeKey:
    def test_none_identity_returns_none(self) -> None:
        assert compute_key(backend_identity=None, params={"runs": 3}) is None

    def test_empty_identity_returns_none(self) -> None:
        assert compute_key(backend_identity="", params={"runs": 3}) is None

    def test_stable_across_calls(self) -> None:
        k1 = compute_key(backend_identity="hf:foo:/tmp/a", params={"runs": 3})
        k2 = compute_key(backend_identity="hf:foo:/tmp/a", params={"runs": 3})
        assert k1 == k2

    def test_changes_when_params_change(self) -> None:
        k1 = compute_key(backend_identity="hf:foo:/tmp/a", params={"runs": 3})
        k2 = compute_key(backend_identity="hf:foo:/tmp/a", params={"runs": 5})
        assert k1 != k2

    def test_changes_when_identity_changes(self) -> None:
        k1 = compute_key(backend_identity="hf:foo:/tmp/a", params={"runs": 3})
        k2 = compute_key(backend_identity="hf:foo:/tmp/b", params={"runs": 3})
        assert k1 != k2


class TestLoadSave:
    def test_save_then_load_roundtrip(self, isolated_cache) -> None:
        stats = {"null_stats": {"delta_kl": {"mean": 0.01, "std": 0.002, "n": 3}}}
        key = compute_key(backend_identity="hf:foo:/tmp/a", params={"runs": 3})
        assert key is not None
        save(key, stats)
        loaded = load(key)
        assert loaded == stats

    def test_load_miss_returns_none(self, isolated_cache) -> None:
        key = compute_key(backend_identity="hf:foo:/tmp/a", params={"runs": 3})
        assert load(key) is None

    def test_none_key_roundtrip_noop(self, isolated_cache) -> None:
        save(None, {"null_stats": {}})
        assert load(None) is None

    def test_malformed_json_is_treated_as_miss(self, isolated_cache, tmp_path) -> None:
        key = compute_key(backend_identity="hf:foo:/tmp/a", params={"runs": 3})
        assert key is not None
        # Manually write malformed content at the expected path.
        cache_root = tmp_path / "dlm-sway" / "null-stats"
        cache_root.mkdir(parents=True)
        (cache_root / f"{key}.json").write_text("{ not json")
        assert load(key) is None

    def test_env_disable_bypasses_both(self, isolated_cache, monkeypatch) -> None:
        monkeypatch.setenv("SWAY_DISABLE_NULL_CACHE", "1")
        key = compute_key(backend_identity="hf:foo:/tmp/a", params={"runs": 3})
        save(key, {"null_stats": {"delta_kl": {"mean": 0.01, "std": 0.002, "n": 3}}})
        assert load(key) is None
