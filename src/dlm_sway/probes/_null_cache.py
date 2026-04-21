"""On-disk cache for null-adapter calibration stats.

Null calibration runs a miniature version of every downstream numeric
probe across N seeds before the suite proper. For a 10-probe suite at
``runs=3`` that's ~120 forward passes; on an HF backend against a real
model this can dominate wall time. Results are deterministic in the
calibration inputs — so we cache them at
``~/.dlm-sway/null-stats/<key>.json`` keyed by the tuple that actually
influences the output.

Scope here is intentionally minimal. Sprint 07 adds a shared
forward-pass cache that cuts into a lower level; this module only
amortizes the per-suite calibration pass.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

#: Environment knob — set to ``"1"`` to bypass load + save (development
#: / CI tests that want to prove calibration actually runs).
_ENV_DISABLE = "SWAY_DISABLE_NULL_CACHE"


def _cache_root() -> Path:
    """Root directory for cached null stats. Honors ``$XDG_CACHE_HOME``
    when set; otherwise falls back to ``~/.dlm-sway/null-stats``."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "dlm-sway" / "null-stats"
    return Path.home() / ".dlm-sway" / "null-stats"


def compute_key(*, backend_identity: str | None, params: dict[str, Any]) -> str | None:
    """Hash backend identity + calibration params into a stable filename.

    Returns ``None`` when ``backend_identity`` is ``None`` — backends that
    can't uniquely identify themselves (e.g., the dummy backend used in
    tests) skip caching entirely.
    """
    if not backend_identity:
        return None
    payload = {
        "backend": backend_identity,
        "params": params,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def load(key: str | None) -> dict[str, Any] | None:
    """Return the cached null-stats dict for ``key``, or ``None`` on miss.

    Malformed / unreadable cache files are treated as a miss — we'd
    rather recompute than crash the suite. A stale / schema-mismatched
    cache can be wiped with ``rm -rf ~/.dlm-sway/null-stats``.
    """
    if key is None or os.environ.get(_ENV_DISABLE) == "1":
        return None
    path = _cache_root() / f"{key}.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def save(key: str | None, stats: dict[str, Any]) -> None:
    """Persist ``stats`` under ``key``. Silently no-ops on I/O errors —
    the cache is a speed-up, not a correctness contract."""
    if key is None or os.environ.get(_ENV_DISABLE) == "1":
        return
    root = _cache_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{key}.json"
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, sort_keys=True)
        tmp.replace(path)
    except OSError:
        return
