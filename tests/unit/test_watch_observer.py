"""Smoke tests for :func:`dlm_sway.watch.core.start_observing`.

These exercise the watchdog wiring end-to-end against the real OS
filesystem-event API. They're fast (real filesystem on tmp_path) but
unavoidably timing-sensitive — pinned to generous deadlines so noise
on heavily-loaded CI runners doesn't flake.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

# watchdog is in the [watch] extra; gate the whole module so contributors
# without it don't see import errors.
pytest.importorskip("watchdog")

from dlm_sway.watch.core import (  # noqa: E402
    RunResult,
    Watcher,
    WatcherConfig,
    start_observing,
)

SPEC_YAML = """\
version: 1
models:
  base: { kind: dummy, base: tiny-base }
  ft:   { kind: dummy, base: tiny-base, adapter: ./fake-adapter }
defaults:
  seed: 0
  differential: true
  coverage_threshold: 0.6
suite:
  - { name: smoke, kind: dir,
      prompt: "the cat",
      target: " sat",
      distractor: " ran",
      assert: { delta_logprob_gte: 0.0 } }
"""


@pytest.fixture
def store_with_adapter(tmp_path: Path) -> tuple[Path, Path]:
    store = tmp_path / "store"
    (store / "adapter" / "versions" / "v0001").mkdir(parents=True)
    pointer = store / "adapter" / "current.txt"
    pointer.write_text("adapter/versions/v0001\n", encoding="utf-8")
    return store, pointer


@pytest.fixture
def spec_path(tmp_path: Path) -> Path:
    p = tmp_path / "spec.yaml"
    p.write_text(SPEC_YAML, encoding="utf-8")
    return p


def _stub_runner(verdict: str = "pass") -> Any:
    def _run(spec_path: Path, spec_obj: Any, pointer_text: str, serve_url: str | None) -> RunResult:
        return RunResult(
            verdict=verdict,
            json_payload=json.dumps({"verdict": verdict, "score": {"overall": 0.8}}),
            run_seconds=0.01,
            overall_score=0.8,
        )

    return _run


def test_observer_fires_on_pointer_write(
    spec_path: Path,
    store_with_adapter: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """A pointer rewrite triggers exactly one suite run within 2s."""
    store, pointer = store_with_adapter
    history = tmp_path / "history"
    seen: list[Any] = []
    watcher = Watcher(
        WatcherConfig(spec_path=spec_path, history_dir=history),
        pointer_path=pointer,
        runner=_stub_runner(),
        on_outcome=seen.append,
    )
    observed = start_observing(watcher, debounce_s=0.05)
    try:
        # New adapter version → atomic rewrite of current.txt.
        (store / "adapter" / "versions" / "v0002").mkdir()
        pointer.write_text("adapter/versions/v0002\n", encoding="utf-8")
        deadline = time.monotonic() + 2.0
        while not seen and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(seen) == 1
        assert seen[0].adapter_pointer == "adapter/versions/v0002"
    finally:
        observed.stop()


def test_observer_debounces_burst_writes(
    spec_path: Path,
    store_with_adapter: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """A burst of writes inside the debounce window collapses to one fire."""
    store, pointer = store_with_adapter
    history = tmp_path / "history"
    seen: list[Any] = []
    watcher = Watcher(
        WatcherConfig(spec_path=spec_path, history_dir=history),
        pointer_path=pointer,
        runner=_stub_runner(),
        on_outcome=seen.append,
    )
    observed = start_observing(watcher, debounce_s=0.3)
    try:
        # Burst within debounce window — same target each time, so the
        # final fire sees one new pointer value.
        (store / "adapter" / "versions" / "v0009").mkdir()
        for _ in range(5):
            pointer.write_text("adapter/versions/v0009\n", encoding="utf-8")
            time.sleep(0.02)
        # Wait past the debounce window for the single fire.
        deadline = time.monotonic() + 2.0
        while not seen and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(seen) == 1
    finally:
        observed.stop()


def test_observer_stop_terminates_cleanly(
    spec_path: Path,
    store_with_adapter: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """observed.stop() returns within the timeout even with no events."""
    _, pointer = store_with_adapter
    history = tmp_path / "history"
    watcher = Watcher(
        WatcherConfig(spec_path=spec_path, history_dir=history),
        pointer_path=pointer,
        runner=_stub_runner(),
    )
    observed = start_observing(watcher, debounce_s=0.05)
    started_stop = time.monotonic()
    observed.stop()
    assert time.monotonic() - started_stop < 5.0
