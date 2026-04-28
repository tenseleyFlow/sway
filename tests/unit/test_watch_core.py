"""Unit tests for :mod:`dlm_sway.watch.core`.

Tests cover the orchestrator's contract independent of watchdog:
pointer dedupe, half-write retry, spec-mtime reload, history rotation,
on-fail spawning, and the runner-error path. The watchdog wiring has
its own end-to-end test in ``test_watch_observer.py``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from dlm_sway.core.errors import SwayError
from dlm_sway.watch.core import (
    RunResult,
    Watcher,
    WatcherConfig,
    resolve_pointer_path,
)

# Fixtures -----------------------------------------------------------------


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
def store_with_pointer(tmp_path: Path) -> tuple[Path, Path]:
    """A fake dlm store with ``adapter/current.txt`` pointing somewhere."""
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


@pytest.fixture
def history_dir(tmp_path: Path) -> Path:
    return tmp_path / "history"


def _stub_runner(verdict: str = "pass", *, score: float = 0.85) -> Any:
    """Build a runner that returns a deterministic RunResult."""
    calls: list[dict[str, Any]] = []

    def _run(spec_path: Path, spec_obj: Any, pointer_text: str, serve_url: str | None) -> RunResult:
        calls.append(
            {
                "spec_path": spec_path,
                "spec_obj_id": id(spec_obj),
                "pointer_text": pointer_text,
                "serve_url": serve_url,
            }
        )
        return RunResult(
            verdict=verdict,
            json_payload=json.dumps({"verdict": verdict, "score": {"overall": score}}),
            run_seconds=0.05,
            overall_score=score,
        )

    _run.calls = calls  # type: ignore[attr-defined]
    return _run


# Pointer resolution -------------------------------------------------------


def test_resolve_pointer_prefers_current_txt(store_with_pointer: tuple[Path, Path]) -> None:
    store, pointer = store_with_pointer
    assert resolve_pointer_path(store) == pointer


def test_resolve_pointer_falls_back_to_legacy_symlink(tmp_path: Path) -> None:
    store = tmp_path / "store"
    versions = store / "adapter" / "versions" / "v0001"
    versions.mkdir(parents=True)
    legacy = store / "adapter" / "latest"
    legacy.symlink_to(versions)
    assert resolve_pointer_path(store) == legacy


def test_resolve_pointer_returns_none_when_missing(tmp_path: Path) -> None:
    store = tmp_path / "store"
    (store / "adapter").mkdir(parents=True)
    assert resolve_pointer_path(store) is None


# Watcher init -------------------------------------------------------------


def test_watcher_rejects_missing_spec(tmp_path: Path) -> None:
    config = WatcherConfig(
        spec_path=tmp_path / "ghost.yaml",
        history_dir=tmp_path / "history",
    )
    with pytest.raises(SwayError, match="spec not found"):
        Watcher(config, pointer_path=tmp_path / "ptr", runner=_stub_runner())


def test_watcher_rejects_negative_max_history(spec_path: Path, tmp_path: Path) -> None:
    config = WatcherConfig(spec_path=spec_path, history_dir=tmp_path / "history", max_history=-1)
    with pytest.raises(SwayError, match="max_history"):
        Watcher(config, pointer_path=tmp_path / "ptr", runner=_stub_runner())


def test_watcher_rejects_zero_retry_attempts(spec_path: Path, tmp_path: Path) -> None:
    config = WatcherConfig(spec_path=spec_path, history_dir=tmp_path / "history", retry_attempts=0)
    with pytest.raises(SwayError, match="retry_attempts"):
        Watcher(config, pointer_path=tmp_path / "ptr", runner=_stub_runner())


# Trigger semantics --------------------------------------------------------


def test_trigger_once_writes_history_and_returns_outcome(
    spec_path: Path,
    store_with_pointer: tuple[Path, Path],
    history_dir: Path,
) -> None:
    _, pointer = store_with_pointer
    runner = _stub_runner(verdict="pass", score=0.91)
    watcher = Watcher(
        WatcherConfig(spec_path=spec_path, history_dir=history_dir),
        pointer_path=pointer,
        runner=runner,
    )
    outcome = watcher.trigger_once()
    assert outcome is not None
    assert outcome.sequence == 1
    assert outcome.verdict == "pass"
    assert outcome.overall_score == 0.91
    assert outcome.adapter_pointer == "adapter/versions/v0001"
    assert outcome.result_path.exists()
    payload = json.loads(outcome.result_path.read_text())
    assert payload["sway_watch"]["sequence"] == 1
    assert payload["sway_watch"]["verdict"] == "pass"
    assert payload["sway_watch"]["adapter_pointer"] == "adapter/versions/v0001"


def test_trigger_once_dedupes_by_pointer(
    spec_path: Path,
    store_with_pointer: tuple[Path, Path],
    history_dir: Path,
) -> None:
    _, pointer = store_with_pointer
    runner = _stub_runner()
    watcher = Watcher(
        WatcherConfig(spec_path=spec_path, history_dir=history_dir),
        pointer_path=pointer,
        runner=runner,
    )
    first = watcher.trigger_once()
    second = watcher.trigger_once()
    assert first is not None
    assert second is None
    assert len(runner.calls) == 1


def test_trigger_once_force_runs_even_when_unchanged(
    spec_path: Path,
    store_with_pointer: tuple[Path, Path],
    history_dir: Path,
) -> None:
    _, pointer = store_with_pointer
    runner = _stub_runner()
    watcher = Watcher(
        WatcherConfig(spec_path=spec_path, history_dir=history_dir),
        pointer_path=pointer,
        runner=runner,
    )
    watcher.trigger_once()
    second = watcher.trigger_once(force=True)
    assert second is not None
    assert second.sequence == 2
    assert len(runner.calls) == 2


def test_pointer_change_triggers_new_run(
    spec_path: Path,
    store_with_pointer: tuple[Path, Path],
    history_dir: Path,
) -> None:
    store, pointer = store_with_pointer
    runner = _stub_runner()
    watcher = Watcher(
        WatcherConfig(spec_path=spec_path, history_dir=history_dir),
        pointer_path=pointer,
        runner=runner,
    )
    watcher.trigger_once()
    (store / "adapter" / "versions" / "v0002").mkdir()
    pointer.write_text("adapter/versions/v0002\n", encoding="utf-8")
    second = watcher.trigger_once()
    assert second is not None
    assert second.adapter_pointer == "adapter/versions/v0002"
    assert second.sequence == 2


# Spec mtime reload --------------------------------------------------------


def test_spec_mtime_change_marks_outcome_reloaded(
    spec_path: Path,
    store_with_pointer: tuple[Path, Path],
    history_dir: Path,
) -> None:
    _, pointer = store_with_pointer
    runner = _stub_runner()
    watcher = Watcher(
        WatcherConfig(spec_path=spec_path, history_dir=history_dir),
        pointer_path=pointer,
        runner=runner,
    )
    first = watcher.trigger_once()
    assert first is not None
    assert first.spec_reloaded is False  # initial load doesn't count

    # Bump mtime by 2s so coarse-grained filesystems still see a change.
    future = time.time() + 2.0
    os.utime(spec_path, (future, future))
    second = watcher.trigger_once(force=True)
    assert second is not None
    assert second.spec_reloaded is True


# Half-write retry ---------------------------------------------------------


def test_half_write_retry_recovers(
    spec_path: Path,
    tmp_path: Path,
    history_dir: Path,
) -> None:
    pointer = tmp_path / "current.txt"
    pointer.write_text("", encoding="utf-8")  # simulate half-write

    # Schedule the recovery write *during* the retry window.
    def _fix_after_two_reads() -> None:
        # Crude: rely on test-side timing — retry sleeps 0.05s, so 0.07s
        # gives one failed read.
        time.sleep(0.07)
        pointer.write_text("adapter/versions/v0001\n", encoding="utf-8")

    import threading

    threading.Thread(target=_fix_after_two_reads, daemon=True).start()
    runner = _stub_runner()
    watcher = Watcher(
        WatcherConfig(
            spec_path=spec_path,
            history_dir=history_dir,
            retry_attempts=5,
            retry_backoff_s=0.05,
        ),
        pointer_path=pointer,
        runner=runner,
    )
    outcome = watcher.trigger_once()
    assert outcome is not None
    assert outcome.adapter_pointer == "adapter/versions/v0001"


def test_half_write_retry_gives_up_eventually(
    spec_path: Path,
    tmp_path: Path,
    history_dir: Path,
) -> None:
    pointer = tmp_path / "current.txt"
    pointer.write_text("", encoding="utf-8")
    runner = _stub_runner()
    watcher = Watcher(
        WatcherConfig(
            spec_path=spec_path,
            history_dir=history_dir,
            retry_attempts=2,
            retry_backoff_s=0.01,
        ),
        pointer_path=pointer,
        runner=runner,
    )
    with pytest.raises(SwayError, match="failed to read adapter pointer"):
        watcher.trigger_once()
    # Runner never invoked because we never resolved the pointer.
    assert len(runner.calls) == 0


def test_missing_pointer_raises(spec_path: Path, tmp_path: Path, history_dir: Path) -> None:
    pointer = tmp_path / "ghost.txt"  # does not exist
    runner = _stub_runner()
    watcher = Watcher(
        WatcherConfig(
            spec_path=spec_path,
            history_dir=history_dir,
            retry_attempts=2,
            retry_backoff_s=0.01,
        ),
        pointer_path=pointer,
        runner=runner,
    )
    with pytest.raises(SwayError, match="adapter pointer missing"):
        watcher.trigger_once()


# Runner errors → verdict=error -------------------------------------------


def test_runner_exception_becomes_error_verdict(
    spec_path: Path,
    store_with_pointer: tuple[Path, Path],
    history_dir: Path,
) -> None:
    _, pointer = store_with_pointer

    def _raising_runner(*args: Any, **kwargs: Any) -> RunResult:
        raise RuntimeError("backend went sideways")

    watcher = Watcher(
        WatcherConfig(spec_path=spec_path, history_dir=history_dir),
        pointer_path=pointer,
        runner=_raising_runner,
    )
    outcome = watcher.trigger_once()
    assert outcome is not None
    assert outcome.verdict == "error"
    assert outcome.result_path.exists()
    payload = json.loads(outcome.result_path.read_text())
    assert payload.get("error")
    assert "RuntimeError" in payload["error"]
    assert payload["sway_watch"]["verdict"] == "error"
    # Critical: errored-pointer must NOT be marked as last_pointer, so
    # the next trigger fires again rather than dedup-skipping.
    assert watcher.last_pointer is None


def test_error_does_not_set_last_pointer(
    spec_path: Path,
    store_with_pointer: tuple[Path, Path],
    history_dir: Path,
) -> None:
    _, pointer = store_with_pointer
    counter = {"calls": 0}

    def _flaky_runner(*args: Any, **kwargs: Any) -> RunResult:
        counter["calls"] += 1
        if counter["calls"] == 1:
            raise RuntimeError("first one fails")
        return RunResult(
            verdict="pass",
            json_payload="{}",
            run_seconds=0.1,
            overall_score=0.9,
        )

    watcher = Watcher(
        WatcherConfig(spec_path=spec_path, history_dir=history_dir),
        pointer_path=pointer,
        runner=_flaky_runner,
    )
    first = watcher.trigger_once()
    assert first is not None
    assert first.verdict == "error"
    second = watcher.trigger_once()
    assert second is not None
    assert second.verdict == "pass"
    assert counter["calls"] == 2


# History rotation ---------------------------------------------------------


def test_max_history_prunes_oldest(
    spec_path: Path,
    store_with_pointer: tuple[Path, Path],
    history_dir: Path,
) -> None:
    store, pointer = store_with_pointer
    runner = _stub_runner()
    watcher = Watcher(
        WatcherConfig(spec_path=spec_path, history_dir=history_dir, max_history=3),
        pointer_path=pointer,
        runner=runner,
    )
    for n in range(5):
        version = store / "adapter" / "versions" / f"v{n:04d}"
        version.mkdir(exist_ok=True)
        pointer.write_text(f"adapter/versions/v{n:04d}\n", encoding="utf-8")
        watcher.trigger_once()
        # Force distinct mtimes so sort-by-mtime is stable across noise.
        time.sleep(0.01)
    files = sorted(history_dir.glob("*.result.json"))
    assert len(files) == 3


def test_max_history_zero_retains_everything(
    spec_path: Path,
    store_with_pointer: tuple[Path, Path],
    history_dir: Path,
) -> None:
    store, pointer = store_with_pointer
    runner = _stub_runner()
    watcher = Watcher(
        WatcherConfig(spec_path=spec_path, history_dir=history_dir, max_history=0),
        pointer_path=pointer,
        runner=runner,
    )
    for n in range(4):
        version = store / "adapter" / "versions" / f"v{n:04d}"
        version.mkdir(exist_ok=True)
        pointer.write_text(f"adapter/versions/v{n:04d}\n", encoding="utf-8")
        watcher.trigger_once()
        time.sleep(0.01)
    assert len(list(history_dir.glob("*.result.json"))) == 4


# on_fail spawning ---------------------------------------------------------


def test_on_fail_spawns_only_on_fail_verdict(
    spec_path: Path,
    store_with_pointer: tuple[Path, Path],
    history_dir: Path,
    tmp_path: Path,
) -> None:
    _, pointer = store_with_pointer
    sentinel = tmp_path / "fired"
    cmd = f"echo hi > {sentinel}"
    runner = _stub_runner(verdict="pass")
    watcher = Watcher(
        WatcherConfig(spec_path=spec_path, history_dir=history_dir, on_fail_cmd=cmd),
        pointer_path=pointer,
        runner=runner,
    )
    watcher.trigger_once()
    # pass shouldn't fire the hook.
    time.sleep(0.2)
    assert not sentinel.exists()


def test_on_fail_spawns_with_result_path(
    spec_path: Path,
    store_with_pointer: tuple[Path, Path],
    history_dir: Path,
    tmp_path: Path,
) -> None:
    _, pointer = store_with_pointer
    sentinel = tmp_path / "captured.txt"
    # The shell sees SWAY_RESULT_PATH; write it out so the test can read it.
    cmd = f'sh -c "echo $SWAY_RESULT_PATH > {sentinel}"'
    runner = _stub_runner(verdict="fail")
    watcher = Watcher(
        WatcherConfig(spec_path=spec_path, history_dir=history_dir, on_fail_cmd=cmd),
        pointer_path=pointer,
        runner=runner,
    )
    outcome = watcher.trigger_once()
    assert outcome is not None
    # Wait briefly for the spawned subprocess to land.
    deadline = time.monotonic() + 2.0
    while not sentinel.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert sentinel.exists(), "on-fail command did not fire"
    written = sentinel.read_text(encoding="utf-8").strip()
    assert written == str(outcome.result_path)


def test_on_fail_unset_does_not_spawn_anything(
    spec_path: Path,
    store_with_pointer: tuple[Path, Path],
    history_dir: Path,
) -> None:
    _, pointer = store_with_pointer
    runner = _stub_runner(verdict="fail")
    watcher = Watcher(
        WatcherConfig(spec_path=spec_path, history_dir=history_dir),
        pointer_path=pointer,
        runner=runner,
    )
    # No on_fail_cmd, no crash.
    outcome = watcher.trigger_once()
    assert outcome is not None
    assert outcome.verdict == "fail"


# Hooks --------------------------------------------------------------------


def test_on_outcome_hook_fires(
    spec_path: Path,
    store_with_pointer: tuple[Path, Path],
    history_dir: Path,
) -> None:
    _, pointer = store_with_pointer
    seen: list[Any] = []
    watcher = Watcher(
        WatcherConfig(spec_path=spec_path, history_dir=history_dir),
        pointer_path=pointer,
        runner=_stub_runner(),
        on_outcome=seen.append,
    )
    watcher.trigger_once()
    assert len(seen) == 1
    assert seen[0].verdict == "pass"


def test_on_error_hook_fires(
    spec_path: Path,
    tmp_path: Path,
    history_dir: Path,
) -> None:
    pointer = tmp_path / "missing.txt"
    seen: list[BaseException] = []
    watcher = Watcher(
        WatcherConfig(
            spec_path=spec_path,
            history_dir=history_dir,
            retry_attempts=1,
            retry_backoff_s=0.01,
        ),
        pointer_path=pointer,
        runner=_stub_runner(),
        on_error=seen.append,
    )
    with pytest.raises(SwayError):
        watcher.trigger_once()
    assert len(seen) == 1


# JSON shape ---------------------------------------------------------------


def test_history_json_preserves_runner_payload(
    spec_path: Path,
    store_with_pointer: tuple[Path, Path],
    history_dir: Path,
) -> None:
    _, pointer = store_with_pointer

    def _runner(*args: Any, **kwargs: Any) -> RunResult:
        return RunResult(
            verdict="pass",
            json_payload=json.dumps(
                {"verdict": "pass", "extra": "preserved", "score": {"overall": 0.7}}
            ),
            run_seconds=0.5,
            overall_score=0.7,
        )

    watcher = Watcher(
        WatcherConfig(spec_path=spec_path, history_dir=history_dir),
        pointer_path=pointer,
        runner=_runner,
    )
    outcome = watcher.trigger_once()
    assert outcome is not None
    payload = json.loads(outcome.result_path.read_text())
    # Original runner payload merged.
    assert payload["extra"] == "preserved"
    assert payload["score"]["overall"] == 0.7
    # Watch metadata layered on top under its own key.
    assert "sway_watch" in payload
    assert payload["sway_watch"]["overall_score"] == 0.7
