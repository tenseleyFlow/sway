"""End-to-end loop: watchdog → run_suite → JSON written to history.

This is the integration counterpart to ``tests/unit/test_watch_observer.py``.
It runs against the real ``dlm_sway.suite.runner`` + the in-process dummy
backend (no model load), so it exercises the same shapes a real
``sway watch <spec>`` session would, just without paying for an HF
download. The slow-lane sister test for HF backend warm-path lives in
``tests/integration/test_serve_warm_path.py`` (S36).

Marked ``slow`` because it sleeps for several deadlines while waiting
for filesystem-event delivery and a real probe sweep.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("watchdog")

pytestmark = [pytest.mark.slow]

from dlm_sway.suite import report  # noqa: E402
from dlm_sway.watch.core import (  # noqa: E402
    RunnerFn,
    RunResult,
    Watcher,
    WatcherConfig,
    start_observing,
)

SPEC_YAML = """\
version: 1
models:
  base: { kind: dummy, base: dummy-base }
  ft:   { kind: dummy, base: dummy-base }
defaults:
  seed: 0
  coverage_threshold: 0.0
suite:
  - { name: smoke, kind: delta_kl, prompts: [hello] }
"""


@pytest.fixture
def fake_store(tmp_path: Path) -> tuple[Path, Path]:
    store = tmp_path / "store"
    (store / "adapter" / "versions" / "v0001").mkdir(parents=True)
    pointer = store / "adapter" / "current.txt"
    pointer.write_text("adapter/versions/v0001\n", encoding="utf-8")
    return store, pointer


@pytest.fixture
def real_spec(tmp_path: Path) -> Path:
    p = tmp_path / "spec.yaml"
    p.write_text(SPEC_YAML, encoding="utf-8")
    return p


def _real_runner_against_dummy_backend() -> RunnerFn:
    """Build a runner that uses the real run_suite + dummy backend.

    The dummy backend can't be constructed from a ModelSpec (by design
    — see ``backends/__init__.py``); we hand it in directly so the test
    doesn't depend on the HF wheels being present.
    """
    from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
    from dlm_sway.suite.runner import run as run_suite
    from dlm_sway.suite.score import compute as compute_score

    def _run(
        spec_path: Path, spec_obj: object, pointer_text: str, serve_url: str | None
    ) -> RunResult:  # noqa: ARG001
        backend = DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
        result = run_suite(spec_obj, backend, spec_path=str(spec_path))  # type: ignore[arg-type]
        score_obj = compute_score(result)
        payload = report.to_json(result, score_obj)
        return RunResult(
            verdict="pass",
            json_payload=payload,
            run_seconds=0.01,
            overall_score=float(score_obj.overall),
        )

    return _run


def test_pointer_change_drives_real_run_suite(
    real_spec: Path,
    fake_store: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """A pointer rewrite triggers a full run_suite within 5s.

    Asserts:
    - history JSON appears
    - JSON parses, has the standard ``probes`` block from
      ``report.to_json`` (proves we ran the real runner, not a stub)
    - ``sway_watch`` envelope is present and matches the new pointer
    """
    store, pointer = fake_store
    history = tmp_path / "history"
    seen: list[object] = []

    watcher = Watcher(
        WatcherConfig(spec_path=real_spec, history_dir=history),
        pointer_path=pointer,
        runner=_real_runner_against_dummy_backend(),
        on_outcome=seen.append,
    )

    observed = start_observing(watcher, debounce_s=0.05)
    try:
        (store / "adapter" / "versions" / "v0002").mkdir()
        pointer.write_text("adapter/versions/v0002\n", encoding="utf-8")
        deadline = time.monotonic() + 5.0
        while not seen and time.monotonic() < deadline:
            time.sleep(0.05)
        assert seen, "no outcome within 5s — observer didn't pick up the pointer rewrite"
    finally:
        observed.stop()

    # Standard sway report shape from a real run_suite.
    payload = json.loads(seen[0].result_path.read_text())  # type: ignore[attr-defined]
    assert "probes" in payload, f"expected real run_suite output, got: {sorted(payload)}"
    assert payload["sway_watch"]["adapter_pointer"] == "adapter/versions/v0002"
    # Sequence is 1-indexed and counts only fires that produced a result.
    assert payload["sway_watch"]["sequence"] == 1


def test_three_pointer_changes_produce_three_history_entries(
    real_spec: Path,
    fake_store: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    """Sprint DoD: 3 pointer changes → 3 history JSONs."""
    store, pointer = fake_store
    history = tmp_path / "history"
    seen: list[object] = []

    watcher = Watcher(
        WatcherConfig(spec_path=real_spec, history_dir=history),
        pointer_path=pointer,
        runner=_real_runner_against_dummy_backend(),
        on_outcome=seen.append,
    )
    observed = start_observing(watcher, debounce_s=0.05)
    try:
        for n in range(2, 5):
            (store / "adapter" / "versions" / f"v{n:04d}").mkdir()
            pointer.write_text(f"adapter/versions/v{n:04d}\n", encoding="utf-8")
            # Wait for each fire to land before the next write so the
            # debounce window doesn't merge them.
            target = len(seen) + 1
            deadline = time.monotonic() + 5.0
            while len(seen) < target and time.monotonic() < deadline:
                time.sleep(0.05)
        assert len(seen) == 3
    finally:
        observed.stop()

    files = sorted(history.glob("*.result.json"))
    assert len(files) == 3
