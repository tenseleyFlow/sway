"""Watch a dlm adapter pointer and re-run a sway spec on every change.

The pointer file is ``<store>/adapter/current.txt`` written by dlm
on each retrain. We tail it with watchdog (optional dep — install
the ``[watch]`` extra), debounce events, re-resolve the .dlm handle,
re-run the spec, and write a JSON result to the history dir. dlm's
write is atomic but watchdog can fire mid-rename — we retry the
read up to ``retry_attempts`` times with ``retry_backoff_s`` backoff.

The :class:`Watcher` is deliberately watchdog-agnostic. It exposes
:meth:`Watcher.trigger_once` (read pointer, run, write result) which
is what tests exercise. :func:`start_observing` glues a Watcher to
``watchdog.Observer`` for production use.

Architecture decision: the runner — what actually executes a suite
— is injected. The CLI builds one that calls ``_execute_spec`` (or
``ServeClient.run`` when ``$SWAY_SERVE_URL`` is set). Tests pass a
stub. Keeps the orchestrator small and unit-testable without an
HF backend, and makes serve-delegation a one-liner switch in the
CLI rather than a branch through the watcher.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dlm_sway.core.errors import SwayError

if TYPE_CHECKING:
    from dlm_sway.suite.spec import SwaySpec

_log = logging.getLogger(__name__)


# Public structural types --------------------------------------------------


@dataclass(frozen=True, slots=True)
class WatcherConfig:
    """Knobs for one ``sway watch`` invocation.

    Attributes
    ----------
    spec_path:
        Path to the sway YAML spec. Re-read on every fire if its
        mtime changes (the user can edit probes mid-watch).
    history_dir:
        Where per-fire result JSONs are written. Created on first
        fire if it does not exist.
    max_history:
        Cap on retained result JSONs. Older files (by mtime) are
        deleted after each successful fire. ``0`` disables rotation.
    on_fail_cmd:
        Optional shell command to spawn when the fire's verdict is
        ``fail``. Spawned with ``SWAY_RESULT_PATH`` set to the JSON
        path. Fire-and-forget; we don't wait for it.
    serve_url:
        When set (typically from ``$SWAY_SERVE_URL``), the runner
        delegates the fire to a ``sway serve`` daemon over HTTP
        instead of building a local backend. Caller-injected runners
        must respect this flag.
    retry_attempts:
        How many times to retry pointer-file reads when the read
        races with dlm's atomic rename. ``3`` ≈ 600 ms total backoff.
    retry_backoff_s:
        Sleep between retries. Linear, not exponential — the race
        is short.
    """

    spec_path: Path
    history_dir: Path
    max_history: int = 100
    on_fail_cmd: str | None = None
    serve_url: str | None = None
    retry_attempts: int = 3
    retry_backoff_s: float = 0.2


@dataclass(frozen=True, slots=True)
class RunResult:
    """What the runner returned to the watcher.

    The runner — see :class:`RunnerFn` — is responsible for actually
    executing a suite. The watcher just packages the result, writes
    it to disk, and rotates history.
    """

    verdict: str
    """``"pass"`` | ``"fail"`` | ``"error"`` — the gate-style verdict.

    A run is ``fail`` when any probe failed or when the composite
    overall score fell below the spec's coverage_threshold. ``error``
    means a runtime exception bubbled out of the suite.
    """

    json_payload: str
    """The full report JSON (matches ``sway run --json`` output)."""

    run_seconds: float
    """Wall-clock seconds for the suite execution itself."""

    overall_score: float | None = None
    """Composite SwayScore overall, when computed."""


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """Per-fire summary returned by :meth:`Watcher.trigger_once`."""

    sequence: int
    """1-indexed fire counter for this Watcher instance."""

    result_path: Path
    """JSON file the watcher wrote."""

    verdict: str
    """Mirrors :attr:`RunResult.verdict`."""

    run_seconds: float
    overall_score: float | None
    spec_reloaded: bool
    adapter_pointer: str
    """Raw text of ``current.txt`` at the time of the fire — useful
    for downstream tooling that wants to dedupe."""


# Runner protocol ----------------------------------------------------------


# A runner takes (spec_path, spec, adapter_pointer, serve_url) and
# returns a RunResult. We pass the resolved-and-validated spec so the
# runner doesn't re-read the YAML; we pass the pointer so the runner
# can stamp it into reports / pass to the daemon for cache lookups.
RunnerFn = Callable[[Path, "SwaySpec", str, str | None], RunResult]


# Helpers ------------------------------------------------------------------


def resolve_pointer_path(store_root: Path) -> Path | None:
    """Locate the dlm adapter pointer for a store root.

    Checks ``<store>/adapter/current.txt`` first (canonical since
    dlm v0.9). Falls back to ``<store>/adapter/latest`` symlink for
    older dlm builds. Returns ``None`` when neither exists.

    The returned :class:`~pathlib.Path` is the *pointer*, not the
    adapter dir — the watcher reads its contents on each fire to
    avoid caching a stale value across atomic renames.
    """
    canonical = store_root / "adapter" / "current.txt"
    if canonical.exists():
        return canonical
    legacy = store_root / "adapter" / "latest"
    if legacy.exists() or legacy.is_symlink():
        return legacy
    return None


def _read_pointer(pointer_path: Path) -> str:
    """Read ``current.txt`` as a stripped string, or read a symlink target."""
    if pointer_path.is_symlink():
        # ``adapter/latest`` legacy form; the symlink target *is* the
        # version dir.
        return str(pointer_path.readlink())
    return pointer_path.read_text(encoding="utf-8").strip()


# Watcher ------------------------------------------------------------------


class Watcher:
    """Pure-Python orchestrator for ``sway watch``.

    The Watcher holds:

    - The :class:`WatcherConfig`.
    - The path to the dlm adapter pointer (resolved once at init).
    - A monotonically increasing fire counter.
    - The last successfully resolved pointer text (for dedupe).
    - The last spec mtime (for spec-edit detection).

    It does **not** know about watchdog. :func:`start_observing`
    wires it to the OS event loop. Tests call :meth:`trigger_once`
    directly with a stubbed runner.
    """

    def __init__(
        self,
        config: WatcherConfig,
        *,
        pointer_path: Path,
        runner: RunnerFn,
        on_outcome: Callable[[RunOutcome], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        if not config.spec_path.exists():
            raise SwayError(f"spec not found: {config.spec_path}")
        if config.max_history < 0:
            raise SwayError(f"max_history must be >= 0, got {config.max_history}")
        if config.retry_attempts < 1:
            raise SwayError(f"retry_attempts must be >= 1, got {config.retry_attempts}")
        self._config = config
        self._pointer_path = pointer_path
        self._runner = runner
        self._on_outcome = on_outcome
        self._on_error = on_error
        self._sequence = 0
        self._last_pointer: str | None = None
        self._last_spec_mtime: float | None = None
        self._spec: SwaySpec | None = None
        self._lock = threading.Lock()

    # -- read-only views ---------------------------------------------------

    @property
    def config(self) -> WatcherConfig:
        return self._config

    @property
    def pointer_path(self) -> Path:
        return self._pointer_path

    @property
    def sequence(self) -> int:
        return self._sequence

    @property
    def last_pointer(self) -> str | None:
        return self._last_pointer

    # -- main entry point --------------------------------------------------

    def trigger_once(self, *, force: bool = False) -> RunOutcome | None:
        """Resolve pointer, run, write result. Idempotent.

        Returns ``None`` when the pointer is unchanged from the last
        successful fire (and ``force=False``). Otherwise returns the
        :class:`RunOutcome`. Exceptions propagate after the
        ``on_error`` hook fires.
        """
        # Serialize fires so concurrent watchdog events don't double-run.
        with self._lock:
            try:
                return self._trigger_locked(force=force)
            except Exception as exc:  # noqa: BLE001 — last-resort hook
                if self._on_error is not None:
                    try:
                        self._on_error(exc)
                    except Exception:  # noqa: BLE001
                        _log.exception("on_error hook raised")
                raise

    def _trigger_locked(self, *, force: bool) -> RunOutcome | None:
        pointer_text = self._read_pointer_with_retries()
        if not force and pointer_text == self._last_pointer:
            _log.debug("pointer unchanged (%r); skipping run", pointer_text)
            return None

        spec, reloaded = self._reload_spec_if_changed()

        self._sequence += 1
        sequence = self._sequence
        run_started = time.monotonic()
        try:
            run = self._runner(
                self._config.spec_path,
                spec,
                pointer_text,
                self._config.serve_url,
            )
        except Exception as exc:
            run_seconds = time.monotonic() - run_started
            payload = json.dumps(
                {
                    "verdict": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "sequence": sequence,
                    "adapter_pointer": pointer_text,
                    "run_seconds": run_seconds,
                },
                indent=2,
                sort_keys=True,
            )
            run = RunResult(
                verdict="error",
                json_payload=payload,
                run_seconds=run_seconds,
            )

        result_path = self._write_history(sequence, run, pointer_text)
        self._rotate_history()
        outcome = RunOutcome(
            sequence=sequence,
            result_path=result_path,
            verdict=run.verdict,
            run_seconds=run.run_seconds,
            overall_score=run.overall_score,
            spec_reloaded=reloaded,
            adapter_pointer=pointer_text,
        )
        # Only commit "last seen" on success — a half-written pointer
        # that produced an error shouldn't suppress the next retry.
        if run.verdict != "error":
            self._last_pointer = pointer_text

        if run.verdict == "fail" and self._config.on_fail_cmd:
            self._spawn_on_fail(self._config.on_fail_cmd, result_path)

        if self._on_outcome is not None:
            try:
                self._on_outcome(outcome)
            except Exception:  # noqa: BLE001
                _log.exception("on_outcome hook raised")

        return outcome

    # -- helpers -----------------------------------------------------------

    def _read_pointer_with_retries(self) -> str:
        last_exc: Exception | None = None
        for attempt in range(self._config.retry_attempts):
            try:
                if not self._pointer_path.exists():
                    raise SwayError(f"adapter pointer missing: {self._pointer_path}")
                text = _read_pointer(self._pointer_path)
                if not text:
                    # Half-write: file exists but is empty mid-rename.
                    raise SwayError("adapter pointer is empty")
                return text
            except (OSError, SwayError) as exc:
                last_exc = exc
                if attempt + 1 < self._config.retry_attempts:
                    time.sleep(self._config.retry_backoff_s)
        # Out of retries.
        raise SwayError(
            f"failed to read adapter pointer after {self._config.retry_attempts} attempts: "
            f"{last_exc}"
        )

    def _reload_spec_if_changed(self) -> tuple[SwaySpec, bool]:
        """Return (spec, reloaded). Reloads from disk if mtime changed."""
        from dlm_sway.suite.loader import load_spec

        mtime = self._config.spec_path.stat().st_mtime
        if self._spec is None or self._last_spec_mtime != mtime:
            self._spec = load_spec(self._config.spec_path)
            reloaded = self._last_spec_mtime is not None
            self._last_spec_mtime = mtime
            if reloaded:
                _log.info("spec %s reloaded (mtime changed)", self._config.spec_path)
            return self._spec, reloaded
        return self._spec, False

    def _write_history(self, sequence: int, run: RunResult, pointer_text: str) -> Path:
        self._config.history_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        # Sequence in the filename guarantees ordering even when two
        # fires happen inside the same second (rare but possible on
        # NVMe with the warm-backend daemon).
        name = f"{ts}-{sequence:05d}.result.json"
        path = self._config.history_dir / name

        # Annotate the payload with watch-specific metadata. The base
        # JSON shape stays untouched so downstream tools that already
        # parse ``sway run`` output keep working.
        try:
            base_payload: dict[str, Any] = json.loads(run.json_payload)
        except json.JSONDecodeError:
            base_payload = {"raw": run.json_payload}
        base_payload["sway_watch"] = {
            "sequence": sequence,
            "adapter_pointer": pointer_text,
            "spec_path": str(self._config.spec_path),
            "verdict": run.verdict,
            "run_seconds": run.run_seconds,
            "overall_score": run.overall_score,
            "wrote_at": ts,
        }
        path.write_text(json.dumps(base_payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _rotate_history(self) -> None:
        if self._config.max_history <= 0:
            return
        try:
            entries = sorted(
                self._config.history_dir.glob("*.result.json"),
                key=lambda p: p.stat().st_mtime,
            )
        except FileNotFoundError:
            return
        excess = len(entries) - self._config.max_history
        for old in entries[:excess] if excess > 0 else []:
            try:
                old.unlink()
            except OSError as exc:
                _log.warning("failed to prune history file %s: %s", old, exc)

    def _spawn_on_fail(self, cmd: str, result_path: Path) -> None:
        env = dict(os.environ)
        env["SWAY_RESULT_PATH"] = str(result_path)
        try:
            subprocess.Popen(  # noqa: S602 — user-supplied shell command, intentional
                cmd,
                shell=True,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
        except OSError as exc:
            _log.warning("failed to spawn --on-fail command %r: %s", cmd, exc)


# Watchdog wiring ----------------------------------------------------------


@dataclass(slots=True)
class _ObservedWatch:
    """Handle returned by :func:`start_observing`. Caller calls ``.stop()``."""

    observer: Any
    """``watchdog.observers.Observer`` instance — kept generic to avoid
    forcing watchdog onto callers that just want type hints."""

    stop: Callable[[], None]
    """Stop the observer and join its worker thread."""


def start_observing(
    watcher: Watcher,
    *,
    debounce_s: float = 0.1,
) -> _ObservedWatch:
    """Wire a :class:`Watcher` to a watchdog ``Observer``.

    Watches the *parent directory* of the pointer (watchdog can't
    reliably watch a single file across atomic renames on Linux),
    filters events to the pointer's filename, debounces by
    ``debounce_s`` (multiple syscalls during one rename collapse to
    one fire), and calls :meth:`Watcher.trigger_once`.

    Requires the ``[watch]`` extra. Raises :class:`SwayError` with
    an actionable message when watchdog isn't installed.
    """
    try:
        from watchdog.events import FileSystemEvent, FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError as exc:
        raise SwayError(
            "sway watch requires the [watch] extra: pip install 'dlm-sway[watch]'"
        ) from exc

    target_name = watcher.pointer_path.name
    target_dir = watcher.pointer_path.parent
    if not target_dir.exists():
        raise SwayError(f"pointer directory does not exist: {target_dir}")

    state = _DebouncedTrigger(watcher=watcher, debounce_s=debounce_s)

    class _Handler(FileSystemEventHandler):
        def on_any_event(self, event: FileSystemEvent) -> None:  # noqa: ARG002
            # Filter by basename — watchdog reports source AND dest
            # paths on rename; treat any of them landing on
            # ``current.txt`` as a fire.
            src = getattr(event, "src_path", "")
            dest = getattr(event, "dest_path", "")
            if Path(src).name == target_name or Path(dest).name == target_name:
                state.fire()

    observer = Observer()
    observer.schedule(_Handler(), str(target_dir), recursive=False)
    observer.start()

    def _stop() -> None:
        observer.stop()
        observer.join(timeout=5.0)
        state.shutdown()

    return _ObservedWatch(observer=observer, stop=_stop)


@dataclass(slots=True)
class _DebouncedTrigger:
    """Coalesces N watchdog events arriving within ``debounce_s`` into one fire.

    A single dlm adapter rename produces 4-6 events on Linux (modify,
    move from temp, create, modify metadata, ...). Without coalescing,
    we'd fire the suite once per event.
    """

    watcher: Watcher
    debounce_s: float
    _timer: threading.Timer | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _shutdown: bool = field(default=False, init=False)

    def fire(self) -> None:
        with self._lock:
            if self._shutdown:
                return
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_s, self._run)
            self._timer.daemon = True
            self._timer.start()

    def _run(self) -> None:
        try:
            self.watcher.trigger_once()
        except Exception:  # noqa: BLE001 — keep observer alive
            _log.exception("watch fire raised; observer remains active")

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
