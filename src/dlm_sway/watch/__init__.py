"""``sway watch`` — re-run a spec on every dlm adapter pointer change.

Public surface re-exports:

- :class:`WatcherConfig` — paths + behaviour knobs.
- :class:`Watcher` — pure orchestrator. Stateful but watchdog-agnostic.
- :class:`RunOutcome` — what a single fire produced.
- :class:`RunResult` — what the injected runner returned.
- :func:`start_observing` — wires a :class:`Watcher` to watchdog.

The CLI is :func:`dlm_sway.cli.commands.watch_cmd`.
"""

from __future__ import annotations

from dlm_sway.watch.core import (
    RunOutcome,
    RunResult,
    Watcher,
    WatcherConfig,
    resolve_pointer_path,
    start_observing,
)

__all__ = [
    "RunOutcome",
    "RunResult",
    "Watcher",
    "WatcherConfig",
    "resolve_pointer_path",
    "start_observing",
]
