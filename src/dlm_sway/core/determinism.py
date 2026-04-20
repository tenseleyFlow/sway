"""Deterministic-execution helper.

Mirrors ``dlm.train.determinism.seed_everything`` so running the same
suite twice on the same host produces the same :class:`ProbeResult`
payloads. The dlm project treats determinism as a contract; sway takes
the same posture for scoring operations.

Generation is allowed to use non-deterministic attention kernels when
``temperature > 0``, because a deterministic sampled generation is a
contradiction. Scoring (logprobs, rolling logprobs, next-token dists)
always runs under :func:`torch.use_deterministic_algorithms(True)`.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Literal

DeterminismClass = Literal["strict", "best_effort", "loose"]


@dataclass(frozen=True, slots=True)
class DeterminismSummary:
    """What seeding actually accomplished, for logging in the report."""

    class_: DeterminismClass
    seed: int
    notes: tuple[str, ...] = ()


def seed_everything(seed: int, *, strict: bool = True) -> DeterminismSummary:
    """Seed every RNG sway's probes touch and flip backend flags.

    Idempotent — safe to call repeatedly with the same seed.

    Parameters
    ----------
    seed:
        The seed. Callers typically use the value from ``sway.yaml``'s
        ``defaults.seed`` (default 0).
    strict:
        If ``True`` (the default), request deterministic CUDA algorithms
        and set ``CUBLAS_WORKSPACE_CONFIG``. Scoring probes need this;
        generation-only runs can set it ``False``.

    Returns
    -------
    :class:`DeterminismSummary` with a classification:

    - ``"strict"`` — deterministic algorithms active, no warnings.
    - ``"best_effort"`` — platform doesn't support full determinism
      (MPS, some CPU kernels).
    - ``"loose"`` — seeded but deterministic algorithms refused.
    """

    notes: list[str] = []
    clazz: DeterminismClass = "best_effort"

    # Env vars must come first — torch reads them at cuBLAS init.
    if strict:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)

    # numpy is a hard dep; safe to seed unconditionally.
    import numpy as np

    np.random.seed(seed)

    try:
        import torch  # noqa: PLC0415 — lazy: torch is an optional extra.
    except ModuleNotFoundError:
        notes.append("torch not installed; seeded python + numpy only")
        return DeterminismSummary(class_="best_effort", seed=seed, notes=tuple(notes))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        clazz = "strict"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        clazz = "best_effort"
        notes.append("MPS: bit-identical across runs is best-effort")
    else:
        clazz = "best_effort"
        notes.append("CPU-only backend: strict determinism depends on BLAS impl")

    if strict:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
        except Exception as exc:  # noqa: BLE001 — torch raises a naked Exception
            clazz = "loose"
            notes.append(f"deterministic algorithms refused: {exc}")

    return DeterminismSummary(class_=clazz, seed=seed, notes=tuple(notes))
