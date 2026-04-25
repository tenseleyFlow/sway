"""Loader for dlm's ``training_state.pt`` (Sprint 25, gradient_ghost).

The pre-run ``gradient_ghost`` probe diagnoses adapter convergence
without any model load: dlm writes the optimizer state snapshot at
end of training, and we can read its ``global_step`` + per-parameter
Adam ``exp_avg_sq`` magnitudes to answer "did this train long enough?"
in ~50 ms.

This module is **dlm-aware via file convention only** — we never
``import dlm``. We just ``torch.load`` a path the user gave us. The
file format is dlm's contract; we follow it conservatively and fail
loudly when the shape drifts.

## File-shape reference (verified against dlm 2026-04-20 stores)

```
training_state.pt:
  global_step: int
  epoch: float
  best_val_loss: float
  optimizer_state_dict:
    state: dict[int param_id → {step, exp_avg, exp_avg_sq}]
    param_groups: list[dict]   # lr, betas, eps, params: list[int]
  scheduler_state_dict: dict
  scaler_state_dict: dict | None
  *_rng_state: tensor / tuple / None
  pinned_versions: dict
  base_model_revision: str
  dlm_manifest_hash: str | None
  use_qlora: bool
```

Per-param ids are integer indices, NOT names — name attribution lives
in ``_param_id_mapping.py``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dlm_sway.core.errors import MissingTrainingStateError, SwayError


class TrainingStateError(SwayError):
    """Raised when ``training_state.pt`` exists but can't be parsed."""


@dataclass(frozen=True, slots=True)
class ParamStat:
    """Optimizer state for a single trainable parameter (one int id).

    The probe reads ``exp_avg_sq_mean`` as a proxy for "how much
    gradient variance this parameter has seen recently." Adam's
    second-moment estimate is a moving average of squared gradients;
    a high value at end-of-training means the gradients haven't
    shrunk — typical of an undertrained parameter.
    """

    param_id: int
    step: int
    exp_avg_norm: float
    exp_avg_sq_mean: float
    numel: int


@dataclass(frozen=True, slots=True)
class TrainingStateSnapshot:
    """Everything ``gradient_ghost`` needs from a ``training_state.pt``.

    Constructed by :func:`load_training_state`; consumed by
    ``GradientGhostProbe.run``. Frozen so a probe can pass it around
    without defensive copies.
    """

    global_step: int
    epoch: float
    best_val_loss: float
    per_param: tuple[ParamStat, ...]
    pinned_versions: dict[str, str]
    base_model_revision: str | None
    dlm_manifest_hash: str | None
    use_qlora: bool


def load_training_state(adapter_dir: Path) -> TrainingStateSnapshot:
    """Load + parse ``adapter_dir/training_state.pt``.

    Parameters
    ----------
    adapter_dir:
        A dlm adapter version directory (e.g.
        ``~/.dlm/store/<id>/adapter/versions/v0001/``). Must contain
        a ``training_state.pt`` file.

    Returns
    -------
    TrainingStateSnapshot
        Frozen snapshot of the fields the probe consumes.

    Raises
    ------
    MissingTrainingStateError
        ``training_state.pt`` doesn't exist under ``adapter_dir``. The
        caller should SKIP (this is a clean signal, not an error).
    TrainingStateError
        File exists but can't be parsed (torch import missing,
        unexpected dict shape, optimizer state missing). The caller
        should ERROR — something is structurally wrong.
    """
    adapter_dir = Path(adapter_dir)
    state_path = adapter_dir / "training_state.pt"
    if not state_path.exists():
        raise MissingTrainingStateError(adapter_dir)

    # Lazy-import torch so non-gradient-ghost users don't pay the
    # ~700 ms torch-import cost on `import dlm_sway`.
    try:
        import torch
    except ImportError as exc:
        raise TrainingStateError(
            "torch not installed — gradient_ghost reads pytorch-pickled "
            "training_state.pt files. Install with: pip install 'dlm-sway[hf]'"
        ) from exc

    # ``weights_only=False`` is required: dlm's training_state.pt
    # carries pickled RNG state (numpy / python random). Suppressing
    # the ``FutureWarning`` keeps the probe output clean — this is a
    # known-trusted artifact dlm produced, not arbitrary user input.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        try:
            state = torch.load(str(state_path), map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 — torch.load can raise many shapes
            raise TrainingStateError(
                f"failed to torch.load {state_path}: {type(exc).__name__}: {exc}"
            ) from exc

    if not isinstance(state, dict):
        raise TrainingStateError(f"{state_path}: expected dict, got {type(state).__name__}")
    opt = state.get("optimizer_state_dict")
    if not isinstance(opt, dict):
        raise TrainingStateError(
            f"{state_path}: missing 'optimizer_state_dict' (got {type(opt).__name__})"
        )
    per_param_state = opt.get("state")
    if not isinstance(per_param_state, dict):
        raise TrainingStateError(
            f"{state_path}: optimizer_state_dict.state is not a dict "
            f"(got {type(per_param_state).__name__})"
        )

    per_param: list[ParamStat] = []
    for pid, ps in per_param_state.items():
        if not isinstance(pid, int):
            raise TrainingStateError(
                f"{state_path}: optimizer_state_dict.state has non-int key "
                f"{pid!r} (type {type(pid).__name__})"
            )
        if not isinstance(ps, dict):
            continue  # Skip malformed entries silently — same as torch.optim does.
        step_v = _scalar_int(ps.get("step", 0))
        exp_avg = ps.get("exp_avg")
        exp_avg_sq = ps.get("exp_avg_sq")
        per_param.append(
            ParamStat(
                param_id=pid,
                step=step_v,
                exp_avg_norm=_tensor_norm(exp_avg),
                exp_avg_sq_mean=_tensor_mean(exp_avg_sq),
                numel=_tensor_numel(exp_avg_sq),
            )
        )
    per_param.sort(key=lambda s: s.param_id)

    return TrainingStateSnapshot(
        global_step=int(state.get("global_step", 0) or 0),
        epoch=float(state.get("epoch", 0.0) or 0.0),
        best_val_loss=float(state.get("best_val_loss", 0.0) or 0.0),
        per_param=tuple(per_param),
        pinned_versions=dict(state.get("pinned_versions") or {}),
        base_model_revision=state.get("base_model_revision"),
        dlm_manifest_hash=state.get("dlm_manifest_hash"),
        use_qlora=bool(state.get("use_qlora", False)),
    )


def _scalar_int(v: Any) -> int:
    """torch saves ``step`` as a 0-dim tensor; coerce safely."""
    if v is None:
        return 0
    item = getattr(v, "item", None)
    if callable(item):
        try:
            return int(item())
        except Exception:  # noqa: BLE001
            return 0
    try:
        return int(v)
    except Exception:  # noqa: BLE001
        return 0


def _tensor_norm(t: Any) -> float:
    if t is None:
        return 0.0
    norm = getattr(t, "norm", None)
    if callable(norm):
        try:
            return float(norm().item())
        except Exception:  # noqa: BLE001
            return 0.0
    return 0.0


def _tensor_mean(t: Any) -> float:
    if t is None:
        return 0.0
    mean = getattr(t, "mean", None)
    if callable(mean):
        try:
            return float(mean().item())
        except Exception:  # noqa: BLE001
            return 0.0
    return 0.0


def _tensor_numel(t: Any) -> int:
    if t is None:
        return 0
    numel = getattr(t, "numel", None)
    if callable(numel):
        try:
            return int(numel())
        except Exception:  # noqa: BLE001
            return 0
    return 0
