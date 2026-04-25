"""Map optimizer-state param-ids → adapter layer indices (S25, P2).

The ``gradient_ghost`` probe needs to attribute per-param Adam stats
(``exp_avg_sq`` magnitudes) back to layer indices for the per-layer
reporting the sprint DoD requires.

## Why not name-level attribution?

The ``optimizer_state_dict.state`` dict in dlm's ``training_state.pt``
keys parameters by integer id (0..N-1) in the order PEFT registered
them with the optimizer. The full param-id → name mapping requires
walking ``model.named_parameters()`` — but the probe's whole pitch
is "no model load." So we punt on naming individual modules and
group at the layer level instead.

## Why layer-grouping is safe

PEFT registers trainable LoRA params in **layer-major order**: for
each transformer block, all its LoRA factors first, then the next
block. Verified against dlm-trained SmolLM2-135M:

- 30 transformer layers
- 4 target modules per layer (q_proj, k_proj, v_proj, o_proj)
- 2 LoRA factors per module (A, B)
- Total: 30 × 4 × 2 = 240 params, in layer-major order

Even when the probe doesn't know *which* module within a layer a
given param belongs to, grouping by layer index gives meaningful
per-layer health reporting — and matches how a user thinks about
"my adapter's layer 5 isn't learning."
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from dlm_sway.core.errors import SwayError


class ParamMappingError(SwayError):
    """Raised when ``adapter_model.safetensors`` can't be read or
    the param-count doesn't divide evenly across layers."""


@dataclass(frozen=True, slots=True)
class LayerGrouping:
    """Result of grouping per-param ids by transformer-layer index.

    Attributes
    ----------
    layer_indices:
        Sorted, deduped list of transformer-layer indices the
        adapter touches (e.g. ``[0, 1, 2, ..., 29]`` for SmolLM2).
    params_per_layer:
        How many trainable params each layer has (constant across
        layers — PEFT applies the same target_modules everywhere).
    layer_of:
        Function mapping a param-id to its transformer layer index,
        or ``None`` when the param-id falls outside the layered
        param space (e.g. embedding overrides via
        ``modules_to_save``).
    """

    layer_indices: tuple[int, ...]
    params_per_layer: int
    layer_of: dict[int, int]

    @property
    def num_layers(self) -> int:
        return len(self.layer_indices)


_LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")


def map_param_ids_to_layers(adapter_dir: Path, num_params: int) -> LayerGrouping:
    """Group ``optimizer_state_dict.state`` param-ids by layer index.

    Parameters
    ----------
    adapter_dir:
        Adapter directory containing ``adapter_model.safetensors``.
        We don't load the safetensors weights themselves — just
        introspect the keys to count params per layer.
    num_params:
        Number of param-ids in the optimizer state. Used as a
        cross-check: the safetensors key count should match.

    Returns
    -------
    LayerGrouping
        Per-layer grouping ready for the probe to attribute stats.

    Raises
    ------
    ParamMappingError
        ``adapter_model.safetensors`` is missing OR the param count
        doesn't divide evenly by the layer count (unexpected adapter
        shape — PEFT's per-layer module set wasn't uniform).
    """
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if not safetensors_path.exists():
        raise ParamMappingError(
            f"adapter_model.safetensors missing at {safetensors_path} — "
            "can't recover layer indices for per-layer reporting"
        )

    try:
        from safetensors import safe_open  # noqa: PLC0415 — lazy
    except ImportError as exc:
        raise ParamMappingError(
            "safetensors not installed — gradient_ghost needs it for "
            "per-layer attribution. Install with: pip install 'dlm-sway[hf]'"
        ) from exc

    # Use ``safe_open`` so we never materialize the tensors — we only
    # need the key list. Saves the 7+ MB read on a typical adapter.
    # safetensors ships no py.typed; safe_open is untyped to mypy.
    with safe_open(  # type: ignore[no-untyped-call]
        str(safetensors_path), framework="numpy", device="cpu"
    ) as fh:
        keys = list(fh.keys())
    if not keys:
        raise ParamMappingError(f"{safetensors_path}: no keys found")

    # Extract layer index from each key. Keys without a layer index
    # (modules_to_save full-tensor overrides) are tracked separately
    # and excluded from per-layer attribution.
    keys_by_layer: dict[int, int] = {}
    for k in keys:
        match = _LAYER_INDEX_RE.search(k)
        if match is None:
            continue
        idx = int(match.group(1))
        keys_by_layer[idx] = keys_by_layer.get(idx, 0) + 1

    if not keys_by_layer:
        raise ParamMappingError(
            f"{safetensors_path}: no keys carry a layer index — adapter "
            "may target only embeddings / lm_head"
        )

    layer_indices = tuple(sorted(keys_by_layer))
    counts = set(keys_by_layer.values())
    if len(counts) != 1:
        # Heterogeneous per-layer param counts. The simple "K per
        # layer in order" mapping breaks; refuse rather than guess.
        raise ParamMappingError(
            f"{safetensors_path}: per-layer param count is heterogeneous "
            f"({sorted(set(keys_by_layer.values()))}) — gradient_ghost "
            "can't safely attribute params to layers in this configuration"
        )
    params_per_layer = next(iter(counts))

    # Cross-check vs optimizer-state count. They should match: each
    # safetensors key has one optimizer-state entry. If we see fewer
    # optimizer params than safetensors keys, the user pruned state;
    # if we see more, modules_to_save params have their own optimizer
    # entries (not handled here).
    expected = params_per_layer * len(layer_indices)
    if num_params < expected:
        raise ParamMappingError(
            f"optimizer state has {num_params} params but safetensors "
            f"has {expected} layered weight keys — adapter / state mismatch"
        )

    layer_of: dict[int, int] = {}
    for ordinal in range(expected):
        layer_of[ordinal] = layer_indices[ordinal // params_per_layer]
    # Param-ids beyond the layered range (modules_to_save) get None
    # via dict.get(...) — the probe filters them out.

    return LayerGrouping(
        layer_indices=layer_indices,
        params_per_layer=params_per_layer,
        layer_of=layer_of,
    )
