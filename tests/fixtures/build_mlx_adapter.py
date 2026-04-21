"""Reproducible MLX adapter fixture for the smoke test.

Run this once on darwin-arm64 with the ``[mlx]`` extra installed to
generate a tiny LoRA adapter under
``tests/fixtures/mlx_adapter_smollm2_135m/``. The integration test
at ``tests/integration/test_mlx_smoke.py`` uses the directory if it
exists, else skips with a pointer here.

This script is deliberately *not* vendored as a binary in the repo —
regenerating it from scratch stays reproducible, keeps the repo small,
and lets us bump the base model without re-checking binaries in.

Usage:

    # Prerequisites: darwin-arm64 + ``uv pip install -e ".[mlx]"``
    python tests/fixtures/build_mlx_adapter.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if sys.platform != "darwin":
        print("build_mlx_adapter.py requires darwin-arm64", file=sys.stderr)
        return 1

    try:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.tuner.utils import linear_to_lora_layers
    except ImportError as exc:
        print(
            f"mlx / mlx_lm not importable: {exc}\n"
            "Install the [mlx] extra: uv pip install -e '.[mlx]'",
            file=sys.stderr,
        )
        return 1

    model_id = "mlx-community/SmolLM2-135M-Instruct-4bit"
    out_dir = Path(__file__).parent / "mlx_adapter_smollm2_135m"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_id} (will download on first run)…")
    model, _tokenizer = load(model_id)

    # Apply a tiny LoRA shim over the attention projections and
    # randomize the A/B weights deterministically.
    lora_config = {
        "rank": 4,
        "alpha": 8,
        "dropout": 0.0,
        "scale": 10.0,
        "keys": ["q_proj", "v_proj"],
    }
    linear_to_lora_layers(model, num_layers=2, config=lora_config)

    # Seed-scale lora_B so the adapter actually changes outputs.
    mx.random.seed(0)
    params = dict(model.trainable_parameters())
    for name, arr in params.items():
        if "lora_b" in name.lower():
            params[name] = mx.random.normal(arr.shape) * 0.05

    # Save adapters to the mlx_lm convention: <dir>/adapters.safetensors
    # plus an adapter_config.json stub mlx_lm.load can find.
    import json

    from mlx.utils import tree_flatten
    from safetensors.numpy import save_file as save_safetensors

    save_safetensors(
        str(out_dir / "adapters.safetensors"),
        {k: v.astype(mx.float16) for k, v in tree_flatten(params)},
    )
    (out_dir / "adapter_config.json").write_text(json.dumps(lora_config, indent=2))
    print(f"Wrote adapter fixture to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
