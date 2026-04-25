"""Unit tests for the PEFT→MLX-LM LoRA converter (Sprint 24, F01).

Builds synthetic PEFT-shaped inputs (no torch / peft install required —
just safetensors + json + numpy) and asserts the converter produces
mlx-lm-shaped outputs with the right keys, shape transposes, and
config fields.

End-to-end verification against real PEFT and a real
``mlx_lm.load(..., adapter_path=...)`` call lives in
``tests/integration/test_mlx_converter_e2e.py`` (slow + online +
darwin-arm64).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from dlm_sway.backends._mlx_convert import (
    MlxConvertError,
    _extract_layer_index,
    _strip_layer_prefix,
    convert_peft_to_mlx,
)


def _write_synthetic_peft_adapter(
    dst: Path,
    *,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    num_layers: int = 3,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    in_features: int = 64,
    out_features: int = 64,
    modules_to_save: list[str] | None = None,
) -> None:
    """Produce a minimal but format-correct PEFT adapter directory."""
    dst.mkdir(parents=True, exist_ok=True)
    weights: dict[str, np.ndarray] = {}
    for layer_idx in range(num_layers):
        for module in target_modules:
            base = f"base_model.model.model.layers.{layer_idx}.self_attn.{module}"
            # PEFT shapes: lora_A=(r, in), lora_B=(out, r)
            weights[f"{base}.lora_A.weight"] = (
                np.random.RandomState(layer_idx).randn(rank, in_features).astype(np.float32)
            )
            weights[f"{base}.lora_B.weight"] = (
                np.random.RandomState(layer_idx + 1000).randn(out_features, rank).astype(np.float32)
            )
    save_file(weights, str(dst / "adapter_model.safetensors"))
    config = {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "target_modules": list(target_modules),
        "modules_to_save": modules_to_save or [],
        "task_type": "CAUSAL_LM",
        "bias": "none",
    }
    (dst / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")


class TestStripLayerPrefix:
    """The helper that turns full attribute paths into layer-relative
    paths for MLX's ``adapter_config.json::keys`` field."""

    def test_typical_decoder_layer_path(self) -> None:
        assert _strip_layer_prefix("model.layers.5.self_attn.q_proj") == "self_attn.q_proj"

    def test_gpt2_style_path(self) -> None:
        assert _strip_layer_prefix("transformer.h.0.attn.c_attn") == "attn.c_attn"

    def test_no_layer_index_returns_input(self) -> None:
        """Embedding-style paths (no numeric segment) pass through."""
        assert _strip_layer_prefix("model.embed_tokens") == "model.embed_tokens"

    def test_extract_layer_index(self) -> None:
        assert _extract_layer_index("model.layers.0.self_attn.q_proj") == 0
        assert _extract_layer_index("model.layers.42.self_attn.q_proj") == 42
        assert _extract_layer_index("model.embed_tokens") is None


class TestConvertPeftToMlxBasic:
    """Happy path: standard PEFT LoRA → MLX adapter."""

    def test_produces_expected_output_files(self, tmp_path: Path) -> None:
        src = tmp_path / "peft"
        dst = tmp_path / "mlx"
        _write_synthetic_peft_adapter(src)

        report = convert_peft_to_mlx(src, dst)

        assert (dst / "adapters.safetensors").exists()
        assert (dst / "adapter_config.json").exists()
        assert report["rank"] == 8
        assert report["scale"] == pytest.approx(2.0)  # 16 / 8
        assert report["num_keys"] == 12  # 3 layers × 2 modules × 2 (lora_a + lora_b)
        assert report["num_layers"] == 3

    def test_mlx_config_shape(self, tmp_path: Path) -> None:
        """The written ``adapter_config.json`` matches mlx-lm's
        ``load_adapters`` expectations."""
        src = tmp_path / "peft"
        dst = tmp_path / "mlx"
        _write_synthetic_peft_adapter(src, rank=16, alpha=32, dropout=0.1, num_layers=4)

        convert_peft_to_mlx(src, dst)
        cfg = json.loads((dst / "adapter_config.json").read_text(encoding="utf-8"))

        assert cfg["fine_tune_type"] == "lora"
        assert cfg["num_layers"] == 4
        params = cfg["lora_parameters"]
        assert params["rank"] == 16
        assert params["scale"] == pytest.approx(2.0)  # 32 / 16
        assert params["dropout"] == pytest.approx(0.1)
        assert params["keys"] == ["self_attn.q_proj", "self_attn.v_proj"]

    def test_lora_factor_shapes_transposed(self, tmp_path: Path) -> None:
        """PEFT lora_A=(r, in) → MLX lora_a=(in, r); same for lora_B/lora_b."""
        src = tmp_path / "peft"
        dst = tmp_path / "mlx"
        _write_synthetic_peft_adapter(src, rank=8, in_features=64, out_features=128, num_layers=1)
        # Sanity-check the synthetic input first.
        peft_w = load_file(str(src / "adapter_model.safetensors"))
        a_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        b_key = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
        assert peft_w[a_key].shape == (8, 64)
        assert peft_w[b_key].shape == (128, 8)

        convert_peft_to_mlx(src, dst)
        mlx_w = load_file(str(dst / "adapters.safetensors"))
        assert "model.layers.0.self_attn.q_proj.lora_a" in mlx_w
        assert "model.layers.0.self_attn.q_proj.lora_b" in mlx_w
        assert mlx_w["model.layers.0.self_attn.q_proj.lora_a"].shape == (64, 8)
        assert mlx_w["model.layers.0.self_attn.q_proj.lora_b"].shape == (8, 128)

    def test_values_preserved_through_transpose(self, tmp_path: Path) -> None:
        """Round-trip the underlying numbers — transpose must be the
        only operation, not a reshape with data loss."""
        src = tmp_path / "peft"
        dst = tmp_path / "mlx"
        _write_synthetic_peft_adapter(src, num_layers=1)

        peft_w = load_file(str(src / "adapter_model.safetensors"))
        convert_peft_to_mlx(src, dst)
        mlx_w = load_file(str(dst / "adapters.safetensors"))

        a_in = peft_w["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
        a_out = mlx_w["model.layers.0.self_attn.q_proj.lora_a"]
        np.testing.assert_array_equal(a_in.T, a_out)


class TestConvertPeftToMlxErrors:
    """Structural errors must surface as ``MlxConvertError`` with
    actionable messages, not as cryptic IO / KeyError tracebacks."""

    def test_missing_safetensors_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "empty"
        src.mkdir()
        (src / "adapter_config.json").write_text('{"peft_type": "LORA", "r": 8}')
        with pytest.raises(MlxConvertError, match="missing adapter_model.safetensors"):
            convert_peft_to_mlx(src, tmp_path / "mlx")

    def test_missing_config_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "empty"
        src.mkdir()
        save_file({}, str(src / "adapter_model.safetensors"))
        with pytest.raises(MlxConvertError, match="missing adapter_config.json"):
            convert_peft_to_mlx(src, tmp_path / "mlx")

    def test_non_lora_peft_type_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "ia3"
        _write_synthetic_peft_adapter(src)
        cfg = json.loads((src / "adapter_config.json").read_text())
        cfg["peft_type"] = "IA3"
        (src / "adapter_config.json").write_text(json.dumps(cfg))
        with pytest.raises(MlxConvertError, match="unsupported PEFT type"):
            convert_peft_to_mlx(src, tmp_path / "mlx")

    def test_invalid_rank_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "bad"
        _write_synthetic_peft_adapter(src)
        cfg = json.loads((src / "adapter_config.json").read_text())
        cfg["r"] = 0
        (src / "adapter_config.json").write_text(json.dumps(cfg))
        with pytest.raises(MlxConvertError, match="invalid LoRA rank"):
            convert_peft_to_mlx(src, tmp_path / "mlx")

    def test_dst_not_empty_refuses_without_overwrite(self, tmp_path: Path) -> None:
        src = tmp_path / "peft"
        dst = tmp_path / "mlx"
        _write_synthetic_peft_adapter(src)
        # Pre-create the output to simulate a stale conversion.
        dst.mkdir()
        (dst / "adapters.safetensors").write_bytes(b"old")
        with pytest.raises(MlxConvertError, match="overwrite=True"):
            convert_peft_to_mlx(src, dst)

    def test_dst_overwrite_replaces_existing(self, tmp_path: Path) -> None:
        src = tmp_path / "peft"
        dst = tmp_path / "mlx"
        _write_synthetic_peft_adapter(src)
        dst.mkdir()
        (dst / "adapters.safetensors").write_bytes(b"old")
        convert_peft_to_mlx(src, dst, overwrite=True)
        # Should not be the placeholder bytes any more.
        assert (dst / "adapters.safetensors").read_bytes()[:4] != b"old\x00"

    def test_unexpected_key_prefix_raises(self, tmp_path: Path) -> None:
        """A safetensors file whose keys don't have the
        ``base_model.model.`` PEFT-wrapper prefix shouldn't be silently
        emitted with a wrong MLX path."""
        src = tmp_path / "weird"
        src.mkdir()
        save_file(
            {"some.other.lora_A.weight": np.zeros((8, 64), dtype=np.float32)},
            str(src / "adapter_model.safetensors"),
        )
        cfg = {
            "peft_type": "LORA",
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["x"],
        }
        (src / "adapter_config.json").write_text(json.dumps(cfg))
        with pytest.raises(MlxConvertError, match="missing 'base_model.model.' prefix"):
            convert_peft_to_mlx(src, tmp_path / "mlx")


class TestEnsureMlxAdapterAutoConvert:
    """``MLXDifferentialBackend.__init__`` calls ``_ensure_mlx_adapter``
    to upgrade PEFT-shaped adapter dirs to MLX format on the fly. The
    function lives in ``backends/mlx.py`` so it doesn't pull mlx-lm
    when the path is already MLX-shaped."""

    def test_passes_through_when_dir_is_already_mlx_shape(self, tmp_path: Path) -> None:
        """Existing ``adapters.safetensors`` → no conversion, return
        the same path unchanged. (Manual conversions / pre-built MLX
        adapters from other tools must not be re-converted.)"""
        from dlm_sway.backends.mlx import _ensure_mlx_adapter

        mlx_dir = tmp_path / "mlx"
        mlx_dir.mkdir()
        save_file({}, str(mlx_dir / "adapters.safetensors"))
        (mlx_dir / "adapter_config.json").write_text('{"fine_tune_type":"lora"}')
        out = _ensure_mlx_adapter(mlx_dir)
        assert out == mlx_dir

    def test_auto_converts_peft_dir_into_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A PEFT-shaped dir gets converted into XDG_CACHE_HOME on
        first call; the returned path is the cache dir, not the source."""
        from dlm_sway.backends.mlx import _ensure_mlx_adapter

        cache_root = tmp_path / "cache"
        monkeypatch.setenv("XDG_CACHE_HOME", str(cache_root))

        peft_dir = tmp_path / "peft"
        _write_synthetic_peft_adapter(peft_dir)
        out = _ensure_mlx_adapter(peft_dir)

        assert out != peft_dir
        assert (out / "adapters.safetensors").exists()
        assert (out / "adapter_config.json").exists()
        assert str(out).startswith(str(cache_root))

    def test_repeated_calls_short_circuit_on_cache_hit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same PEFT bytes → same cache hash → second call returns the
        cached dir without re-converting (touch mtime to detect)."""
        from dlm_sway.backends.mlx import _ensure_mlx_adapter

        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        peft_dir = tmp_path / "peft"
        _write_synthetic_peft_adapter(peft_dir)

        first = _ensure_mlx_adapter(peft_dir)
        first_mtime = (first / "adapters.safetensors").stat().st_mtime_ns

        # Second call — should NOT rewrite the file.
        second = _ensure_mlx_adapter(peft_dir)
        assert second == first
        assert (second / "adapters.safetensors").stat().st_mtime_ns == first_mtime

    def test_passes_through_unrecognized_dir(self, tmp_path: Path) -> None:
        """A directory with neither shape — let mlx_lm.load surface
        its own error rather than this helper second-guessing."""
        from dlm_sway.backends.mlx import _ensure_mlx_adapter

        empty = tmp_path / "empty"
        empty.mkdir()
        out = _ensure_mlx_adapter(empty)
        assert out == empty


class TestModulesToSave:
    """``modules_to_save`` (e.g. embed_tokens, lm_head) must be skipped
    cleanly with a report entry, not crash the converter."""

    def test_modules_to_save_skipped_and_reported(self, tmp_path: Path) -> None:
        src = tmp_path / "peft"
        dst = tmp_path / "mlx"
        _write_synthetic_peft_adapter(src, num_layers=1, modules_to_save=["embed_tokens"])
        # Inject a non-LoRA full-weight tensor that simulates
        # PEFT's modules_to_save serialization.
        existing = load_file(str(src / "adapter_model.safetensors"))
        existing["base_model.model.model.embed_tokens.modules_to_save.default.weight"] = np.zeros(
            (100, 64), dtype=np.float32
        )
        save_file(existing, str(src / "adapter_model.safetensors"))

        report = convert_peft_to_mlx(src, dst)
        assert len(report["modules_to_save_skipped"]) == 1
        assert "embed_tokens" in report["modules_to_save_skipped"][0]
        # Real LoRA factors still extracted.
        assert report["num_keys"] == 4  # 1 layer × 2 modules × 2 factors
