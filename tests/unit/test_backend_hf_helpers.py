"""Unit tests for the pure helpers inside ``backends/hf.py``.

The helpers (``_resolve_dtype``, ``_detect_device``) gate the HF
backend's dtype/device choices. They're exercised end-to-end by the
integration suite, but doing direct unit coverage on them keeps the
slow lane focused on the actual backend behavior — and makes the
fast lane catch dtype/device regressions in seconds.
"""

from __future__ import annotations

import importlib.util

import pytest

# These tests need torch to construct the dtype objects we're asserting
# on; skip cleanly when the [hf] extra isn't installed.
if importlib.util.find_spec("torch") is None:
    pytest.skip(
        "torch not installed — install the [hf] extra to run HF helper tests",
        allow_module_level=True,
    )

from dlm_sway.backends.hf import _detect_device, _resolve_dtype


class TestResolveDtype:
    def test_explicit_fp16(self) -> None:
        import torch

        assert _resolve_dtype("fp16", "cpu") is torch.float16

    def test_explicit_bf16(self) -> None:
        import torch

        assert _resolve_dtype("bf16", "cpu") is torch.bfloat16

    def test_explicit_fp32(self) -> None:
        import torch

        assert _resolve_dtype("fp32", "cpu") is torch.float32

    def test_auto_on_cpu_picks_fp32_for_numerical_stability(self) -> None:
        import torch

        assert _resolve_dtype("auto", "cpu") is torch.float32

    def test_auto_on_mps_picks_fp16(self) -> None:
        import torch

        assert _resolve_dtype("auto", "mps") is torch.float16


class TestDetectDevice:
    def test_returns_one_of_supported_devices(self) -> None:
        device = _detect_device()
        assert device in ("cuda", "mps", "cpu")
