"""Tests for :mod:`dlm_sway.core.determinism`."""

from __future__ import annotations

import os
import random

import numpy as np

from dlm_sway.core.determinism import DeterminismSummary, seed_everything


class TestSeedEverything:
    def test_returns_summary(self) -> None:
        summary = seed_everything(0)
        assert isinstance(summary, DeterminismSummary)
        assert summary.seed == 0
        assert summary.class_ in {"strict", "best_effort", "loose"}

    def test_idempotent_for_stdlib_random(self) -> None:
        seed_everything(42)
        a = [random.random() for _ in range(5)]
        seed_everything(42)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_idempotent_for_numpy(self) -> None:
        seed_everything(17)
        a = np.random.rand(5)
        seed_everything(17)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_cublas_workspace_set_under_strict(self) -> None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        seed_everything(0, strict=True)
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"

    def test_non_strict_does_not_set_cublas(self) -> None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        seed_everything(0, strict=False)
        # Non-strict mode must not leak the env var in either direction;
        # the host environment's prior value wins.
        assert (
            "CUBLAS_WORKSPACE_CONFIG" not in os.environ
            or os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8"
        )
