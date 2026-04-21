"""Tests for ``preflight_finite_check`` on the shipped backends.

The HF backend's check is exercised in the integration suite (it needs
a real model). Here we verify the dummy backend's contract and the
PreflightCheckable Protocol shape.
"""

from __future__ import annotations

import math

import numpy as np

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.scoring import PreflightCheckable, TokenDist


class TestProtocolShape:
    def test_dummy_satisfies_preflight_protocol(self) -> None:
        backend = DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
        assert isinstance(backend, PreflightCheckable)


class TestDummyDefaultIsFinite:
    def test_default_dummy_passes_preflight(self) -> None:
        backend = DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
        ok, reason = backend.preflight_finite_check()
        assert ok is True
        assert reason == ""


class TestDummyNanDetection:
    def test_nan_in_ft_token_dist_caught(self) -> None:
        nan_dist = TokenDist(
            token_ids=np.array([1, 2, 3], dtype=np.int64),
            logprobs=np.array([-0.1, math.nan, -2.0], dtype=np.float32),
            vocab_size=100,
        )
        ft = DummyResponses(token_dists={"preflight": nan_dist})
        backend = DummyDifferentialBackend(base=DummyResponses(), ft=ft)
        ok, reason = backend.preflight_finite_check()
        assert ok is False
        assert "ft view" in reason
        assert "non-finite" in reason

    def test_nan_in_base_token_dist_caught(self) -> None:
        nan_dist = TokenDist(
            token_ids=np.array([1, 2], dtype=np.int64),
            logprobs=np.array([math.nan, math.nan], dtype=np.float32),
            vocab_size=100,
        )
        base = DummyResponses(token_dists={"preflight": nan_dist})
        backend = DummyDifferentialBackend(base=base, ft=DummyResponses())
        ok, reason = backend.preflight_finite_check()
        assert ok is False
        assert "base view" in reason

    def test_inf_caught(self) -> None:
        inf_dist = TokenDist(
            token_ids=np.array([1, 2], dtype=np.int64),
            logprobs=np.array([-0.5, math.inf], dtype=np.float32),
            vocab_size=100,
        )
        ft = DummyResponses(token_dists={"preflight": inf_dist})
        backend = DummyDifferentialBackend(base=DummyResponses(), ft=ft)
        ok, _reason = backend.preflight_finite_check()
        assert ok is False
