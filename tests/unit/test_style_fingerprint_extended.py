"""Tests for the extended (9-dim) style fingerprint.

The extended path requires the ``style`` extra (spaCy + textstat).
We don't gate the test on the extra being installed — instead we test
that:

1. ``extended=False`` always returns 6 dims (backward-compat with the
   v1 fingerprint contract).
2. ``extended=True`` returns 9 dims **when** ``_has_style_extra()`` is
   true; the test patches the extra-detection to simulate both states.
3. The probe's ``extended="on"`` SKIPs cleanly when the extra is
   missing rather than crashing.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.result import Verdict
from dlm_sway.probes.base import RunContext, build_probe
from dlm_sway.probes.style_fingerprint import (
    FINGERPRINT_SCHEMA_VERSION,
    fingerprint,
)


def _backend_with_generations(prompts: list[str]) -> DummyDifferentialBackend:
    base = {p: f"base {p} response with several words and a period." for p in prompts}
    ft = {p: f"ft {p} response, longer, with extra punctuation, more words." for p in prompts}
    return DummyDifferentialBackend(
        base=DummyResponses(generations=base),
        ft=DummyResponses(generations=ft),
    )


class TestFingerprintFunction:
    def test_default_is_6_dim(self) -> None:
        fp = fingerprint("Hello there. This is a test sentence. Another one follows.")
        assert fp.shape == (6,)

    def test_empty_text_returns_zeros_at_requested_dim(self) -> None:
        assert fingerprint("", extended=False).shape == (6,)
        assert fingerprint("", extended=True).shape == (9,)

    def test_extended_falls_back_when_extra_missing(self) -> None:
        with patch("dlm_sway.probes.style_fingerprint._extended_fingerprint", return_value=None):
            fp = fingerprint("Some text. Another sentence.", extended=True)
            assert fp.shape == (6,)


class TestProbeExtendedOnRequiresExtra:
    def test_extended_on_skips_without_extra(self) -> None:
        prompts = ["p1"]
        backend = _backend_with_generations(prompts)
        probe, spec = build_probe(
            {
                "name": "sf",
                "kind": "style_fingerprint",
                "prompts": prompts,
                "doc_reference": "The reference document. Has some sentences. Three of them.",
                "extended": "on",
            }
        )
        ctx = RunContext(backend=backend)
        with patch("dlm_sway.probes.style_fingerprint._has_style_extra", return_value=False):
            result = probe.run(spec, ctx)
        assert result.verdict == Verdict.SKIP
        assert "style" in result.message.lower()


class TestProbeExtendedAuto:
    def test_auto_off_when_extra_missing(self) -> None:
        prompts = ["p1"]
        backend = _backend_with_generations(prompts)
        probe, spec = build_probe(
            {
                "name": "sf",
                "kind": "style_fingerprint",
                "prompts": prompts,
                "doc_reference": "The reference document. Has some sentences.",
                "extended": "auto",
            }
        )
        ctx = RunContext(backend=backend)
        with patch("dlm_sway.probes.style_fingerprint._has_style_extra", return_value=False):
            result = probe.run(spec, ctx)
        assert result.evidence.get("extended") is False
        assert len(result.evidence["base_fp"]) == 6
        assert result.evidence["schema_version"] == FINGERPRINT_SCHEMA_VERSION


class TestExtendedProducesNineDimWhenSimulated:
    def test_extended_path_returns_9_dim(self) -> None:
        """Patch ``_extended_fingerprint`` to return a 9-vector and confirm
        ``fingerprint(..., extended=True)`` passes it through unchanged."""
        nine = np.arange(9, dtype=np.float64) / 9.0
        with patch(
            "dlm_sway.probes.style_fingerprint._extended_fingerprint",
            return_value=nine,
        ):
            fp = fingerprint("Some text. Another sentence.", extended=True)
            assert fp.shape == (9,)
            np.testing.assert_array_equal(fp, nine)
