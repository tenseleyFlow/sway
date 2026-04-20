"""Integration-test configuration.

Integration tests need network + heavy deps. Re-export the tiny_model
fixture here so test modules can pick it up without a long import
path.
"""

from __future__ import annotations

from tests.fixtures.tiny_model import tiny_model_dir  # noqa: F401 — re-export
