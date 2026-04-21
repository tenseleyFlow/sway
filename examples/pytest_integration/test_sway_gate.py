"""Example — @pytest.mark.sway replaces a subprocess ``sway gate`` wrapper.

Install the plugin alongside the HF backend you use::

    pip install 'dlm-sway[hf,pytest]'

Then run it like any other pytest file::

    pytest examples/pytest_integration/test_sway_gate.py -v

Each sway probe lands as its own test item in the pytest report:
``test_adapter_healthy::adherence``, ``test_adapter_healthy::calibration``,
``test_adapter_healthy::__gate__``. Probe-level failures isolate; a
failing adherence probe doesn't mask a failing calibration one.

The single `threshold` kwarg adds a synthetic ``__gate__`` item that
fires only when the composite score drops below the given value —
one place to put the CI regression gate.
"""

from __future__ import annotations

import pytest

# -------- the one-liner --------


@pytest.mark.sway(spec="sway.yaml", threshold=0.6)
def test_adapter_healthy() -> None:
    """Sway-gated CI check. The decorator owns the body."""


# -------- what it replaces --------
#
# Before:
#
#     import subprocess
#
#     def test_adapter_healthy_legacy() -> None:
#         result = subprocess.run(
#             ["sway", "gate", "sway.yaml", "--threshold", "0.6"],
#             capture_output=True, text=True, check=False,
#         )
#         assert result.returncode == 0, (
#             f"sway gate failed:\nstdout:\n{result.stdout}\n"
#             f"stderr:\n{result.stderr}"
#         )
#
# Problems with the legacy shape:
#
#   * Per-probe failures collapse into one big "sway gate failed" —
#     users have to scrape stdout to know which probe regressed.
#   * ``pytest -k adherence`` can't select just one probe.
#   * No per-probe marker filtering, no JUnit-XML per probe, no
#     integration with ``pytest-html`` / ``pytest --lf`` / any of
#     pytest's ecosystem.
#   * Slow-test markers have to be applied to the one wrapper — can't
#     say "fast lane, skip the ablation probe but keep the others."
#
# After (with ``@pytest.mark.sway``):
#
#   * Each probe is its own test item. ``pytest -k calibration`` runs
#     just that probe.
#   * FAIL / ERROR → pytest Failed; SKIP → pytest Skipped; WARN →
#     pytest warning; so the whole pytest ecosystem reads verdicts
#     correctly.
#   * ``--junitxml`` produces one <testcase> per probe — CI dashboards
#     can parse it with their existing pipeline.
#   * Suite runs **once per decorated test**, cached across synthetic
#     items (no N× model load tax from the expansion).
