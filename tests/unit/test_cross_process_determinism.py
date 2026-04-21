"""F08 + audit stronger-test #12: cross-process determinism.

The ``_NullView`` RNG used ``hash(prompt)`` as part of its seed. Python
salts the built-in ``hash()`` per interpreter invocation via
``PYTHONHASHSEED``, which meant the null-calibration output was
process-salted even for a fixed ``(seed, prompt)`` pair. The dummy
backend's null-stats disk cache could serve stale values on a restart
after any CI run that set a different ``PYTHONHASHSEED``.

This test runs the same null-calibration call in two subprocesses with
explicitly different ``PYTHONHASHSEED`` values and asserts the
resulting TokenDist logprobs are byte-identical — the stable-seed fix
in concrete form. Before the fix this test fails in seconds; after, it
passes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


_WORKER = r"""
import json
import sys

import numpy as np

from dlm_sway.backends.dummy import DummyResponses, _NullView

view = _NullView(
    base_responses=DummyResponses(),
    seed=42,
    init_scale=0.02,
    rank_scale=1.0,
)
dist = view.next_token_dist("cross-process-determinism")
payload = {"logprobs": dist.logprobs.astype(np.float64).tolist()}
json.dump(payload, sys.stdout)
"""


def _run_worker(pythonhashseed: str) -> list[float]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = pythonhashseed
    result = subprocess.run(  # noqa: S603 — controlled local invocation
        [sys.executable, "-c", _WORKER],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
    )
    payload = json.loads(result.stdout)
    return list(payload["logprobs"])


def test_null_view_is_invariant_under_pythonhashseed() -> None:
    """Same seed + same prompt + different PYTHONHASHSEED → identical
    logprobs."""
    logprobs_1 = _run_worker(pythonhashseed="1")
    logprobs_2 = _run_worker(pythonhashseed="99991")
    assert logprobs_1 == logprobs_2, (
        "dummy _NullView RNG stream drifted across PYTHONHASHSEED values — "
        "the stable-seed fix in backends/dummy.py (F08) regressed."
    )
