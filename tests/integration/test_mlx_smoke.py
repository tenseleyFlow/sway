"""C5: MLX backend smoke test (darwin-arm64-only).

Exercises ``MLXDifferentialBackend.next_token_dist`` and ``generate``
on a real mlx_lm-loaded model with a small LoRA adapter attached.
The adapter is built once via ``tests/fixtures/build_mlx_adapter.py``
(see that script's docstring); this test skips when the fixture
directory is absent rather than building it on the fly, since fixture
builds take ~30s and re-running the test should be cheap.

Skips on:
- non-darwin platforms (mlx is Apple Silicon only)
- non-arm64 architectures
- missing ``mlx_lm`` import
- missing fixture directory
"""

from __future__ import annotations

import math
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.online]

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "mlx_adapter_smollm2_135m"
_MODEL_ID = "mlx-community/SmolLM2-135M-Instruct-4bit"


def _platform_supports_mlx() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


@pytest.fixture(scope="module")
def mlx_backend():
    if not _platform_supports_mlx():
        pytest.skip("MLX requires darwin-arm64")
    pytest.importorskip("mlx_lm", reason="install the [mlx] extra to run MLX tests")

    if not _FIXTURE_DIR.exists():
        pytest.skip(
            f"missing MLX adapter fixture at {_FIXTURE_DIR} — generate it via "
            f"`python tests/fixtures/build_mlx_adapter.py`"
        )

    from dlm_sway.backends.mlx import MLXDifferentialBackend
    from dlm_sway.core.model import ModelSpec

    backend = MLXDifferentialBackend(
        base_spec=ModelSpec(base=_MODEL_ID, kind="mlx"),
        adapter_path=_FIXTURE_DIR,
    )
    yield backend
    backend.close()


def test_next_token_dist_returns_finite_topk(mlx_backend) -> None:
    with mlx_backend.as_base() as b:
        d = b.next_token_dist("The capital of France is", top_k=32)
    assert d.token_ids.shape == (32,)
    assert d.logprobs.shape == (32,)
    assert np.all(np.isfinite(d.logprobs))
    # Top-k must be sorted in descending probability order.
    assert np.all(np.diff(d.logprobs) <= 1e-7)


def test_adapter_changes_distribution(mlx_backend) -> None:
    """The whole point of the differential backend: base ≠ ft."""
    prompt = "The adapter does"
    with mlx_backend.as_base() as b:
        base_dist = b.next_token_dist(prompt, top_k=32)
    with mlx_backend.as_finetuned() as f:
        ft_dist = f.next_token_dist(prompt, top_k=32)
    same_ids = np.array_equal(base_dist.token_ids, ft_dist.token_ids)
    if same_ids:
        assert not np.allclose(base_dist.logprobs, ft_dist.logprobs, atol=1e-5)


def test_logprob_of_finite(mlx_backend) -> None:
    with mlx_backend.as_base() as b:
        lp = b.logprob_of("The capital of France is", " Paris")
    assert math.isfinite(lp)
    assert lp < 0.0


def test_generate_returns_nonempty_string(mlx_backend) -> None:
    with mlx_backend.as_base() as b:
        out = b.generate("Hello", max_new_tokens=8, seed=0)
    assert isinstance(out, str)
    assert len(out) > 0
