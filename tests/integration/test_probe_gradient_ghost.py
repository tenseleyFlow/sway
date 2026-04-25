"""S25 — gradient_ghost integration tests.

Two flavors:

1. **Real-store (skipped on CI):** runs against a known-undertrained
   adapter at ``~/.dlm/store/01KPPFAB2Z6DWCWY0QV702TSTX/`` if
   present. This is the prove-the-value test the sprint DoD requires
   on a real dlm-trained adapter. Skipped cleanly when the store is
   absent so CI without local dlm install still passes.
2. **Synthetic-converged (runs everywhere):** writes a fully-formed
   converged training_state.pt + matching safetensors fixture and
   asserts PASS. Pairs with the real-store FAIL case to give end-
   to-end "FAIL on undertrained, PASS on converged" coverage in CI.

Marked ``slow + online`` because building a synthetic converged
training_state.pt requires torch-pickle round-tripping a real-shape
optimizer state — heavier than a unit test should be.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="needs the [hf] extra (torch)")
safetensors_numpy = pytest.importorskip(
    "safetensors.numpy", reason="needs the [hf] extra (safetensors)"
)

from dlm_sway.core.result import Verdict  # noqa: E402
from dlm_sway.probes.base import RunContext, build_probe  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.online]


_REAL_STORE_PATH = (
    Path.home() / ".dlm" / "store" / "01KPPFAB2Z6DWCWY0QV702TSTX" / "adapter" / "versions" / "v0001"
)


def test_real_undertrained_dlm_store_fails(tmp_path: Path) -> None:
    """If a known dlm-trained undertrained adapter is on disk, the
    probe must FAIL on it.

    Skipped on machines without the local fixture (CI). The store
    was the ground-truth artifact that drove the sprint design — it
    was a real ``--max-steps 2`` smoke-test run.
    """
    if not (_REAL_STORE_PATH / "training_state.pt").exists():
        pytest.skip(
            f"no dlm store fixture at {_REAL_STORE_PATH} — skipping the "
            "real-adapter prove-the-value test (synthetic test below "
            "still runs)"
        )

    probe, spec = build_probe(
        {
            "name": "gg_real",
            "kind": "gradient_ghost",
            "adapter_path": str(_REAL_STORE_PATH),
        }
    )
    result = probe.run(spec, RunContext())

    assert result.verdict == Verdict.FAIL, (
        f"expected FAIL on a known-undertrained dlm store, got {result.verdict}: {result.message}"
    )
    # The real fixture is global_step=2 — a clean primary-signal hit.
    assert result.evidence["global_step"] < 50
    assert result.evidence["primary_signal"] in (
        "global_step_below_threshold",
        "all_optimizer_state_nan",
    )


def _build_converged_fixture(adapter_dir: Path) -> int:
    """Write a synthetic 'converged' adapter pair.

    - safetensors with realistic per-layer LoRA tensor names
    - training_state.pt with global_step=500 (well above threshold)
      and a flat per-param exp_avg_sq distribution (no layer
      crosses the per-layer ratio).
    """
    adapter_dir.mkdir(parents=True, exist_ok=True)
    num_layers = 4
    target_modules = ("q_proj", "v_proj")
    rank = 8
    in_features = 64

    weights: dict[str, np.ndarray] = {}
    for layer_idx in range(num_layers):
        for mod in target_modules:
            base = f"base_model.model.model.layers.{layer_idx}.self_attn.{mod}"
            weights[f"{base}.lora_A.weight"] = np.zeros((rank, in_features), dtype=np.float32)
            weights[f"{base}.lora_B.weight"] = np.zeros((in_features, rank), dtype=np.float32)
    safetensors_numpy.save_file(weights, str(adapter_dir / "adapter_model.safetensors"))
    num_keys = len(weights)

    # Flat distribution: every param's exp_avg_sq is 0.1 (a small but
    # finite value typical of a converged Adam state).
    state_dict: dict[int, dict[str, object]] = {}
    for pid in range(num_keys):
        state_dict[pid] = {
            "step": torch.tensor(500.0),
            "exp_avg": torch.zeros((4,), dtype=torch.float32),
            "exp_avg_sq": torch.full((4,), 0.1, dtype=torch.float32),
        }

    payload = {
        "optimizer_state_dict": {
            "state": state_dict,
            "param_groups": [{"lr": 1e-4, "params": list(range(num_keys))}],
        },
        "scheduler_state_dict": {},
        "scaler_state_dict": None,
        "torch_rng_state": torch.zeros(8, dtype=torch.uint8),
        "cuda_rng_state": None,
        "numpy_rng_state": None,
        "python_random_state": None,
        "global_step": 500,
        "epoch": 5.0,
        "best_val_loss": 0.42,
        "dlm_manifest_hash": None,
        "base_model_revision": "synthetic-test-fixture",
        "pinned_versions": {"torch": "2.11.0"},
        "use_qlora": False,
    }
    torch.save(payload, str(adapter_dir / "training_state.pt"))
    return num_keys


def test_synthetic_converged_adapter_passes(tmp_path: Path) -> None:
    """A hand-rolled converged training_state (global_step=500, flat
    exp_avg_sq distribution) must PASS.

    Together with the real-store FAIL test above, covers the
    sprint's prove-the-value: 'undertrained → FAIL, converged → PASS'.
    """
    adapter_dir = tmp_path / "synthetic-converged"
    _build_converged_fixture(adapter_dir)

    probe, spec = build_probe(
        {
            "name": "gg_synth",
            "kind": "gradient_ghost",
            "adapter_path": str(adapter_dir),
        }
    )
    result = probe.run(spec, RunContext())

    assert result.verdict == Verdict.PASS, (
        f"expected PASS on a synthetic converged adapter, got {result.verdict}: {result.message}"
    )
    assert result.evidence["global_step"] == 500
    assert result.evidence["frac_layers_undertrained"] == 0.0
    assert result.evidence["num_layers"] == 4


def test_runner_skips_backend_for_pure_pre_run_suite(tmp_path: Path) -> None:
    """End-to-end: a suite containing only gradient_ghost runs
    successfully with backend=None. Confirms the S25 P5 runner
    contract holds end-to-end (not just at the probe level)."""
    from dlm_sway.core.model import ModelSpec
    from dlm_sway.suite.runner import run as run_suite
    from dlm_sway.suite.spec import SuiteDefaults, SuiteModels, SwaySpec

    adapter_dir = tmp_path / "synthetic-converged"
    _build_converged_fixture(adapter_dir)

    spec = SwaySpec(
        version=1,
        models=SuiteModels(
            base=ModelSpec(base="dummy", kind="dummy"),
            ft=ModelSpec(base="dummy", kind="dummy", adapter=adapter_dir),
        ),
        defaults=SuiteDefaults(seed=0),
        suite=[
            {
                "name": "gg",
                "kind": "gradient_ghost",
                "adapter_path": str(adapter_dir),
            },
        ],
    )
    result = run_suite(spec, backend=None, spec_path="<integration>")
    assert len(result.probes) == 1
    assert result.probes[0].verdict == Verdict.PASS
    # No backend, no backend stats.
    assert result.backend_stats == {}
