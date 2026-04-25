"""Unit tests for the ``gradient_ghost`` probe (Sprint 25, F01-style).

Builds synthetic ``training_state.pt`` + ``adapter_model.safetensors``
fixtures so every verdict branch (PASS / FAIL / WARN / SKIP / ERROR)
runs without needing a real dlm install. The end-to-end check against
a real dlm-store fixture lives in
``tests/integration/test_probe_gradient_ghost.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# torch + safetensors ride the [hf] extra. Skip the whole module
# when missing rather than fail collection — same idiom as
# tests/unit/test_mlx_convert.py.
torch = pytest.importorskip("torch", reason="needs the [hf] extra (torch)")
safetensors_numpy = pytest.importorskip(
    "safetensors.numpy", reason="needs the [hf] extra (safetensors)"
)

from dlm_sway.core.errors import (  # noqa: E402 — import-after-skip
    BackendNotAvailableError,
    MissingTrainingStateError,
)
from dlm_sway.core.result import Verdict  # noqa: E402
from dlm_sway.probes._param_id_mapping import (  # noqa: E402
    ParamMappingError,
    map_param_ids_to_layers,
)
from dlm_sway.probes._training_state import (  # noqa: E402
    TrainingStateError,
    load_training_state,
)
from dlm_sway.probes.base import RunContext, build_probe  # noqa: E402
from dlm_sway.probes.gradient_ghost import GradientGhostProbe  # noqa: E402


def _write_synthetic_safetensors(
    dst: Path,
    *,
    num_layers: int = 4,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    rank: int = 8,
    in_features: int = 64,
    out_features: int = 64,
) -> int:
    """Write a PEFT-shaped safetensors fixture next to the training
    state. Returns the total number of weight keys (matches the
    expected number of optimizer-state params)."""
    weights: dict[str, np.ndarray] = {}
    for layer_idx in range(num_layers):
        for mod in target_modules:
            base = f"base_model.model.model.layers.{layer_idx}.self_attn.{mod}"
            weights[f"{base}.lora_A.weight"] = np.zeros((rank, in_features), dtype=np.float32)
            weights[f"{base}.lora_B.weight"] = np.zeros((out_features, rank), dtype=np.float32)
    safetensors_numpy.save_file(weights, str(dst / "adapter_model.safetensors"))
    return len(weights)


def _write_synthetic_training_state(
    dst: Path,
    *,
    global_step: int,
    num_params: int,
    exp_avg_sq_per_param: list[float] | None = None,
    nan_per_param: bool = False,
) -> None:
    """Write a minimal ``training_state.pt`` whose shape matches
    dlm's contract.

    ``exp_avg_sq_per_param`` lets a test plant per-param means (one
    float per param-id) for the per-layer ratio branches.
    ``nan_per_param=True`` sets every exp_avg_sq tensor to NaN
    (proves the all-NaN FAIL branch).
    """
    if exp_avg_sq_per_param is None:
        exp_avg_sq_per_param = [1.0] * num_params

    state_dict: dict[int, dict[str, object]] = {}
    for pid, sq_mean in enumerate(exp_avg_sq_per_param):
        if nan_per_param:
            tensor = torch.full((4,), float("nan"), dtype=torch.float32)
        else:
            tensor = torch.full((4,), float(sq_mean), dtype=torch.float32)
        state_dict[pid] = {
            "step": torch.tensor(float(global_step)),
            "exp_avg": torch.zeros((4,), dtype=torch.float32),
            "exp_avg_sq": tensor,
        }

    payload = {
        "optimizer_state_dict": {
            "state": state_dict,
            "param_groups": [{"lr": 1e-4, "params": list(range(num_params))}],
        },
        "scheduler_state_dict": {},
        "scaler_state_dict": None,
        "torch_rng_state": torch.zeros(8, dtype=torch.uint8),
        "cuda_rng_state": None,
        "numpy_rng_state": None,
        "python_random_state": None,
        "global_step": global_step,
        "epoch": float(global_step),
        "best_val_loss": float("inf"),
        "dlm_manifest_hash": None,
        "base_model_revision": "deadbeef",
        "pinned_versions": {"torch": "2.11.0"},
        "use_qlora": False,
    }
    torch.save(payload, str(dst / "training_state.pt"))


# === Tests ===


class TestProbeRegistry:
    def test_kind_registered(self) -> None:
        """Probe must be discoverable via build_probe."""
        probe, _ = build_probe(
            {"name": "x", "kind": "gradient_ghost", "adapter_path": "/nonexistent"}
        )
        assert isinstance(probe, GradientGhostProbe)

    def test_needs_backend_false(self) -> None:
        """needs_backend=False enables the runner's skip-backend path."""
        assert GradientGhostProbe.needs_backend is False

    def test_category_calibration(self) -> None:
        """Category must match the sprint's classification."""
        assert GradientGhostProbe.category == "calibration"


class TestVerdictLadder:
    """Each branch in the verdict ladder gets its own test."""

    def test_pass_when_global_step_high_and_distribution_flat(self, tmp_path: Path) -> None:
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        num_keys = _write_synthetic_safetensors(adapter, num_layers=4)
        # Flat distribution — every param has the same exp_avg_sq.
        _write_synthetic_training_state(
            adapter,
            global_step=200,
            num_params=num_keys,
            exp_avg_sq_per_param=[1.0] * num_keys,
        )

        probe, spec = build_probe(
            {"name": "gg", "kind": "gradient_ghost", "adapter_path": str(adapter)}
        )
        result = probe.run(spec, RunContext())
        assert result.verdict == Verdict.PASS
        assert result.evidence["global_step"] == 200
        assert result.evidence["frac_layers_undertrained"] == 0.0

    def test_fail_when_global_step_below_threshold(self, tmp_path: Path) -> None:
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        num_keys = _write_synthetic_safetensors(adapter, num_layers=4)
        _write_synthetic_training_state(adapter, global_step=2, num_params=num_keys)

        probe, spec = build_probe(
            {"name": "gg", "kind": "gradient_ghost", "adapter_path": str(adapter)}
        )
        result = probe.run(spec, RunContext())
        assert result.verdict == Verdict.FAIL
        assert result.evidence["global_step"] == 2
        assert result.evidence["primary_signal"] == "global_step_below_threshold"
        assert "severely undertrained" in (result.message or "")

    def test_fail_when_all_exp_avg_sq_nan(self, tmp_path: Path) -> None:
        """Even with global_step >= threshold, every NaN per-param
        triggers a separate FAIL branch — training propagated nothing."""
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        num_keys = _write_synthetic_safetensors(adapter, num_layers=4)
        _write_synthetic_training_state(
            adapter,
            global_step=200,
            num_params=num_keys,
            nan_per_param=True,
        )

        probe, spec = build_probe(
            {"name": "gg", "kind": "gradient_ghost", "adapter_path": str(adapter)}
        )
        result = probe.run(spec, RunContext())
        assert result.verdict == Verdict.FAIL
        assert result.evidence["primary_signal"] == "all_optimizer_state_nan"
        assert result.evidence["num_nonfinite_exp_avg_sq"] == num_keys

    def test_warn_when_some_layers_high_but_under_threshold(self, tmp_path: Path) -> None:
        """A heavy-tailed exp_avg_sq distribution where < layer_failure_frac
        of layers cross the per-layer threshold → WARN."""
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        # 4 layers × 2 modules × 2 factors = 16 params.
        num_keys = _write_synthetic_safetensors(adapter, num_layers=4)
        # Flat baseline; bump layer 0's params to 3× (above the 2×
        # threshold) but only 1 of 4 layers crosses (25%).
        magnitudes = [1.0] * num_keys
        for pid in range(4):  # First 4 params = layer 0
            magnitudes[pid] = 3.0
        _write_synthetic_training_state(
            adapter,
            global_step=200,
            num_params=num_keys,
            exp_avg_sq_per_param=magnitudes,
        )

        probe, spec = build_probe(
            {
                "name": "gg",
                "kind": "gradient_ghost",
                "adapter_path": str(adapter),
                "layer_failure_frac": 0.5,  # Need >50% to FAIL.
            }
        )
        result = probe.run(spec, RunContext())
        assert result.verdict == Verdict.WARN
        assert result.evidence["num_layers_undertrained"] == 1
        assert result.evidence["frac_layers_undertrained"] == pytest.approx(0.25)

    def test_fail_when_too_many_layers_high(self, tmp_path: Path) -> None:
        """When more than layer_failure_frac of layers cross the
        per-layer threshold, secondary signal also FAILs."""
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        num_keys = _write_synthetic_safetensors(adapter, num_layers=4)
        # First 12 params = first 3 layers all bumped 3×; last layer flat.
        magnitudes = [3.0] * 12 + [1.0] * 4
        _write_synthetic_training_state(
            adapter,
            global_step=200,
            num_params=num_keys,
            exp_avg_sq_per_param=magnitudes,
        )

        probe, spec = build_probe(
            {
                "name": "gg",
                "kind": "gradient_ghost",
                "adapter_path": str(adapter),
                "layer_failure_frac": 0.3,
            }
        )
        result = probe.run(spec, RunContext())
        assert result.verdict == Verdict.FAIL
        assert result.evidence["frac_layers_undertrained"] == pytest.approx(0.75)

    def test_skip_when_training_state_missing(self, tmp_path: Path) -> None:
        """No training_state.pt → SKIP (legitimate for non-dlm
        adapters), not ERROR."""
        adapter = tmp_path / "adapter-no-state"
        adapter.mkdir()
        # adapter_model.safetensors doesn't matter — probe SKIPs first.
        probe, spec = build_probe(
            {"name": "gg", "kind": "gradient_ghost", "adapter_path": str(adapter)}
        )
        result = probe.run(spec, RunContext())
        assert result.verdict == Verdict.SKIP
        assert "training_state.pt" in (result.message or "")


class TestParamIdMapping:
    """The layer-grouping helper is exercised indirectly via probe
    runs above; this class adds direct coverage of edge cases."""

    def test_correct_layer_groupings(self, tmp_path: Path) -> None:
        adapter = tmp_path / "a"
        adapter.mkdir()
        num_keys = _write_synthetic_safetensors(adapter, num_layers=3, target_modules=("q_proj",))
        # 3 layers × 1 module × 2 factors = 6 keys.
        assert num_keys == 6
        grouping = map_param_ids_to_layers(adapter, num_params=num_keys)
        assert grouping.num_layers == 3
        assert grouping.params_per_layer == 2
        assert [grouping.layer_of[i] for i in range(6)] == [0, 0, 1, 1, 2, 2]

    def test_missing_safetensors_raises(self, tmp_path: Path) -> None:
        adapter = tmp_path / "empty"
        adapter.mkdir()
        with pytest.raises(ParamMappingError, match="missing"):
            map_param_ids_to_layers(adapter, num_params=10)

    def test_mismatched_param_count_raises(self, tmp_path: Path) -> None:
        adapter = tmp_path / "a"
        adapter.mkdir()
        num_keys = _write_synthetic_safetensors(adapter, num_layers=2)
        # Pretend the optimizer has fewer params than safetensors keys.
        with pytest.raises(ParamMappingError, match="adapter / state mismatch"):
            map_param_ids_to_layers(adapter, num_params=num_keys - 2)


class TestTrainingStateLoader:
    def test_missing_file_raises_typed(self, tmp_path: Path) -> None:
        with pytest.raises(MissingTrainingStateError):
            load_training_state(tmp_path)

    def test_corrupt_pickle_raises_typed(self, tmp_path: Path) -> None:
        (tmp_path / "training_state.pt").write_bytes(b"not a pickle")
        with pytest.raises(TrainingStateError, match="failed to torch.load"):
            load_training_state(tmp_path)

    def test_unexpected_top_level_shape_raises(self, tmp_path: Path) -> None:
        torch.save({"foo": "bar"}, str(tmp_path / "training_state.pt"))
        with pytest.raises(TrainingStateError, match="missing 'optimizer_state_dict'"):
            load_training_state(tmp_path)


class TestRunnerSkipsBackend:
    """S25 P5 — runner contract when only no-backend probes scheduled."""

    def test_runs_with_none_backend_when_only_pre_run_probes(self, tmp_path: Path) -> None:
        """A spec containing only gradient_ghost runs with backend=None."""
        from dlm_sway.core.model import ModelSpec
        from dlm_sway.suite.runner import run as run_suite
        from dlm_sway.suite.spec import SuiteDefaults, SuiteModels, SwaySpec

        adapter = tmp_path / "adapter"
        adapter.mkdir()
        num_keys = _write_synthetic_safetensors(adapter, num_layers=4)
        _write_synthetic_training_state(adapter, global_step=2, num_params=num_keys)

        spec = SwaySpec(
            version=1,
            models=SuiteModels(
                base=ModelSpec(base="dummy", kind="dummy"),
                ft=ModelSpec(base="dummy", kind="dummy", adapter=adapter),
            ),
            defaults=SuiteDefaults(seed=0),
            suite=[
                {
                    "name": "gg",
                    "kind": "gradient_ghost",
                    "adapter_path": str(adapter),
                }
            ],
        )
        result = run_suite(spec, backend=None, spec_path="<test>")
        assert len(result.probes) == 1
        assert result.probes[0].verdict == Verdict.FAIL
        assert result.backend_stats == {}  # No backend means no stats.

    def test_raises_when_backend_required_but_none(self, tmp_path: Path) -> None:
        """A spec with delta_kl + None backend → BackendNotAvailableError."""
        from dlm_sway.core.model import ModelSpec
        from dlm_sway.suite.runner import run as run_suite
        from dlm_sway.suite.spec import SuiteDefaults, SuiteModels, SwaySpec

        spec = SwaySpec(
            version=1,
            models=SuiteModels(
                base=ModelSpec(base="dummy", kind="dummy"),
                ft=ModelSpec(base="dummy", kind="dummy"),
            ),
            defaults=SuiteDefaults(seed=0),
            suite=[
                {"name": "dk", "kind": "delta_kl", "prompts": ["x"]},
            ],
        )
        with pytest.raises(BackendNotAvailableError, match="delta_kl"):
            run_suite(spec, backend=None, spec_path="<test>")
