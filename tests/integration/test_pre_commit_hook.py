"""S19 pre-commit hook integration test.

Exercises the actual runtime behavior of the ``sway-gate`` hook: set
up a tmp git repo, drop a ``.pre-commit-config.yaml`` declaring the
hook inline (mirroring the shape shipped in ``.pre-commit-hooks.yaml``),
run ``pre-commit run sway-gate --all-files`` as a subprocess, assert
the exit code + stdout banner.

Two cases cover the headline contracts:

- **Pass case.** Gate threshold 0.0 + a probe whose dummy-backend
  behavior passes ``assert_mean_gte: 0.0``. Exit 0.
- **Fail case.** Identical spec with ``assert_mean_gte: 100.0`` —
  impossible divergence floor, probe fails. Exit 1 with
  ``gate FAILED`` in stdout.

**Why not exercise the shipped ``.pre-commit-hooks.yaml`` via
``repo: .``?** That path requires committing the hooks file — fine
for production use but fragile inside a pytest fixture. The
``.pre-commit-hooks.yaml`` file itself is validated via a
parse-and-assert smoke test at module top; the runtime behavior is
validated via the inline ``repo: local`` pattern below. Together they
cover "file is valid" + "entry point works" without tangling the two.

Marked ``slow+online`` because the hook's entry point runs a real HF
backend — the fixture builds a tiny LoRA on SmolLM2-135M.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.slow, pytest.mark.online]


REPO_ROOT = Path(__file__).resolve().parents[2]
HOOKS_PATH = REPO_ROOT / ".pre-commit-hooks.yaml"


def test_pre_commit_hooks_yaml_is_valid() -> None:
    """Smoke: ``.pre-commit-hooks.yaml`` parses and declares the two
    expected hooks. Cheap — runs without network — and catches
    structural drift before the slow-lane invocation."""
    assert HOOKS_PATH.exists(), f"missing {HOOKS_PATH}"
    hooks = yaml.safe_load(HOOKS_PATH.read_text(encoding="utf-8"))
    assert isinstance(hooks, list)
    ids = [h["id"] for h in hooks]
    assert ids == ["sway-gate", "sway-gate-isolated"], f"unexpected hook ids: {ids}"
    system_hook = next(h for h in hooks if h["id"] == "sway-gate")
    assert system_hook["language"] == "system"
    assert system_hook["pass_filenames"] is False
    assert system_hook["entry"] == "sway gate"


def _build_random_lora_adapter(base_dir: Path, out_dir: Path) -> None:
    """Same deterministic LoRA build the other integration tests use.

    SmolLM2-135M + ``torch.manual_seed(0)`` + init-scale 0.05 lora_B.
    Produces a real, tiny adapter the hook can actually gate against.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(str(base_dir), torch_dtype=torch.float32)
    cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base, cfg)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn_like(param) * 0.05)
    peft_model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))


@pytest.fixture(scope="module")
def hook_adapter(
    tiny_model_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    adapter_dir = tmp_path_factory.mktemp("precommit-hook-adapter")
    _build_random_lora_adapter(tiny_model_dir, adapter_dir)
    return adapter_dir


def _write_spec(path: Path, *, base_dir: Path, adapter_dir: Path, assert_mean_gte: float) -> None:
    """Write a minimal 1-probe spec at ``path``. Adjust
    ``assert_mean_gte`` between 0.0 (pass) and 100.0 (fail)."""
    spec = {
        "version": 1,
        "models": {
            "base": {
                "base": str(base_dir),
                "kind": "hf",
                "adapter": str(adapter_dir),
                "dtype": "fp32",
                "device": "cpu",
            },
            "ft": {
                "base": str(base_dir),
                "kind": "hf",
                "adapter": str(adapter_dir),
                "dtype": "fp32",
                "device": "cpu",
            },
        },
        "defaults": {"seed": 0, "coverage_threshold": 0.0, "differential": True},
        "suite": [
            {
                "name": "dk",
                "kind": "delta_kl",
                "prompts": ["The capital of France is", "Water boils at"],
                "divergence": "js",
                "assert_mean_gte": assert_mean_gte,
            }
        ],
    }
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")


def _write_config(config_path: Path, spec_rel: str) -> None:
    """Write a ``.pre-commit-config.yaml`` declaring ``sway-gate``
    inline via ``repo: local`` so pre-commit doesn't need to fetch
    anything. Mirrors the shape of the shipped ``.pre-commit-hooks.yaml``."""
    config = textwrap.dedent(
        f"""
        repos:
          - repo: local
            hooks:
              - id: sway-gate
                name: sway gate
                entry: sway gate
                language: system
                files: '.*'
                pass_filenames: false
                args: [{spec_rel}, --threshold=0.0]
        """
    ).strip()
    config_path.write_text(config + "\n", encoding="utf-8")


@pytest.fixture
def precommit_repo(
    tmp_path: Path, tiny_model_dir: Path, hook_adapter: Path
) -> Iterator[Path]:
    """Initialize a tmp git repo with a spec + pre-commit config.

    Yields the repo root. The spec file lives at
    ``<repo>/sway.yaml``; the config at ``<repo>/.pre-commit-config.yaml``.
    The caller chooses ``assert_mean_gte`` by rewriting the spec after
    the fixture yields — cheaper than re-initializing git per test.
    """
    subprocess.run(  # noqa: S603
        ["git", "init", "--quiet"], cwd=tmp_path, check=True
    )
    # Silence the "Please tell me who you are" commit warning on fresh
    # CI runners without a global git identity.
    for key, val in (
        ("user.email", "sway-ci@example.com"),
        ("user.name", "sway ci"),
        ("commit.gpgsign", "false"),
    ):
        subprocess.run(  # noqa: S603
            ["git", "config", key, val], cwd=tmp_path, check=True
        )
    # Default pass case — the caller rewrites the spec for fail cases.
    _write_spec(
        tmp_path / "sway.yaml",
        base_dir=tiny_model_dir,
        adapter_dir=hook_adapter,
        assert_mean_gte=0.0,
    )
    _write_config(tmp_path / ".pre-commit-config.yaml", "sway.yaml")
    # Stage everything so ``pre-commit run --all-files`` has content.
    subprocess.run(  # noqa: S603
        ["git", "add", "-A"], cwd=tmp_path, check=True
    )
    yield tmp_path
    # Clean up pre-commit's cache dir too so back-to-back test
    # invocations don't trip over stale ``language: system`` state.
    cache = tmp_path / ".cache"
    if cache.exists():
        shutil.rmtree(cache, ignore_errors=True)


def _run_hook(cwd: Path) -> subprocess.CompletedProcess[str]:
    """Invoke ``pre-commit run sway-gate --all-files`` from ``cwd``."""
    env = os.environ.copy()
    # Point pre-commit's cache at the test tmp dir so we don't
    # pollute the user's ``~/.cache/pre-commit``.
    env["PRE_COMMIT_HOME"] = str(cwd / ".pre-commit-cache")
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pre_commit", "run", "sway-gate", "--all-files"],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_hook_passes_when_gate_passes(precommit_repo: Path) -> None:
    """Pass case: ``assert_mean_gte=0.0`` — probe passes, hook exits 0."""
    result = _run_hook(precommit_repo)
    # pre-commit reports ``Passed`` / ``Failed`` for each hook.
    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"hook failed unexpectedly (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "sway gate" in combined.lower() or "passed" in combined.lower()


def test_hook_fails_when_gate_fails(
    precommit_repo: Path, tiny_model_dir: Path, hook_adapter: Path
) -> None:
    """Fail case: impossible ``assert_mean_gte`` — probe FAILs, hook
    exits non-zero with ``gate FAILED`` in the output."""
    # Overwrite the spec with an impossible threshold.
    _write_spec(
        precommit_repo / "sway.yaml",
        base_dir=tiny_model_dir,
        adapter_dir=hook_adapter,
        assert_mean_gte=100.0,
    )
    subprocess.run(  # noqa: S603
        ["git", "add", "-A"], cwd=precommit_repo, check=True
    )
    result = _run_hook(precommit_repo)
    combined = result.stdout + result.stderr
    assert result.returncode != 0, (
        f"hook should have failed on an impossible gate (got rc=0):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "gate FAILED" in combined, (
        f"expected the 'gate FAILED' banner in hook output:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
