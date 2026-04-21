"""Cross-platform determinism golden — S18 / stretch-list F-item.

Runs a minimal 2-probe suite against a deterministically-seeded LoRA
adapter on SmolLM2-135M, then compares the JSON output against a
platform-pinned golden file (``tests/golden/expected_<platform>.json``).

Marked ``slow+online``: needs network for the tiny_model fixture and
HF weights for the adapter build. Runs in a dedicated CI matrix
(ubuntu-latest + macos-latest) via ``.github/workflows/ci.yml``'s
``determinism-golden`` job.

Determinism contract this test pins:

- **Within a platform**, two runs of the same spec + adapter produce
  byte-identical JSON (after masking timestamps + wall time). The
  existing ``seed_everything`` wiring already holds this — this test
  just encodes the check.
- **Across platforms**, numeric drift is bounded: raw metrics within
  1e-6, scores within 1e-4. Looser than bitwise (BLAS implementation
  differences make bitwise impossible) but tight enough that a silent
  algorithm change — say a ``top_k=256`` default flipped to 128 —
  surfaces as a clear drift report on both legs.

Regeneration recipes:

- **Locally**: ``SWAY_UPDATE_GOLDENS=1 uv run pytest tests/integration/
  test_determinism_golden.py -m "slow or online"`` writes the
  current-platform golden to ``tests/golden/expected_<platform>.json``.
- **Via CI**: dispatch the ``determinism-golden`` workflow with
  ``regenerate_goldens=true``; download the uploaded artifact; commit
  the platform file to the branch.

First-time Linux runs SKIP with a clear regen-recipe message when the
``expected_linux.json`` file is missing — generated via the CI recipe
above and committed as a follow-up to the opening PR.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from dlm_sway.backends.hf import HuggingFaceDifferentialBackend
from dlm_sway.core.golden import compare_goldens, mask_variable_fields
from dlm_sway.core.model import ModelSpec
from dlm_sway.suite import report
from dlm_sway.suite.runner import run as run_suite
from dlm_sway.suite.score import compute as compute_score
from dlm_sway.suite.spec import SwaySpec

pytestmark = [
    pytest.mark.slow,
    pytest.mark.online,
    # F03 (Audit 03) — macOS CI observed a 20m stall inside
    # ``snapshot_download`` on a run that normally completes in
    # ~1m. Hard cap at 10m so a silent network hang fails as a
    # test (actionable error in the CI log) rather than a
    # workflow timeout (zero output).
    pytest.mark.timeout(600),
]


GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden"
GOLDEN_SPEC_PATH = GOLDEN_DIR / "spec.yaml"


def _platform_tag() -> str:
    """Map ``sys.platform`` to the golden filename suffix.

    ``darwin`` → ``darwin``; ``linux`` → ``linux``. Other platforms
    (windows, freebsd) skip the test in the caller below; the tag
    here still returns something usable for diagnostic messages.
    """
    if sys.platform.startswith("darwin"):
        return "darwin"
    if sys.platform.startswith("linux"):
        return "linux"
    return sys.platform


def _golden_path() -> Path:
    return GOLDEN_DIR / f"expected_{_platform_tag()}.json"


def _build_deterministic_lora_adapter(base_dir: Path, out_dir: Path) -> None:
    """Build a LoRA adapter deterministically from a fixed seed.

    The goal is "bit-identical adapter given the same torch version".
    ``torch.manual_seed(0)`` + a fixed init scale achieves that; any
    drift in the ranker's per-ULP output downstream is caught by the
    golden's tolerance.

    Same seed + init shape as ``test_external_perplexity_e2e``'s
    fixture — intentionally reused so the two integration tests
    stress the same code path.
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
def golden_adapter(tiny_model_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    adapter_dir = tmp_path_factory.mktemp("golden-adapter")
    _build_deterministic_lora_adapter(tiny_model_dir, adapter_dir)
    return adapter_dir


@pytest.fixture(scope="module")
def golden_backend(
    tiny_model_dir: Path, golden_adapter: Path
) -> Iterator[HuggingFaceDifferentialBackend]:
    backend = HuggingFaceDifferentialBackend(
        base_spec=ModelSpec(base=str(tiny_model_dir), kind="hf", dtype="fp32", device="cpu"),
        adapter_path=golden_adapter,
    )
    yield backend
    backend.close()


def _run_golden_suite(backend: HuggingFaceDifferentialBackend) -> dict[str, object]:
    """Load the golden spec, run it, return the JSON payload as a dict."""
    import tempfile

    import yaml

    from dlm_sway.suite.loader import load_spec

    # The checked-in spec has placeholder model paths; substitute the
    # real ones loaded by the fixture. ``load_spec`` takes a file path;
    # write the patched payload to a tmp file and load that instead.

    with GOLDEN_SPEC_PATH.open("r", encoding="utf-8") as f:
        spec_payload = yaml.safe_load(f)

    # Paths don't actually matter to the runner once the backend is
    # built — the backend is passed in directly, not reconstructed
    # from the spec. But ``load_spec`` + ``SwaySpec.model_validate``
    # parse and type-check, so we still need a valid-looking spec.
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(spec_payload, tmp)
        tmp_path = Path(tmp.name)

    spec: SwaySpec = load_spec(tmp_path)
    result = run_suite(spec, backend)
    score = compute_score(result, weights=None)
    payload = json.loads(report.to_json(result, score))
    assert isinstance(payload, dict)
    return payload


def _update_golden(golden_path: Path, payload: dict[str, object]) -> None:
    """Write ``payload`` (masked) to ``golden_path`` and emit a
    human-readable diagnostic so the CI log makes the recipe obvious."""
    masked = mask_variable_fields(payload)
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    golden_path.write_text(json.dumps(masked, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stderr.write(f"[determinism-golden] wrote {golden_path}\n")


def test_suite_output_matches_platform_golden(
    golden_backend: HuggingFaceDifferentialBackend,
) -> None:
    """The cross-platform determinism test.

    Two execution modes:

    - ``SWAY_UPDATE_GOLDENS=1``: writes the current run's output to
      ``tests/golden/expected_<platform>.json`` and asserts nothing.
      Use this locally or from the ``determinism-golden`` CI workflow's
      regenerate mode.
    - Default: masks variable fields, loads the platform golden, and
      asserts ``compare_goldens`` finds no diffs. Missing golden →
      SKIP with a regen recipe.
    """
    payload = _run_golden_suite(golden_backend)
    golden_path = _golden_path()

    if os.environ.get("SWAY_UPDATE_GOLDENS") == "1":
        _update_golden(golden_path, payload)
        pytest.skip(f"wrote golden → {golden_path}; re-run without SWAY_UPDATE_GOLDENS")

    if not golden_path.exists():
        pytest.skip(
            f"no golden for {_platform_tag()!r} at {golden_path}. "
            "Generate it by (a) running locally with SWAY_UPDATE_GOLDENS=1, or "
            "(b) dispatching the ``determinism-golden`` CI workflow with "
            "``regenerate_goldens=true`` and committing the uploaded artifact."
        )

    expected = json.loads(golden_path.read_text(encoding="utf-8"))
    actual = mask_variable_fields(payload)
    diffs = compare_goldens(actual, expected)
    if diffs:
        formatted = "\n".join(f"  - {d}" for d in diffs[:20])
        extra = f"\n  ...and {len(diffs) - 20} more" if len(diffs) > 20 else ""
        pytest.fail(
            f"{len(diffs)} golden drift(s) on {_platform_tag()}:\n{formatted}{extra}\n"
            "If the drift is a deliberate algorithm change, regenerate the "
            "golden via SWAY_UPDATE_GOLDENS=1 and commit the new file."
        )
