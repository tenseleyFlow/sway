# sway

Differential testing for fine-tuned causal language models.

> **⚠️ Pre-alpha — not on PyPI yet.** This project is in active
> development and has **not** been published to PyPI. `pip install dlm-sway`
> will not work. The wheel name is reserved for when publication happens;
> see [Install](#install-from-source) for how to use sway today.

**One question:** *did LoRA/QLoRA training actually change model behavior
in a meaningful way, or is the model just defaulting to the pretrained
base?*

`sway` gives you a trustworthy, reproducible answer with eleven
purpose-built primitives, each z-scored against a null-adapter baseline.
No LLM judges. No external APIs. Deterministic on CPU where possible.

> **Naming convention (for when PyPI goes live).** The source repo and
> CLI entry point are both `sway`. The PyPI wheel will be `dlm-sway`
> because the short `sway` name is already taken on PyPI by an unrelated
> project. The CLI installed by `pip install dlm-sway` will be
> `sway` — mismatched wheel/command names are a PyPA convention (see
> `pyyaml` → `import yaml`).

## Install from source

Until the wheel lands on PyPI, install editable from a clone:

```bash
git clone https://github.com/tenseleyFlow/sway.git
cd sway

# Create a venv and install with the HF backend extras
uv venv --python 3.11 .venv      # or: python -m venv .venv
source .venv/bin/activate
uv pip install -e ".[hf]" --group dev
```

Available extras:

- `[hf]` — HuggingFace + PEFT backend (required for real models)
- `[mlx]` — Apple Silicon MLX backend (darwin-arm64 only)
- `[style]` — stylistic fingerprint extensions (spaCy + textstat + nlpaug)
- `[semsim]` — sentence-transformers for the revert probe
- `[dlm]` — auto-generate suites from `.dlm` documents
- `[viz]` — matplotlib plots
- `[all]` — everything

Verify the install:

```bash
sway --version
sway doctor
```

## Planned PyPI install (not live yet)

Once the wheel ships the install story will be the single-line flavor:

```bash
# ⚠️ NOT FUNCTIONAL YET — post-PyPI-publish only
pip install "dlm-sway[hf]"
pip install "dlm-sway[hf,style,semsim]"
pip install "dlm-sway[all]"
pip install "dlm-sway[dlm]"
```

Watch the repo's Releases for the first published tag.

## 90-second smoke test

```bash
sway check path/to/adapter --base HuggingFaceTB/SmolLM2-135M-Instruct
```

Outputs a verdict in under a minute on CPU for small models: *your
adapter is 4.2σ above noise* ✅ or *indistinguishable from a null
adapter* ❌.

## Full suite

```yaml
# sway.yaml
version: 1
models:
  base: {kind: hf, base: "HuggingFaceTB/SmolLM2-135M-Instruct"}
  ft:   {kind: hf, base: "HuggingFaceTB/SmolLM2-135M-Instruct",
         adapter: "./runs/adapter/v0003"}
suite:
  - {name: null_baseline,       kind: null_adapter, runs: 3}
  - {name: doc_divergence,      kind: delta_kl,
     prompts: ["The key insight is", "An important rule"]}
  - {name: section_attribution, kind: section_internalization}
  - {name: no_leakage,          kind: leakage}
  - {name: ablation_shape,      kind: adapter_ablation,
     prompts: ["Tell me more about"]}
```

```bash
sway run sway.yaml              # full report to terminal + JSON
sway gate sway.yaml --junit     # CI-friendly; non-zero on fail

# Override the composite weights on the command line (partial overrides
# are fine — unspecified categories keep their defaults):
sway run sway.yaml --weights "attribution=0.5,adherence=0.2"
```

Inside `sway.yaml`, tuning knobs in `defaults` include:

- `seed` — passed to `seed_everything` before any probe runs.
- `differential` (default `true`) — toggle between the single-load PEFT
  path and a two-model load (doubled memory, rarely needed; for custom
  backends that can't do in-place adapter toggling).
- `score_weights` — per-category weight overrides baked into the spec so
  CI runs reproduce the same score without a CLI flag.

## Why it exists

Standard benchmarks (MMLU, HellaSwag) ask *"how good is this model?"*
That's the wrong question after a targeted LoRA fine-tune on a small
user-authored document. The right question is *"did the adapter actually
move the model toward what I wrote?"* — and existing tools answer this
poorly.

`sway` answers it directly via eleven primitives across four categories:

| Category      | Primitives                                            |
|---------------|-------------------------------------------------------|
| Adherence     | `delta_kl`, `adapter_revert`, `prompt_collapse`       |
| Attribution   | `section_internalization`, `paraphrase_invariance`, `preference_flip` |
| Calibration   | `style_fingerprint`, `calibration_drift`, `leakage`   |
| Ablation      | `adapter_ablation` ← the signature primitive          |

**The signature primitive.** `adapter_ablation` scales the LoRA additive
term by λ ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.25} and measures the divergence
curve. A healthy fine-tune shows a smooth, monotonic, non-saturated
response. A degenerate one shows a step function or an overshoot-then-
crash. Nobody else does this because nobody else gets this close to the
adapter math.

**The calibration.** Every numeric probe z-scores its raw metric against
a null-adapter baseline — a same-structure LoRA with random-init weights.
"Your adapter's KL is 4.2σ above noise" is a far stronger claim than a
fixed threshold. The null-adapter calibration requires a backend that
implements `NullCalibratedBackend` (the HF backend does); probes that
can't be calibrated (e.g., `adapter_revert` needs an embedder, the null
proxy doesn't have one) surface `(no calibration)` in the report and
fall back to fixed thresholds. Calibration stats are cached on disk
under `~/.dlm-sway/null-stats/` keyed by backend identity.

**The rank profile.** `null_adapter` takes an optional
`rank_multipliers: list[float]` (default `[1.0]`). Pass
`[0.5, 1.0, 2.0]` and every numeric probe carries a three-point
z-score curve: `z=+4.2σ @ 1x / +6.8σ @ 0.5x / +2.1σ @ 2x`. The shape
is diagnostic:

- **Flat or slightly rising toward 0.5x** — adapter signal is
  rank-stable, roughly independent of noise energy.
- **Sharply higher at 0.5x, lower at 2x** — adapter is rank-saturated:
  a smaller rank would have yielded a clearer separation from noise.
  Consider halving `r`.
- **Low everywhere** — adapter is barely above noise at any rank;
  the signal is real but weak.

Caveat: high z at low rank can also mean the low-rank null is
*pathologically quiet* rather than that the adapter is strong. Read the
profile as a shape, not a scalar — if all three z's move proportionally,
the adapter is doing work; if they spread apart, the rank is mis-sized.

Implementation note: rank scaling is mathematically equivalent to
multiplying the null noise std by `sqrt(rank_scale)` (LoRA's A·B output
variance scales linearly with rank). The shipped backends apply that
scaling rather than reshaping PEFT tensors — no model reload, no
rank-specific adapter cache, same `alpha/r` scaling throughout.

**Determinism.** Every `sway run` calls `seed_everything(spec.defaults.seed)`
before the first probe — seeds python/numpy/torch RNGs and asks torch
for deterministic algorithms (`CUBLAS_WORKSPACE_CONFIG=:4096:8`). The
report footer prints the achieved class — `strict` (CUDA), `best_effort`
(CPU/MPS), or `loose` (deterministic algorithms refused). Same seed +
same host = bit-identical scoring across runs.

## Pytest integration

For teams already testing their training pipeline with pytest, sway
ships a plugin behind the `[pytest]` extra. A single decorator turns
one pytest function into one test item per probe plus an optional
composite-score gate:

```python
import pytest

@pytest.mark.sway(spec="sway.yaml", threshold=0.6)
def test_adapter_healthy() -> None:
    """The decorator owns the body — a bare pass is conventional."""
```

`pytest -v` then reports:

```
test_sway_gate.py::test_adapter_healthy::adherence    PASSED
test_sway_gate.py::test_adapter_healthy::calibration  PASSED
test_sway_gate.py::test_adapter_healthy::__gate__     PASSED
```

`--junitxml` emits one `<testcase>` per probe, `pytest -k adherence`
runs just that probe, `FAIL` / `ERROR` / `SKIP` verdicts translate to
pytest outcomes. See `examples/pytest_integration/` for a full
before/after walkthrough.

```bash
pip install 'dlm-sway[hf,pytest]'
```

## The `.dlm` integration

If you trained your adapter via the [DocumentLanguageModel
project](https://github.com/tenseleyFlow/DocumentLanguageModel), `sway`
auto-generates a test suite from your document's sections.

Install sway with the `[dlm]` extra alongside `[hf]` (pre-PyPI, editable):

```bash
# inside a clone of this repo
uv pip install -e ".[hf,dlm]"
```

Then:

```bash
sway autogen path/to/doc.dlm -o sway.yaml
sway run sway.yaml
```

Per-section attribution tells you *which* parts of your document
actually moved the model — a kind of signal no other tool provides.

## Status

Pre-alpha. API will break. Not yet on PyPI — install editable from source
(see [Install from source](#install-from-source)). Version `0.1.0` will be
the first published tag; until then, every clone pulls the tip of `main`.

## License

MIT
