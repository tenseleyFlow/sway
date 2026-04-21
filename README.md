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
```

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
implements `NullCalibratedBackend` (the HF backend does).

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
