# dlm-sway

Differential testing for fine-tuned causal language models.

**One question:** *did LoRA/QLoRA training actually change model behavior
in a meaningful way, or is the model just defaulting to the pretrained
base?*

`dlm-sway` gives you a trustworthy, reproducible answer with eleven
purpose-built primitives, each z-scored against a null-adapter baseline.
No LLM judges. No external APIs. Deterministic on CPU where possible.

## Install

```bash
pip install "dlm-sway[hf]"                # HuggingFace + PEFT backend
pip install "dlm-sway[hf,style,semsim]"   # full primitive battery
pip install "dlm-sway[all]"               # everything including optional viz
pip install "dlm-sway[dlm]"               # auto-generate tests from a .dlm file
```

## 90-second smoke test

```bash
dlm-sway check path/to/adapter --base HuggingFaceTB/SmolLM2-135M-Instruct
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
  - {name: knows_concept, kind: dir,
     prompt: "The Dunning-Kruger effect describes",
     target: " a cognitive bias where",
     distractor: " a programming language"}
  - {name: no_reversion, kind: adapter_revert, paraphrases: 4}
  - {name: section_attribution, kind: section_internalization}
```

```bash
dlm-sway run sway.yaml              # full report to terminal + JSON
dlm-sway gate sway.yaml --junit     # CI-friendly; non-zero on fail
```

## Why it exists

Standard benchmarks (MMLU, HellaSwag) ask *"how good is this model?"*
That's the wrong question after a targeted LoRA fine-tune on a small
user-authored document. The right question is *"did the adapter actually
move the model toward what I wrote?"* — and existing tools answer this
poorly.

`dlm-sway` answers it directly via eleven primitives across four
categories:

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

## The `.dlm` integration

If you trained your adapter via the [DocumentLanguageModel
project](https://github.com/tenseleyFlow/DocumentLanguageModel), sway
can auto-generate a test suite from your document's sections:

```bash
pip install "dlm-sway[hf,dlm]"
dlm-sway autogen path/to/doc.dlm -o sway.yaml
dlm-sway run sway.yaml
```

Per-section attribution tells you *which* parts of your document
actually moved the model — a kind of signal no other tool provides.

## Status

Pre-alpha. API will break. Version `0.1.0` is the first tag.

## License

MIT
