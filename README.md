# sway

Differential testing for fine-tuned causal language models.

> **Alpha — v0.1.0 on PyPI.** API is not stable; semantic versioning
> applies only from v1.0 onward. Feedback + issues welcome.

**One question:** *did LoRA/QLoRA training actually change model behavior
in a meaningful way, or is the model just defaulting to the pretrained
base?*

`sway` gives you a trustworthy, reproducible answer with thirteen
purpose-built primitives, each z-scored against a null-adapter baseline.
No LLM judges. No external APIs. Deterministic on CPU where possible.

> **Naming convention.** The source repo and CLI entry point are both
> `sway`. The PyPI wheel is `dlm-sway` because the short `sway` name is
> taken on PyPI by an unrelated project. The CLI installed by
> `pip install dlm-sway` is `sway` — mismatched wheel/command names are
> a PyPA convention (see `pyyaml` → `import yaml`).

## Install

```bash
# HF + PEFT backend — required for real models
pip install "dlm-sway[hf]"

# Extras composable as usual
pip install "dlm-sway[hf,style,semsim]"
pip install "dlm-sway[all]"

# .dlm auto-suite generation (requires the DLM sibling project)
pip install "dlm-sway[dlm]"
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

## MLX backend (Apple Silicon)

Install the extra and point sway at any PEFT adapter — the MLX backend
auto-converts on first load and caches the result under
`~/.cache/dlm-sway/mlx-converted/<sha>/`. No manual `.npz` step.

```bash
pip install "dlm-sway[mlx]"

# Either: let sway auto-convert when it loads the adapter
sway run sway.yaml   # spec sets models.ft.kind: mlx, points at any PEFT dir

# Or: convert explicitly (useful for inspection / scripting)
sway convert-adapter ~/path/to/peft-adapter ~/path/to/mlx-adapter
```

Conversion is one-shot — repeated `sway run` invocations on the same
adapter version short-circuit on a content hash. Supports standard
LoRA on `q_proj` / `v_proj` / etc. QLoRA 4-bit and full-weight
`modules_to_save` overrides aren't supported (the converter prints a
clear warning if it sees them).

## Install from source

For the development HEAD (unreleased changes, contributor workflow):

```bash
git clone https://github.com/tenseleyFlow/sway.git
cd sway

uv venv --python 3.11 .venv      # or: python -m venv .venv
source .venv/bin/activate
uv pip install -e ".[hf]" --group dev
```

## 90-second smoke test

```bash
sway check path/to/adapter --base HuggingFaceTB/SmolLM2-135M-Instruct
```

Outputs a verdict in under a minute on CPU for small models: *your
adapter is 4.2σ above noise* ✅ or *indistinguishable from a null
adapter* ❌.

## Pre-run diagnostics

If your adapter was trained with [DocumentLanguageModel](https://github.com/tenseleyFlow/DocumentLanguageModel),
`sway check` runs the **`gradient_ghost`** probe first — a ~50 ms
disk-only inspection of dlm's `training_state.pt`. It catches
adapters that are obviously broken (e.g. left at `--max-steps 5`
from a smoke test) **before** the model ever loads, with a banner
on top of the regular verdict:

```
⚠️  PRE-RUN ALERT — gradient_ghost flagged severe undertraining
   severely undertrained: global_step=2 < threshold 50.
   The probe scores below may be unreliable. Consider retraining.
```

The probe's signal ladder, in order of decisiveness:

1. `global_step` below a configurable threshold (default 50) → FAIL.
2. Every per-param `exp_avg_sq` is NaN → FAIL.
3. More than 30% of LoRA layers show >2× the lowest layer's
   gradient variance → FAIL or WARN.
4. Otherwise → PASS.

Specs that contain *only* pre-run diagnostics like `gradient_ghost`
skip backend construction entirely — `sway run` against a pre-flight
spec doesn't load a model, doesn't allocate GPU memory, doesn't pay
the cold-start tax. Useful for `pre-commit` gates that should fast-
path adapter sanity before authorizing the full battery.

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

`sway` answers it directly via thirteen primitives across four
categories, plus a baseline-calibration primitive:

| Category      | Primitives                                            |
|---------------|-------------------------------------------------------|
| Adherence     | `delta_kl`, `adapter_revert`, `prompt_collapse`, `cluster_kl` |
| Attribution   | `section_internalization`, `paraphrase_invariance`, `preference_flip` |
| Calibration   | `style_fingerprint`, `calibration_drift`, `leakage`, `external_perplexity` |
| Ablation      | `adapter_ablation` ← the signature primitive          |
| Baseline      | `null_adapter` (powers every z-score in the report)   |

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

## Pre-commit

For teams using [pre-commit.com](https://pre-commit.com), sway ships
a `.pre-commit-hooks.yaml` declaring three hooks that run `sway gate`
before every commit touching a spec, `.dlm` document, or adapter
file. Add 4–5 lines to your `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/tenseleyFlow/sway
    rev: v0.1.0
    hooks:
      - id: sway-gate
        args: ["sway.yaml", "--threshold=0.6"]
```

Three variants ship; pick whichever fits your install posture:

| Hook | When to use | First-run cost |
|---|---|---|
| `sway-gate` | you already ran `pip install 'dlm-sway[hf]'` | ~none — uses the sway binary on your `PATH` |
| `sway-gate-isolated` | fresh venv, no existing sway install | ~2 min + ~5 GB — pre-commit builds a fresh venv and installs sway + torch + transformers |
| `sway-gate-docker` | zero-install hosts with docker available | ~1 min — pulls `ghcr.io/tenseleyflow/sway-gate:v0.1.0` (torch baked in, MiniLM weights pre-cached) |

The recommended default is `sway-gate`. Switch to
`sway-gate-isolated` if you can't rely on a host-level sway install.
Reach for `sway-gate-docker` on ephemeral CI runners where docker is
cheaper than a fresh venv.

### Rev pinning

The example above pins to the `v0.1.0` tag. Bump it deliberately
when you want to pick up a new release; `pre-commit autoupdate` will
surface newer tags when you run it explicitly.

### Scope

The hook **only gates** — exits non-zero on FAIL, zero on PASS. No
`--json` / `--markdown` report flags are surfaced; those belong in
`sway run` (ad-hoc or in a separate CI job). Keeps `git commit` fast
and the gate's verdict uncluttered.

See [`examples/precommit-example/`](examples/precommit-example/) for
the full walk-through including the `sway.yaml` template, the
consumer-side `.pre-commit-config.yaml`, and the
try-it-locally-before-you-install recipe.

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

> Heads up — once dlm publishes its sister change, `dlm export --to
> sway-json` writes a ready-to-run `sway.yaml` next to the GGUF.
> Skip the manual `sway autogen` step entirely.

## Tool-use fidelity

Adapters that target tool-calling behavior have a failure mode no
other probe catches: they pass every adherence / attribution probe
but silently degrade the base model's JSON-schema compliance, or
start hallucinating tool names that aren't in the declared surface.
The `tool_use_fidelity` probe scores three independent signals on
each `(prompt, tool_spec)` case:

- **JSON-schema validity delta** — `ft_valid_rate − base_valid_rate`.
  A negative value means the adapter regressed on producing
  schema-valid calls; the default pass criterion tolerates a 5pp
  drop.
- **Tool-name hallucination rate** — over schema-valid ft calls, the
  fraction that pick a tool outside the declared `allowed_tools`
  surface (or differ from `gold_tool_name` when no surface is set).
  Default cap 10%.
- **Argument-field disagreement rate** — over cases where both
  views produce schema-valid calls, the per-leaf-field disagreement
  rate between base and ft arguments. Surfaced as evidence — the
  v1 surface doesn't gate on it.

```yaml
suite:
  - name: tool_calls_intact
    kind: tool_use_fidelity
    cases:
      - prompt: "Search the web for the capital of France."
        tool_spec:
          name: search_web
          parameters:
            type: object
            properties: {query: {type: string}}
            required: [query]
        gold_tool_name: search_web
    allowed_tools: [search_web, calculator, http_fetch]
```

The probe uses an OpenAI-style function-calling schema (the most
common shape across hosted and OSS tool-use stacks). Other schema
flavors (Anthropic, Llama-3.1, Gemini) land in a follow-up sprint;
the v1 implementation deliberately ships the format that covers
the largest user surface.

`json_valid_rate_ft` is z-scored against the null-adapter baseline
when `null_adapter` is in the suite — the principled "the adapter
preserved the base's tool-call structure beyond noise" signal.

## Reproducing a sway run

Sometimes you want a coworker (or a future-you, or a bug report) to
re-execute exactly what you ran without recreating your environment.
`sway pack` bundles the spec + the source `.dlm` document + your
locally-cached null-stats + (optionally) a known-good report into a
single `*.swaypack.tar.gz` you can email or commit:

```bash
sway pack sway.yaml -o audit-2026-04.swaypack.tar.gz \
  --include-golden last-run.json
# wrote audit-2026-04.swaypack.tar.gz (4.2 KB)  sections=312b  null_stats=0  golden=yes

# share the tarball with a coworker
sway unpack audit-2026-04.swaypack.tar.gz
# extracted: <cwd>/swaypack
#   spec_path: <cwd>/swaypack/sway.yaml
#   ...
# To run the bundled spec:
#   SWAY_NULL_CACHE_DIR=<cwd>/swaypack/null-stats sway run <cwd>/swaypack/sway.yaml
```

The packed null-stats cache means the consumer doesn't pay the
re-calibration cost (~120 forward passes on a typical 10-probe
suite). The unpack output prints the exact `SWAY_NULL_CACHE_DIR=...`
env var to use; that env redirects null-stats lookups at the bundled
cache instead of `~/.dlm-sway/`.

A bundled golden JSON makes it trivial for the consumer to verify
their re-run matches yours. Defaults to a 50 MB pack-size cap;
override with `--max-size-mb`.

## Status

Pre-alpha. API will break. Not yet on PyPI — install editable from source
(see [Install from source](#install-from-source)). Version `0.1.0` will be
the first published tag; until then, every clone pulls the tip of `main`.

## License

MIT
