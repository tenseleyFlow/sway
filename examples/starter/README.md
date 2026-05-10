# Starter sway spec

This is the recipe Audit 13 recommends as a new user's first sway
spec. The same shape works for both `sway run` (interactive verdict
in the terminal) and `sway gate` (CI-friendly, exit non-zero on fail).

## Install with the bridge

```bash
pip install 'dlm-sway[hf,dlm]'
```

The `[dlm]` extra activates three probes that opt-out (SKIP) without
the bridge:

- **`section_internalization`** — per-section attribution. Tells you
  *which paragraphs of your `.dlm` actually landed*.
- **`leakage`** — verbatim recital + fragility under perturbations.
  Distinguishes pattern-match (healthy) from rote memorization (often
  what fails downstream generation).
- **`paraphrase_invariance`** — separates memorization from
  generalization. The load-bearing probe for *"did Q/A binding work?"*
  — if you trained on instruction sections and this fails, the model
  shifted distributions but didn't bind questions to answers.

## Run it

```bash
# Edit `dlm_source:` to point at your .dlm, then:
sway run sway.yaml          # full report to terminal + JSON
sway gate sway.yaml --junit # CI gate, exits non-zero on fail
```

## Why every numeric probe needs `null_adapter`

A `delta_kl` of 0.08 sounds small. Without a noise baseline, you
can't tell if it's nothing or a real effect. With `null_adapter`
calibration in the same spec, the report shows `0.08 (z=4.2σ)` —
suddenly that's a clear signal vs the noise floor.

The Audit 13 fortran fine-tune got `dk_fortran z=+44.18σ` (massive
distribution shift) but `paraphrase_invariance z=−3.51σ` (Q/A binding
*regressed*). Both were calibrated against null_adapter; the contrast
is what made the diagnosis ("the adapter learned the language but
didn't learn the meta-task") possible.

## Tuning knobs

- `coverage_threshold` — gate exit-code threshold. `0.6` is a sane
  default for active development; `0.8` is a tight CI gate.
- `score_weights` — per-category composite weights. Tilt toward
  attribution if you care about per-section signal; toward calibration
  if your worry is regression of general competence.
- `assert: { z_gte: ..., r_squared_gte: ..., ... }` — per-probe pass
  thresholds. The values in this spec are conservative starting points;
  raise them as your training matures.

## When you don't have a `.dlm`

If you're testing a hand-trained PEFT adapter without dlm in the loop:

1. Drop the `dlm_source:` line.
2. Replace `section_internalization`, `leakage`, and
   `paraphrase_invariance` with `delta_kl` / `prompt_collapse` /
   `style_fingerprint` (or whatever fits the corpus you can produce
   manually — see `examples/pytest_integration/sway.yaml` for a
   leaner spec).

## Related

- `examples/pytest_integration/sway.yaml` — minimal spec for the
  `@pytest.mark.sway` marker.
- `examples/precommit-example/sway.yaml` — pre-commit gate template.
- The full README has the rationale for each probe under
  *"Why it exists"*.
