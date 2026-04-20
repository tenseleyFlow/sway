# Changelog

## 0.1.0.dev0 — 2026-04-20

Initial pre-alpha. Full 11-primitive battery shipped.

### Primitives

- **Adherence**
  - `delta_kl` — mean JS/KL divergence between base and fine-tuned next-token distributions
  - `adapter_revert` — reversion under adversarial paraphrase (needs `sway-eval[semsim]`)
  - `prompt_collapse` — exponential-decay fit of divergence over context length
- **Attribution**
  - `section_internalization` *(flagship)* — per-section `effective_sis` with leak check
  - `paraphrase_invariance` — memorization vs. generalization, intent-aware
  - `preference_flip` — DPO/ORPO chosen/rejected margin inversion
- **Calibration**
  - `style_fingerprint` — 6-dim numpy-only stylistic shift vs. document
  - `calibration_drift` — general-knowledge regression on a packaged 30-item pack
  - `leakage` — greedy LCS recall + perturbation fragility
- **Ablation**
  - `adapter_ablation` *(signature primitive)* — λ-scaled divergence curve with linearity, saturation, overshoot metrics
- **Baseline**
  - `null_adapter` — stats scaffolding for z-score calibration (implementation pending)

### Infrastructure

- `DifferentialBackend` + `ScalableDifferentialBackend` protocols
- HuggingFace + PEFT backend with `disable_adapter` / `set_adapter` toggling and LoRA-scale mutation
- Dummy backend for unit tests (canned responses + linear-blend scalable mode)
- YAML spec loader, composite score (four-category weighted), rich terminal + JSON + JUnit + Markdown reports
- Typer CLI: `run`, `gate`, `check`, `diff`, `autogen`, `doctor`, `report`
- `.dlm` bridge (`dlm-sway[dlm]`): resolver + full-battery autogen
- Matplotlib visualizations (`dlm-sway[viz]`): SIS bar chart, ablation curve, KL histogram

### Known gaps

- Null-adapter baseline is scaffolded but its HF-level materialization (building random-init LoRAs at matched rank) is not yet wired — probes fall back to fixed thresholds until the next milestone.
- Custom backend entry-point dispatch (`kind: custom`) is stubbed but not implemented.
- MLX backend is registered as a future-milestone target; all MLX paths raise `BackendNotAvailableError`.
- PyPI publication of the `dlm-sway` wheel is pending a clean CI release workflow.
