# Changelog

## Unreleased

### Sprint 02 — Universal z-score calibration

Closes Audit 01 findings P02 (delivery), B2, C9.

- **`NullAdapterProbe` is now a per-kind calibration matrix**: iterates
  every downstream numeric kind in the suite (or an explicit
  `calibrate_kinds` list), runs a miniature version of each through the
  `NullCalibrationBackendProxy` for N seeds, and publishes
  `{kind: {mean, std, n}}` under `evidence["null_stats"]`. The old
  "publishes two keys only" implementation is gone (closes B2).
- **`NullCalibrationBackendProxy`** (`_null_proxy.py`): swaps
  `as_finetuned()` to yield `as_null_adapter(seed)` so every numeric
  probe's own math produces the "what does my metric look like when the
  fine-tune is structural noise?" distribution without a bespoke code
  path per probe.
- **`Probe.calibrate_spec(ctx)` classmethod** plus
  `SENTINEL_PROMPTS` / `SENTINEL_DOC` shared constants in
  `probes/base.py`. Each numeric probe overrides `calibrate_spec` to
  return a small, cheap spec for calibration; probes that can't be
  meaningfully calibrated (e.g. `adapter_revert` needs an embedder,
  `adapter_ablation` needs `as_scaled_adapter`, `prompt_collapse`
  can't fit an exponential decay to null noise) opt out by returning
  `None` and surface `(no calibration for <kind>)` in the report.
- **Shared `_zscore` helpers** (`probes/_zscore.py`):
  `z_score(raw, stats)`, `verdict_from_z(z, threshold)`,
  `score_from_z(z)`, `no_calibration_note(kind)`. Every numeric probe
  now flows through these — no bespoke `(raw - mean) / std` math left
  in any probe file. Includes the `MIN_STD = 1e-6` floor so a
  degenerate null distribution (e.g. a single-seed calibration) can't
  produce an infinite z-score (closes C9).
- **All 10 numeric probes thread `get_null_stats(ctx, self.kind)`**:
  `delta_kl`, `adapter_revert`, `prompt_collapse`,
  `section_internalization`, `paraphrase_invariance`,
  `preference_flip`, `style_fingerprint`, `calibration_drift`,
  `leakage`, `adapter_ablation`. Each has an `assert_z_gte: float = 3.0`
  that is *preferred* when stats exist. Lower-is-better probes
  (`adapter_revert`, `calibration_drift`, `leakage`) sign-flip the
  z internally so the shared `z >= threshold` PASS rule still reads as
  "significantly better than null". Two probes (`paraphrase_invariance`,
  `calibration_drift`) keep their intent-aware / compound thresholds
  as the no-calibration fallback.
- **On-disk null-stats cache** (`probes/_null_cache.py`,
  `backends/hf.py`): HF backend exposes `cache_identity()`;
  `NullAdapterProbe` hashes `(backend_identity, runs, init_scale,
  seed_base, top_k, kinds)` into a stable filename under
  `~/.dlm-sway/null-stats/<key>.json` (XDG-respecting). Cache is
  best-effort — a missing / malformed file rebuilds. Disable per-suite
  with `cache: false` in the spec, or globally with
  `SWAY_DISABLE_NULL_CACHE=1`. The dummy backend doesn't expose a
  cache identity, so tests never touch disk unless they opt in.
- **Report shows z-scores in every numeric probe row**: the terminal
  renderer already had a `z` column; the markdown renderer now does
  too. Rows that fell back to fixed thresholds carry
  `(no calibration for <kind>)` directly in the message.

### Sprint 01 — Finite safety & verdict integrity

Closes Audit 01 findings B1, B3 (fix), B4, B5, C3, C4, D1, D2.

- **`safe_finalize` helper** in `core/result.py`: every numeric probe now
  routes its final `ProbeResult` through this guard. A non-finite
  `raw` (or any explicitly-critical field) auto-converts to
  `Verdict.ERROR`; non-finite auxiliaries are nulled out and recorded
  in `evidence["defensively_nulled"]`.
- **`_divergence` non-finite rejection**: `aligned_probs`, `kl`, `js`
  now raise `ProbeError` on NaN/inf inputs instead of silently
  producing garbage. JS results are bounds-checked against the
  theoretical `[0, ln 2]`; out-of-bound results raise rather than
  ship. (Pins the +11639σ bug — JS exceeded ln 2 nats by 19×.)
- **`PreflightCheckable` Protocol**: HF and dummy backends gain
  `preflight_finite_check()` that runs one forward pass per view with
  a sentinel prompt and rejects backends whose logprobs are non-finite.
- **Suite runner preflight gate**: every `sway run` calls
  `backend.preflight_finite_check()` before the probe loop. Failure
  emits a single synthetic ERROR probe and skips all configured
  probes; `sway gate` exits non-zero. Disable with `skip_preflight=True`
  for sub-second test suites.
- **`style_fingerprint` zero-vector fix (B4)**: a fine-tuned model that
  produces empty / whitespace-only generations no longer reports a
  spurious "+0.82 shift toward doc" PASS. The `_cosine_shift` math is
  replaced by `_projection_shift = (ft-base)·(doc-base) / ||doc-base||²`
  which goes to zero when ft equals base.
- **`_saturation_lambda` full-range search (B3)**: the adapter-ablation
  saturation detector now searches the entire λ range (not just λ ≤ 1.0)
  with `max(divs)` as the reference, and returns a typed reason
  (`"found"` / `"non_monotonic"` / `"flat_curve"` / `"below_floor"`)
  so a flat NaN-adapter curve is distinguishable from a healthy
  saturating one.
- **Property-based divergence tests** (`hypothesis`): symmetry,
  `JS(p,p) = 0`, `JS ≤ ln 2`, `KL(p,p) = 0`, KL non-negativity. Plus
  explicit non-finite-raises tests pinning every entry point.
- **End-to-end NaN regression** (`tests/integration/test_nan_adapter_regression.py`,
  slow+online): builds a real PEFT adapter with all-NaN weights via
  PEFT, persists it through safetensors, then asserts the HF backend's
  preflight rejects it and the suite produces ERROR — not the
  +11639σ headline the audit caught.
- Added `huggingface_hub` to dev dep group so integration tests can
  resolve the `tiny_model_dir` fixture (closes C1 ahead of Sprint 04).

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
