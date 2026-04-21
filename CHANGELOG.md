# Changelog

## Unreleased

### Sprint 06 — CLI & report UX polish

Closes Audit 01 findings D3, D4, D5, D6, D7, D8, D9, D10, D11, D12,
D13, D14, D15, B16, B17, B18.

- **D3 — extras rollup footer:** terminal + markdown reports now end
  with a single `pip install 'dlm-sway[...]'` line collecting every
  extra mentioned in SKIP messages. Single rollup, no per-row scan.
- **D4 — `sway check` infers `--base`:** reads
  `base_model_name_or_path` from the adapter's `adapter_config.json`
  when `--base` is omitted; echoes "(inferred base model: …)" so the
  user knows what got picked. Falls back to a clear required-flag
  error when the field is missing.
- **D5 — `sway autogen` annotates the YAML:** generated specs carry a
  header block (source `.dlm`, dlm_id, base, adapter, generation
  timestamp, sway version) and a one-line `# intent` comment above
  every probe entry. No new dep — pyyaml + a position-based
  post-processor instead of `ruamel.yaml`.
- **D6 — `sway run --dry-run` + `sway list-probes`:** dry-run validates
  the spec, prints a probe table (#, name, kind, category, enabled),
  and exits 0 without building a backend. `list-probes` prints every
  shipped probe kind with its category and the first line of its
  docstring.
- **D7 — `sway doctor --json`:** machine-readable doctor payload
  (`sway_version`, `python`, `platform`, `extras`) keyed by extra and
  module. CI-grep-friendly.
- **D8 — `dlm_source` resolution warns instead of silently swallowing:**
  when the spec sets `dlm_source` but the `[dlm]` extra isn't installed
  (or the bridge errors), the runner emits a yellow stderr warning with
  the install hint and continues with `sections=None`. No more silent
  "why are my section probes SKIPping?" debugging.
- **D9 — markdown column parity with terminal:** the markdown probes
  table now carries `score`, `raw`, `z`, `duration`, and `note`
  columns. A `Top findings` section follows the table; a `Skipped
  probes` section closes when extras are missing.
- **D10 — unified number formatters:** `format_score`, `format_raw`,
  `format_z`, `format_duration_s` live in `suite/report.py` and are
  the only numeric formatters used. Thousands separators above 1 000;
  `—` glyph for `None` / non-finite. Single source kills cross-surface
  drift.
- **D11 — `sway report --format` validated via `StrEnum`:** the
  Typer-level enum rejects unknown formats with the standard
  "Invalid value for '--format'" error. No more silent fallback to
  the terminal renderer on a typo.
- **D12 — `sway check` verdict banner:** prints
  `✅ adapter is +4.2σ above noise` (green ≥ 3σ),
  `⚠️ adapter is +1.5σ above noise — marginal` (yellow ≥ 1σ), or
  `❌ adapter is +0.3σ — indistinguishable from noise` (red) above
  the full report. Calibrated on the delta_kl z-score; `check`
  now runs `null_adapter` first so the z-score is available.
- **D13 — `sway diff` regression summary:** after per-probe deltas,
  the diff prints `A→B: N regressed >0.10, M regressed >0.20,
  composite Δ=±X.XX` color-coded by direction.
- **D14 — adapter paths with spaces render safely:** `_adapter_label`
  wraps the path in double quotes when any whitespace is present.
- **D15 — long messages wrap, don't truncate:** the terminal probe
  table column uses Rich's `overflow="fold"` instead of an 80-char
  hard cut with an ellipsis. The full message is always visible.
- **B16 — single markdown renderer:** `report.from_json` round-trips
  saved JSON back into the canonical dataclass pair, so
  `sway report --format md` and `sway run --markdown` both flow
  through `report.to_markdown` with identical output. The legacy
  `_render_markdown_from_json` and `_render_junit_from_json` helpers
  in `cli/commands.py` are deleted.
- **B17 — `None` scores render as `—`:** the unified formatters cover
  every render path; no more `0.00` masking missing data.
- **B18 — `baseline` row labeled `(informational, weight=0)`:** in
  both terminal and markdown component breakdowns, matching the
  Sprint 03 explicit-weight-zero decision.

### Sprint 05 — Probe quality & edge cases

Closes Audit 01 findings B6, B7, B8, B9, B10, B11, B12, B13, B14, B20,
B21, B22.

- **B6 — `tail_logprob` semantics:** the field is now `float | None`
  with three discrete states: `None` (top-k covered the full vocab — no
  tail to redistribute), `0.0` (a real tail underflowed below fp32),
  and a measurable negative log-prob. HF, MLX, and dummy backends all
  emit `None` when `k == vocab`. Preflight check guards against
  non-finite `tail_logprob` only when it's a number.
- **B7 — probe validation before backend build:** new
  `validate_all_probes(suite)` helper collects every spec error in one
  pass; `_execute_spec` calls it *before* materializing the backend so
  a typo in `kind:` surfaces immediately, and *all* typos surface in a
  single error message.
- **B8 — `autogen` style prompts:** `style_fingerprint` no longer
  receives the leading sentence of a prose section (which elicited doc
  *content*, not stylistic voice). Replaced with a fixed
  `_STYLE_ELICITATION_PROMPTS` set of 6 open-ended, content-neutral
  prompts.
- **B9 — sentence-transformer caching:** `_load_embedder` is now
  `functools.lru_cache(maxsize=4)`. Suites that run `adapter_revert`
  back-to-back (multi-adapter diff, repeated probes) reuse the same
  ~80 MB embedder instead of re-loading every probe call.
- **B10 — `_lcs_ratio` docstring rot:** the function name is a
  historical misnomer (it's gestalt similarity via
  `difflib.SequenceMatcher`, not LCS). Docstring rewritten to make the
  contract explicit; rename deferred to v0.2 to preserve the public
  API surface.
- **B11 — leakage adversarial perturbations:** added 4 new perturbations
  (`synonym_swap` with a hand-curated 50-pair table, `clause_reverse`,
  `prefix_inject`, `register_shift`) alongside the original three.
  Default perturbation list is now all 7; the spec accepts any subset.
- **B12 — calibration pack expansion:** `BUILT_IN_PACK` grew from 30
  to **200** items (geography, natural sciences, arithmetic, language,
  history, biology, technology, miscellaneous). All public-domain /
  hand-composed grade-school facts — no third-party dataset license
  attaches. `items_limit` docstring documents the new resolution
  (~0.5 pp per regressed item vs the old ~3.3 pp).
- **B13 — tokenizer-aware `_stuffing`:** `prompt_collapse` now derives
  its padding from the model's pad / unk / EOS token via the backend's
  tokenizer, making the metric language-agnostic. The pre-B13 hardcoded
  English string remains as the dummy-backend fallback and behind a
  `legacy_stuffing: bool = False` spec field for one-release
  backward-compat.
- **B14 — `preference_flip` per-triple error fence:** wrapped each
  triple's `logprob_of` calls in `try/except ProbeError`. A single bad
  triple no longer kills the whole batch; `evidence["dropped_triples"]`
  + `dropped_reasons` surface the count + first 5 reasons. When *every*
  triple raises, the probe routes to ERROR with a clear explanation.
- **B20 — custom backend protocol checks:** `_load_custom` now
  isinstance-checks `NullCalibratedBackend` and
  `ScalableDifferentialBackend` after the `DifferentialBackend` check.
  The set of satisfied protocols is stamped onto the instance as
  `__sway_protocols__: tuple[str, ...]` so the report can show which
  features are available without re-checking.
- **B21 — `RunContext.null_stats` truly frozen:** the runner now wraps
  the stats dict in `types.MappingProxyType` before threading it
  through. The dataclass was already `frozen=True` but the dict was
  mutable by reference; B21 makes the docstring's "frozen" claim
  literally true. Field type widened from `dict[str, dict[str, float]]`
  to `Mapping[str, Mapping[str, float]]`.
- **B22 — `ModelSpec.adapter` path normalization:** added a pydantic
  `field_validator` that runs `Path.expanduser().resolve()` on the
  field at spec-load time. Backends no longer re-do the work; the cache
  key in `_null_cache.compute_key` is now stable regardless of how the
  user spelled the path in YAML or on the CLI.

### Sprint 04 — Integration & regression testing

Closes Audit 01 findings C1, C2, C5, C6, C7, C8, C10, C11, C12, B3
(test side), B15.

- **Slow lane runs end-to-end** (C1): the `huggingface_hub` dev dep
  added in S01 is verified; `pytest -m "slow or online"` now executes
  every checked-in integration test from a clean clone.
- **HF backend coverage 21% → 91%** (C2) — measured combined fast +
  slow lane against `dlm_sway.backends.hf`. New tests:
  - `tests/integration/test_hf_adapter_toggle.py` extended with a
    bit-identical ft → base → ft roundtrip (B15 mitigation).
  - `tests/integration/test_hf_scaled_adapter.py`: λ sweep
    monotonicity + `LoraLayer.scaling[key]` restoration on clean exit
    AND on exception.
  - `tests/integration/test_hf_null_adapter.py`: same-seed
    determinism, different-seeds divergence, original adapter
    restoration on clean exit AND on exception.
  - `tests/integration/test_hf_scoring.py`: `logprob_of` (incl.
    zero-token-completion → ProbeError), `rolling_logprob`,
    `next_token_dist`, `_HFView.generate` (greedy + sampled-with-seed
    determinism).
  - `tests/unit/test_backend_hf_helpers.py`: direct unit coverage on
    `_resolve_dtype` and `_detect_device` so dtype regressions are
    caught in the fast lane.
- **MLX smoke test** (C5): `tests/integration/test_mlx_smoke.py`
  exercises `MLXDifferentialBackend` on darwin-arm64 with a small
  LoRA adapter. Skips cleanly on non-darwin / non-arm64 / no-mlx_lm
  / missing-fixture. Reproducible adapter builder ships at
  `tests/fixtures/build_mlx_adapter.py` (run once on a Mac to
  populate the fixture directory).
- **`sway gate` exit code pinned** (C6):
  `tests/integration/test_sway_gate_exit_code.py` covers PASS
  (exit 0), FAIL verdict (exit 1), and below-threshold-with-passing
  verdicts (exit 1) via Typer's `CliRunner`.
- **`dlm` import ban regression-guarded** (C7):
  `tests/unit/test_dlm_not_imported.py` patches every `dlm.*` entry in
  `sys.modules` to `None` and asserts the dummy suite runs end-to-end
  without ImportError, plus that `sway autogen` surfaces a clean
  install-hint error rather than a stack trace.
- **Pathological probe coverage** (C8 + B3 test side):
  `tests/unit/test_probe_adapter_ablation.py` now drives
  monotonically-decreasing curves through the helper and pins
  probe-level `evidence["saturation_reason"]` for flat / found /
  overshoot-with-dip / non_monotonic shapes via a monkeypatched
  `divergence`.
- **WARN-branch numerical formula pinned** (C10):
  `test_warn_branch_score_formula_pinned` in
  `tests/unit/test_probe_preference_flip.py` asserts the exact
  `score = 0.5 + mean_delta / 4.0` formula with a hand-computed
  expected value.
- **Disjoint top-k divergence** (C12):
  `test_disjoint_top_k_supports_produce_finite_divergence` in
  `tests/unit/test_divergence.py` covers the
  `aligned_probs` tail-redistribution path against fully-disjoint
  base / ft supports — JS comes out finite and approaches its
  theoretical ln(2) bound.
- **Report schema snapshots** (C11): `tests/unit/test_report_snapshot.py`
  byte-compares `to_json` / `to_markdown` / `to_junit` against
  checked-in snapshots under `tests/snapshots/`. Intentional schema
  bumps: `SWAY_UPDATE_SNAPSHOTS=1 uv run pytest …` and commit the
  updated files. No new dep — hand-rolled diff helper.
- **CI workflow shipped**: `.github/workflows/ci.yml` runs the fast
  lane (unit + lint + mypy) on every push / PR, and the slow lane
  (integration, HF backend) on schedule (nightly 07:00 UTC), manual
  dispatch, push to main, and PRs that touch `src/dlm_sway/backends/`,
  `tests/integration/`, or `pyproject.toml`.

### Sprint 03 — Documentation truth & dead code

Closes Audit 01 findings P03, P04, P05, P07 (doc), P08, P09, P10, P14,
P15, P17, P18.

- **Determinism is now wired** (P09): the suite runner calls
  `dlm_sway.core.determinism.seed_everything(spec.defaults.seed)` before
  any backend work runs. The achieved determinism class (`strict` /
  `best_effort` / `loose`) is captured in `SuiteResult.determinism` and
  surfaces in the report footer + JSON payload + markdown header.
- **`defaults.differential: false` works end-to-end** (P14): new
  `backends/two_model.py` with `TwoModelDifferential` wrapper +
  `build_two_separate(spec_models)` helper. `_execute_spec` routes to
  the wrapper when the flag is false. Doubles memory; primarily for
  custom backends that can't toggle adapters in place.
- **`score_weights` overridable from YAML and CLI** (P15): new
  `SuiteDefaults.score_weights` field with subset-validating pydantic
  field validator (rejects unknown categories, negative weights, all-zero
  weights). New `--weights k=v,k=v` flag on `sway run` and `sway gate`.
  Partial overrides merge with `DEFAULT_COMPONENT_WEIGHTS` so users
  rarely have to respecify all five categories.
- **`baseline` row labeled `(informational)` with weight 0.0 explicit**
  (P08, B18): added to `DEFAULT_COMPONENT_WEIGHTS` so the row appears
  in reports for transparency but contributes nothing to the composite.
  Terminal renderer adds the `(informational)` annotation column;
  markdown table adds a `weight` column with the same label.
- **Extended (9-dim) style fingerprint when `[style]` is installed**
  (P10): `style_fingerprint` now actually uses spaCy POS-tagging and
  textstat syllable counting. Three new dims appended to the existing 6:
  passive-voice rate, POS 4-gram entropy (Shannon, bits), syllables per
  word. Spec field `extended: "auto" | "on" | "off"` (default `"auto"`)
  controls the path. `evidence["schema_version"]` bumped to `2` so
  snapshot consumers can branch on dimensionality.
- **Stale "Known gaps" entries refreshed** (P03, P04, P05): removed
  "null_adapter not yet wired" (delivered in S01/S02), "custom backend
  stubbed" (it's full), "MLX paths raise" (rewritten to describe the
  real limitation: two model copies because `mlx_lm` has no runtime
  adapter toggle).
- **README adds determinism + weights + differential documentation**
  and a calibration paragraph that points at the per-kind null matrix.
- **`delta_kl` docstring rot fixed** (P17): dropped the dead
  ``:mod:`dir``` reference.
- **Meta-test `test_no_dead_options.py`** (P14/P15 regression guard):
  greps the source tree for documented spec/CLI option names and
  asserts each has a consumer outside its declaration site. Catches the
  exact "documented but unused" pattern Audit 01 flagged.

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
  - `null_adapter` — per-kind null distribution matrix for z-score calibration (wired end-to-end in Unreleased)

### Infrastructure

- `DifferentialBackend` + `ScalableDifferentialBackend` protocols
- HuggingFace + PEFT backend with `disable_adapter` / `set_adapter` toggling and LoRA-scale mutation
- Dummy backend for unit tests (canned responses + linear-blend scalable mode)
- YAML spec loader, composite score (four-category weighted), rich terminal + JSON + JUnit + Markdown reports
- Typer CLI: `run`, `gate`, `check`, `diff`, `autogen`, `doctor`, `report`
- `.dlm` bridge (`dlm-sway[dlm]`): resolver + full-battery autogen
- Matplotlib visualizations (`dlm-sway[viz]`): SIS bar chart, ablation curve, KL histogram

### Known gaps

- MLX backend loads **two** model copies in memory (base + adapter-fused) because `mlx_lm` has no runtime adapter toggle. Memory footprint is ~2x the HF path; fine for the small (<3B) models MLX typically runs.
- MLX requires a pre-converted `.npz` adapter — raw PEFT safetensors are rejected by `mlx_lm.load`. A PEFT-→-MLX converter is a future milestone.
- PyPI publication of the `dlm-sway` wheel is pending a clean CI release workflow.
