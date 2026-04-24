# Changelog

## Unreleased

## 0.1.0 — 2026-04-24

First PyPI release. Alpha — API not guaranteed stable until v1.0.
Rolls up every sprint through Sprint 22; earlier sprints accumulated
under the rolling Unreleased header pre-release.

### Sprint 22 — v0.1.0 release + F06 dlm-compat test + Docker image

First PyPI publish. Version bumped from `0.1.0.dev0` to `0.1.0`.
Pre-commit consumer `rev:` pins switch from commit SHA to `v0.1.0`.
Closes Audit 03 F05 (SHA-pin resolves at tag) and F06 (dlm API
compatibility regression test).

**D-path1 — PyPI publish.**

- **`pyproject.toml`** — version `0.1.0`. `Development Status :: 3 -
  Alpha` classifier already present.
- **`src/dlm_sway/__init__.py`** — `__version__ = "0.1.0"`.
- **`README.md`** — real `pip install "dlm-sway[hf]"` recipe replaces
  the "Planned PyPI install (not live yet)" placeholder. Pre-alpha
  banner swapped for an explicit Alpha + semver-from-v1.0 disclaimer.

**F06 — dlm API compatibility regression test.**

- **`core/errors.DlmCompatError`** — new typed exception. Raised
  when dlm's public surface drifts out of the contract sway's
  resolver depends on (currently: `dlm.base_models.resolve(key).hf_id`).
  Message includes installed-dlm version + a pip-install hint.
- **`integrations/dlm/resolver`** — the `except Exception: return
  base_model` silent fallback is gone. `resolve()` raising or
  returning an object without `hf_id` now raises `DlmCompatError`
  (chained via `__cause__` for tracebacks). An extra
  `_installed_dlm_version()` helper introspects
  `importlib.metadata` for the error message.
- **`tests/unit/test_dlm_bridge`** — 2 new regression tests pin the
  two drift branches (missing `hf_id`; `resolve` itself raising).
- **`tests/integration/test_dlm_api_compat`** — new slow+online+dlm
  integration test iterates dlm's registry keys and asserts every
  entry resolves to a spec with a plausible `hf_id`. Gracefully
  skipped (via `pytest.importorskip`) when the `[dlm]` extra isn't
  installed, so CI without dlm published to PyPI still passes.
- **`pyproject.toml`** — `[dlm]` extra pinned to `dlm>=0.9,<1.0`.
  Upper bound tightens to the range the compat test has validated;
  bump when dlm cuts v1.0 and the contract is re-verified.

**D-path2 — Docker image (`sway-gate`).**

- **`Dockerfile.gate`** — `python:3.11-slim` + `dlm-sway[hf,semsim]`
  at the build-time `SWAY_VERSION` ARG. Pre-fetches the MiniLM
  weights (`sentence-transformers/all-MiniLM-L6-v2`, ~80 MB) so
  `adapter_revert` / `cluster_kl` probes don't cold-download on
  first gate run. `ENTRYPOINT ["sway"]` — pre-commit hook passes
  `gate` as the first container arg.
- **`.github/workflows/docker.yml`** — on `v*` tag push + manual
  dispatch. Builds on GHA, pushes to `ghcr.io/
  tenseleyflow/sway-gate:{vX.Y.Z, latest}`. Smoke-tests the pushed
  image with `--version`.
- **`.pre-commit-hooks.yaml`** — new `sway-gate-docker` variant
  using `language: docker_image` pointing at
  `ghcr.io/tenseleyflow/sway-gate:v0.1.0 gate`. Third option
  alongside `sway-gate` (system PATH) and `sway-gate-isolated`
  (fresh venv).

**F05 closure.**

- **`.pre-commit-hooks.yaml`** — isolated variant's
  `additional_dependencies` migrates from
  `dlm-sway[hf] @ git+https://…@<SHA>` to
  `dlm-sway[hf]==0.1.0`. Consumer `rev:` in
  `.pre-commit-config.yaml` pins to `v0.1.0`. SHA churn over.

**README.**

- Real `pip install` recipes for every extra (`[hf]`, `[dlm]`,
  `[all]`, …). "Install from source" kept for contributor workflow.
- Pre-commit section: three hooks (system / isolated / docker)
  with first-run-cost comparison table.

### Sprint 21 — Audit 03 closure

Closes the Audit 03 short list — 2 🟠 major + 5 🟡 minor findings the
audit recommended for v0.1.0 readiness. Full audit at
`.docs/audits/03-final-audit.md`. Verdict was YELLOW-leaning-GREEN
with zero critical findings; this sprint pushes it to GREEN on the
in-scope items. Deferred: F01 (MLX converter — its own feature
sprint, S24), F05 (resolves naturally at v0.1.0 tag), F06 (dlm API
compat test — lands with v0.1.0 release in S22). `v0.1.0` PyPI
publish is deferred to S22 (requires release coordination +
credentials).

**🟠 Major — degenerate z-score clipping (F02).**

- **`probes/null_adapter`** — null stats now carry an explicit
  `degenerate: 1.0` field when the calibration ran but produced an
  unusable baseline (`runs: 1`, or every seed producing the *exact*
  same raw). The std floor at `1e-6` is preserved for valid-but-tight
  multi-seed nulls so they still calibrate. Audit observed a
  `+290,766σ` leakage-probe z under `runs: 1`; post-fix, z_score
  refuses and the probe falls back to fixed thresholds with a clear
  footer rollup.
- **`probes/_zscore.z_score`** — refuses when `stats["degenerate"]`
  is truthy, independent of the std check. Belt-and-suspenders on
  top of the existing MIN_STD guard.
- **`suite/report.collect_degenerate_null_kinds`** — new rollup
  surface. Terminal + markdown footers gain a
  "N probe kind(s) had a degenerate null baseline — bump `runs:`"
  block, distinct from the existing null-opt-outs rollup.
- 12 new unit tests across `test_null_calibration.py`,
  `test_zscore_helpers.py`, `test_report_extras_rollup.py`, and
  `test_probe_external_perplexity.py`. Existing
  `test_std_floor_prevents_runaway_zscore` was rewritten to assert
  the fixed behavior (the pre-fix contract WAS the audit's bug).

**🟡 Minor — CI resilience (F03).**

- **`pytest-timeout>=2.3`** + **`tenacity>=9.0`** in dev deps.
- **`tests/fixtures/tiny_model.py`** wraps `snapshot_download` with
  exponential-backoff tenacity retry (3 attempts, 5-10-20s backoff)
  + `etag_timeout=10` to bound per-file head probes. Benefits every
  slow+online test.
- **`tests/integration/test_determinism_golden.py`** gains
  `@pytest.mark.timeout(600)`. A silent network hang now surfaces
  as a test failure with actionable output (audit observed 20m
  workflow-timeout hang on Sprint 19 merge run 24747915467).

**🟡 Minor — outlier-miner pool guard (F04).**

- **`mining/outlier_miner.mine_outliers`** — raises `SwayError` when
  the pool has fewer than `2·top_k` distinct scored prompts, with
  an actionable `--top-k N` hint. Pre-fix: 1-distinct-prompt pool
  produced `top=[p], bottom=[p]` (identical lists — no outlier
  contrast). Guard applies AFTER scoring so unsupported probe kinds
  still return the empty-result path.

**🟡 Minor — autogen skipped-probes comment (F07).**

- **`integrations/dlm/autogen.collect_skipped_probe_reasons`** — new
  public helper returning `(probe_kind, reason)` tuples for probes
  `_build_suite` intentionally omitted for the given handle.
- **`_render_annotated_yaml`** — when `skipped` is non-empty, header
  gains a `# skipped: <kind> (<reason>)` block. Users no longer have
  to diff the autogen source to understand which probes are missing
  from their generated `sway.yaml`.

**🟡 Minor — CI Node.js 20 deprecation silence (F08).**

- **`.github/workflows/ci.yml`** sets
  `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: "true"` at the workflow
  level. Silences the Node 20 deprecation warnings each CI log
  currently carries (deadline June 2026). Temporary until every
  action we pull bumps to Node 24 natively.

**🟡 Minor — portable `dlm_source` (F09).**

- **`integrations/dlm/autogen._portable_dlm_source`** emits a
  cwd-relative path when the `.dlm` lives inside the cwd, absolute
  otherwise. The cwd-relative form survives cross-machine checkout
  (CI agents, other devs); the old absolute-path emission broke the
  moment a committed autogen'd YAML was run on a different host.

**Other.**

- **640 unit tests pass** (up from 626 before S21 — 14 new
  regression tests across F02/F03/F04/F07/F09).
- **Pragmatic deviations from original scope:** S21 was audit-scoped
  to ~2 days of cleanup. Actual depth: 6 findings closed with 14
  new regression tests. One existing test
  (`test_std_floor_prevents_runaway_zscore`) explicitly rewritten to
  flip its assertion — the pre-fix contract the test encoded WAS
  the audit's reported bug.

### Sprint 19 — Pre-commit hook `sway gate`

Closes Audit 01 stretch-list F-item "pre-commit hook sway gate." Ships
a `.pre-commit-hooks.yaml` declaring two hook variants so
[pre-commit.com](https://pre-commit.com) users can gate their adapter
on every commit that touches a spec, `.dlm` file, or adapter directory.

- **Two hooks, pick your posture.**
  - `sway-gate` — `language: system`. Uses the sway install on the
    user's `PATH`. Fast, zero install cost. Recommended default.
  - `sway-gate-isolated` — `language: python` + git-based
    `additional_dependencies` (`dlm-sway[hf] @ git+...@<SHA>`).
    Self-contained — pre-commit builds a fresh venv and installs
    sway + torch + transformers on first run (~5 GB, ~2 min).
- **Rev pinning stance.** Example config pins to a commit SHA, not
  `HEAD`. Sway is pre-v0.1.0 — no tagged release yet — and `HEAD`
  drifts silently on every `pre-commit autoupdate`. SHA pinning is
  the honest pre-release pattern; migration to `rev: v0.1.0` is a
  README edit once the first release lands.
- **`pass_filenames: false` on both variants.** `sway gate` takes
  exactly one spec path; the user declares it via `args:`. The
  `files:` regex decides when the hook fires. No ambiguity.
- **Scope discipline.** Neither hook surfaces `--json` or
  `--markdown` flags. The hook gates (non-zero on FAIL) and nothing
  else. Users wanting report artifacts run `sway run` separately.
- **Readme "Pre-commit" section** documents the consumer-side config
  for both variants, the SHA-pinning rationale, and the one-time
  install cost of the isolated variant.
- **`examples/precommit-example/`** — template `sway.yaml`,
  consumer-side `.pre-commit-config.yaml`, and a README walk-through.
- **`tests/integration/test_pre_commit_hook.py`** (slow + online) —
  three cases: (a) parse-and-shape smoke test on
  `.pre-commit-hooks.yaml`; (b) pass-case runtime — spawns
  `pre-commit run sway-gate` as a subprocess against a tmp git repo
  with a real LoRA on SmolLM2-135M, asserts exit 0; (c) fail-case —
  same fixture with an impossible `assert_mean_gte`, asserts
  non-zero exit + `gate FAILED` banner.
- **`pre-commit>=3.8`** in `[dependency-groups].dev` so the
  integration test can invoke it as a subprocess. Keeps the tool
  out of the user's runtime deps.

### Sprint 18 — Cross-platform determinism golden

Closes Audit 01 stretch-list F-item "cross-platform determinism golden
test." The README's "deterministic on CPU where possible" claim held
in theory; now it's pinned by CI on two platforms.

- **New module** `src/dlm_sway/core/golden.py`. Tolerance-aware JSON
  comparator with `compare_goldens(actual, expected, *, logprob_tol=1e-6,
  score_tol=1e-4)` and `mask_variable_fields` for stripping
  timestamps, wall-seconds, `duration_s`, `backend_stats`,
  `sway_version`, and cwd-resolved path identifiers (`adapter_id`,
  `base_model_id`) before comparison. No torch dep; runs in the fast
  lane.
- **New integration test** `tests/integration/test_determinism_golden.py`
  (slow + online). Builds a deterministically-seeded LoRA on
  SmolLM2-135M, runs a minimal 2-probe suite (delta_kl +
  calibration_drift), and diffs the JSON output against
  `tests/golden/expected_<platform>.json`. `SWAY_UPDATE_GOLDENS=1`
  toggles regen mode; missing golden → SKIP with a regen recipe.
- **New CI matrix** `determinism-golden` with
  `strategy.matrix.os: [ubuntu-latest, macos-latest]` in
  `.github/workflows/ci.yml`. Triggered by changes under
  `tests/golden/**`, `src/dlm_sway/core/golden.py`, or the test file
  itself (plus the standard schedule/dispatch/push triggers).
- **`workflow_dispatch` regen mode** — dispatching the CI workflow
  with `regenerate_goldens=true` flips `SWAY_UPDATE_GOLDENS=1` on both
  matrix legs and uploads the regenerated JSONs as per-platform
  artifacts. Meant for deliberate "yes I changed the algorithm"
  flows.
- **Tolerance rationale** (documented in `core/golden.py`): `1e-6` for
  logprob-like numeric fields sits above typical BLAS-implementation
  drift (`1e-8`–`1e-7` band between OpenBLAS on linux and Accelerate
  on darwin) but below real algorithm-change drift. `1e-4` for score
  fields absorbs composition noise.
- **25 new unit tests** for the comparator covering mask coverage,
  tolerance thresholds, structural diffs (missing keys, length
  mismatches, type mismatches), NaN/inf edge cases, and a realistic
  two-masked-payloads round-trip.
- **Pragmatic scope deviation**: the sprint envisioned a checked-in
  5 MB adapter binary. Instead the test builds it from a fixed seed
  at runtime (same pattern as `test_external_perplexity_e2e` and
  `test_cluster_kl_e2e`). Avoids the regeneration chore noted in the
  sprint's risks section.
- **Linux golden bootstrap**: first PR ships `expected_darwin.json`
  only; the linux leg SKIPs with a recipe pointing at the
  `workflow_dispatch` regen mode. Maintainer dispatches, downloads
  the artifact, commits `expected_linux.json`, and the next CI run
  asserts cleanly on both platforms. One-time onboarding cost.

### Sprint 17 — Adversarial paraphrase mining + outlier-prompt miner

Closes Audit 01 innovation item F11. Adds `sway mine` — an evaluation
companion to `paraphrase_invariance` that surfaces the paraphrases a
memorizing adapter most reliably fails on, plus a secondary mode that
ranks `delta_kl` prompts by per-prompt divergence.

**`sway mine --mode paraphrase`** — per `paraphrase_invariance` case:

1. Generate candidate paraphrases via nlpaug `SynonymAug` (WordNet,
   deterministic under a fixed seed; back-translation is a user-supplied
   escape hatch to keep the default fast and offline).
2. Embed candidates through the shared MiniLM cache (same 80 MB load
   `adapter_revert` pulls) and greedy farthest-first select the top-K
   most pairwise-distant ones — dodges nlpaug's tendency to emit
   near-duplicate synonym swaps.
3. Rank by per-token lift gap: `(ft(prompt, gold) - base(prompt, gold))
   - (ft(candidate, gold) - base(candidate, gold))`. Large positive gap
   = the candidate breaks the adapter's lift = the adapter doesn't
   generalize there.
4. Emit `sway-mined-paraphrase.yaml` with a `mined_cases:` block paste-
   compatible with `paraphrase_invariance.cases`.

**`sway mine --mode outliers`** — rank a prompt pool by per-prompt
`delta_kl.raw`. Pool defaults to the spec's own `delta_kl` prompts;
`--from-corpus public_domain_en` draws from S09's CC0 corpus instead.
Emits top-K and bottom-K blocks: the highest-divergence prompts (best
for tightening a gate) and lowest-divergence (candidates to drop from
the suite). `leakage` and `paraphrase_invariance` have section/case-
based specs; outlier mining on those is future work, documented
inline in `mining/outlier_miner.py`.

- **New package** `src/dlm_sway/mining/` — two modules,
  `paraphrase_miner.py` and `outlier_miner.py`, plus a thin
  `corpus_prompts` helper for the `--from-corpus` path.
- **New CLI subcommand** `sway mine SPEC --mode paraphrase|outliers
  [--out FILE] [--top-k N] [--n-candidates N] [--from-corpus NAME]
  [--seed N]`. Paste-compatible YAML emission by default; `--out`
  overrides the filename.
- **18 new unit tests** — 7 for the paraphrase miner (ranker,
  diversity filter, dedup, input validation), 8 for the outlier
  miner (delta_kl ranking, corpus wiring, unsupported-kind skip), 3
  for the CLI (paraphrase + outliers modes + no-prompts error path).
- **Prove-the-value test** at
  `tests/unit/test_paraphrase_miner_prove_value.py`. On a
  deliberately-memorizing dummy backend the hand-written paraphrase
  list passes with `generalization_ratio > 0.5`; substituting the
  mined list (same seed, same adapter) drops the ratio below 0.5
  and flips the verdict to FAIL, with a ≥ 0.3 ratio gap.
- **Dependency footprint**: no new extras. The paraphrase miner
  uses nlpaug (already in `[style]`) + sentence-transformers (in
  `[semsim]`); the diversity filter reuses the embedder
  `adapter_revert` already pulls. Graceful `BackendNotAvailableError`
  with a pip hint when the extras aren't installed.

### Sprint 20 — Audit 02 closure

Closes the full finding inventory from `.docs/audits/02-followup-audit.md`
(1 🔴 critical, 8 🟠 major, 11 🟡 minor, 4 stronger-test opportunities, 3
dead-code items). Sixteen sprints of innovation work had accumulated
since Audit 01; the re-audit surfaced one ship-blocker (S14's bootstrap
CI dropped at the runner boundary) plus a grab-bag of claim-vs-code
gaps, dead branches, and untested contracts.

**🔴 Critical — ship-blocker.**

- **F01** — `suite/runner.py:_with_duration` now forwards every
  `ProbeResult` field. The pre-fix version silently dropped `ci_95`
  on every probe, making S14's headline deliverable runtime-inert in
  any real `sway run`. Regression test at the runner boundary + snapshot
  fixture carrying a populated `ci_95` pin the fix.

**🟠 Major — claim-vs-code gaps.**

- **F02** — real `sklearn.cluster.KMeans` exercised by two new unit
  tests (separation + seed determinism) + a slow+online integration
  test at `tests/integration/test_cluster_kl_e2e.py`. Before: every
  cluster_kl test monkeypatched `_kmeans_cluster` with an argmax stub,
  leaving S16's reason-for-being unverified.
- **F03** — `sway list-probes` falls back to the defining module's
  `__doc__` when a probe class has no docstring. Every one of the 13
  shipped probes now renders a summary row.
- **F04** — `sway doctor` probes `plotly` (the load-bearing `[viz]` dep),
  `sklearn` (S16 cluster_kl), and the `api` extras (`httpx`, `tenacity`).
- **F05** — README primitives table reflects 13 probes (adherence +
  cluster_kl, calibration + external_perplexity, baseline row for
  null_adapter).
- **F06** — `TwoModelDifferential` composes `safe_for_concurrent_views`
  from its two inner backends. Before: the attribute was missing and
  the runner defaulted to `False` even when both inners set `True`,
  breaking S13's concurrency claim at the only supported differential-
  use pattern.
- **F07** — `integrations/dlm/autogen` emits `cluster_kl` when the
  prompt pool clears 20 entries; markdown report gains a per-probe
  "Cluster breakdown" section with per-cluster mean KL + exemplars.
- **F08** — `backends/dummy._NullView` RNG uses `hashlib.md5` instead
  of Python's PYTHONHASHSEED-salted `hash()`. Cross-process determinism
  test at `tests/unit/test_cross_process_determinism.py` pins the fix.
- **F09** — trace writer ↔ analyzer round-trip test at
  `tests/unit/test_runner_backend_stats.py`: runs a suite with
  `trace_path=`, loads the file back, asserts probe labels + hit/miss
  counts match `backend_stats`.

**🟡 Minor — dead code, doc drift, latent brittleness.**

- **F10** — removed dead `_ft_view` branch in `probes/prompt_collapse.py`.
- **F11** — pytest plugin resolves spec paths against `config.rootpath`,
  not process cwd.
- **F12** — `backends/api._post_completions` delegates to tenacity's
  `Retrying()` callable; removes the unreachable post-loop fallback.
- **F13** — `preference_flip` WARN branch emits `ci_95` and `z_by_rank`
  for consistency with every other numeric probe's WARN shape.
- **F14** — `adapter_ablation` module docstring documents why `ci_95`
  renders as em-dash (it's a curve-fit, not a sample-mean aggregator).
- **F15** — report footer surfaces `null_adapter` opt-outs (probes with
  `calibrate_spec=None`) in both terminal and markdown surfaces.
- **F16** — report category ordering derives from
  `DEFAULT_COMPONENT_WEIGHTS` and accepts unknown categories from
  custom `Probe` subclasses.
- **F17** — `cluster_kl` degenerate zero-variance case now returns
  `Verdict.WARN` with `evidence["degenerate_zero_variance"]=True` and
  suppresses z-score to avoid a spurious small-sample-noise calibration.
- **F18** — `_calibration_pack.py` gains per-section provenance notes
  (F18 audit trail) without noisy per-item annotations.
- **F19** — pytest plugin defers `dlm_sway.core.result` /
  `dlm_sway.core.errors` imports to call sites so non-`@pytest.mark.sway`
  users don't pay the load tax.
- **F20** — `sway check` `--help` documents the σ-banner's null-calibration
  dependency (fall-through to composite score band when null SKIPs).

**Stronger-test opportunities — regression tests for things we weren't
pinning.**

- **#9** — `probes/_divergence` rejects effectively-uniform TokenDists
  (spread < 1e-9) via `_check_non_degenerate_token_dist`. Catches a
  shape-broken lm_head that would otherwise compute a trivial constant
  divergence across prompts. Two compatibility touchups rode with the
  change: dummy backend's synthesized `ft` dist and cluster_kl test
  fixture `_dist_broad` gained a tiny monotonic perturbation to clear
  the guard (real models never produce bit-uniform logits thanks to
  fp32 accumulation noise).
- **#10** — cross-verdict consistency test
  (`tests/unit/test_cross_verdict_consistency.py`): runs the same spec
  through `sway run`, `sway gate`, and `sway report --format junit` and
  asserts identical per-verdict tallies.
- **#11** — `sway doctor --json` schema-shape snapshot test locks the
  top-level keys and the per-extra module-name set.
- **#12** — subprocess determinism test asserts `_NullView.next_token_dist`
  is byte-identical across `PYTHONHASHSEED` values; pins F08's fix.

**Dead-code inventory — DC3–DC5.**

- **DC3** — `null_adapter.py`'s pre-S10 cache-promotion branch is
  annotated as legacy; removal queued for the next minor.
- **DC4** — dropped the stderr warning at `suite/runner.py` that fired
  whenever `concurrent_probes > 1` even though execution never fanned
  out. The spec field is still accepted (future pool will light it up);
  the noisy warning is gone.
- **DC5** — `tests/unit/test_model.py` grows coverage of `ModelSpec`'s
  dtype enum, `endpoint`, `trust_remote_code`, and `custom`/`api`
  `kind` branches.

Final state: 582 unit tests + 3 integration tests passing; mypy strict
clean across 55 source files; ruff + format clean across 137 files.
Sprint 20 is a single PR composed of per-finding, per-file commits so
the history reads as a fix-by-fix cleanup.

### Sprint 16 — Cluster-coherent KL probe

Closes Audit 01 innovation item F8. Adds a new `cluster_kl` probe that
answers the question `delta_kl` can't: the mean divergence may be the
same, but did the adapter shift the *right* topics, or is it a uniform
blunt-instrument shift?

- **New probe `cluster_kl`** (category: adherence). Embeds prompts via
  shared MiniLM (same cache key as `adapter_revert`), k-means clusters
  them at a fixed seed, measures per-prompt JS divergence between base
  and ft, and reports a **specificity ratio**:
  `between_variance / (between_variance + within_variance)`. Range
  `[0, 1]`: `≈ 0.5` on a blunt adapter that shifts every topic by the
  same amount, `→ 1.0` on a topic-targeted adapter. Pair
  `(mean_kl, specificity)` tells a more honest story than either number
  alone.
- **Bootstrap CI** on specificity. Resamples `(divergence, cluster_label)`
  pairs with replacement and takes the 2.5/97.5 percentiles of the
  bootstrap ratio distribution. Returns `None` below 4 prompts — matches
  the convention `core.stats.bootstrap_ci` uses.
- **Z-score calibration** against `null_adapter` baseline via the
  existing `_zscore` helpers. `calibrate_spec` synthesizes 8 mixed-topic
  sentinel prompts + k=2 for the null pass; the specificity distribution
  concentrates around 0.5 there, so a real adapter's separation above
  that baseline reads as a clean z-score.
- **Degenerate-input policy.** `< min_prompts` (default 20) → SKIP with
  a clear message; `num_clusters * 2 > num_prompts` → SKIP (per-cluster
  mean not well-resolved); empty prompt list → ERROR. Missing `[semsim]`
  extras → SKIP with a `pip install 'dlm-sway[semsim]'` hint.
- **Zero-variance fallback.** If every prompt produced the same
  divergence (canned stub data, no adapter motion), the specificity
  ratio is mathematically undefined; we return `0.5` — the null-adapter
  expectation — so downstream z-score reports "no signal" instead of
  NaN.
- **New dependency.** `scikit-learn>=1.4` added to the `[semsim]` extra
  (riding the 80 MB MiniLM load that probe already pulls in). Mirrored
  to `[all]` and to mypy's stubless-override list.
- **7 unit tests** in `tests/unit/test_probe_cluster_kl.py`: two-topic
  adapter → high specificity, uniform adapter → 0.5 fallback, too-few
  prompts → SKIP, empty prompts → ERROR, `num_clusters > prompts/2` →
  SKIP, CI bracketing, missing-extras SKIP path.
- **Prove-the-value test** at `tests/unit/test_cluster_kl_prove_value.py`.
  Two backends with comparable `delta_kl` (ratio < 3×) have specificity
  scores that split by at least 0.3 — concrete evidence that `cluster_kl`
  surfaces a structural distinction `delta_kl` merges.

### Sprint 15 — pytest plugin (`@pytest.mark.sway`)

Closes Audit 01 innovation item F10. Packages sway as a pytest
library so teams already running pytest adopt sway with a single
decorator instead of a subprocess wrapper.

- **New plugin** (`src/dlm_sway/pytest_plugin.py`) auto-loaded via
  the `pytest11` entry point after `pip install 'dlm-sway[pytest]'`.
  `@pytest.mark.sway(spec="...", threshold=0.0, weights=None)`
  expands a single pytest function into **one test item per probe**
  in the referenced spec + an optional `__gate__` item that fires
  only when the composite score drops below `threshold`.
- **Verdict translation.** `FAIL` / `ERROR` → pytest Failed,
  `SKIP` → pytest Skipped, `WARN` → pytest warning, `PASS` →
  pytest pass. Probe-level failures isolate: a failing adherence
  probe doesn't mask a failing calibration one, and `pytest -k
  adherence` runs just that probe.
- **Suite runs once per decorated function.** A session-scoped
  `_SuiteCache` keyed on `(spec_path, weights)` ensures the N-way
  item expansion doesn't multiply backend wall time. Two
  `@pytest.mark.sway` tests against the same spec share one run.
- **Malformed marks fail cleanly.** Missing `spec`, non-numeric
  `threshold`, non-dict `weights`, and unknown kwargs produce a
  synthetic `_ConfigErrorItem` with a green-field pytest failure
  line — no cryptic collection-time tracebacks.
- **New `[pytest]` extra** — just `pytest>=8.0`. Matches the pattern
  other plugin-style extras use. Also registers the `pytest11`
  entry point so the plugin is discovered automatically on install,
  consistent with pytest-cov / pytest-xdist.
- **Example directory** at `examples/pytest_integration/` with a
  minimal `sway.yaml` + `test_sway_gate.py` showing the decorator
  replacing a legacy `subprocess.run(["sway", "gate", ...])`
  wrapper. README gains a "Pytest integration" section pointing at
  it.
- **13 new unit tests** via pytest's canonical `pytester` fixture:
  marker registration, N-item expansion, FAIL / SKIP / ERROR
  routing, gate below-threshold failure, gate above-threshold pass,
  gate absence when `threshold=0`, malformed-mark error paths,
  cache sharing across multiple decorated tests.
- **Drive-by fix for the S14 CI-narrowing test flake.** The dummy
  backend's per-prompt noise was seeded via Python's `hash()`,
  which is salted per-process via `PYTHONHASHSEED`. Swapped to a
  stable `hashlib.md5`-derived seed so the narrowing invariant
  holds deterministically across test-order permutations.

### Sprint 14 — Bootstrap CIs + forward-pass trace CLI

Closes Audit 01 innovation items F9 (bootstrap confidence intervals
on raw metrics) + F12 (polished forward-pass trace CLI). Paired
sharpenings of existing probe output and existing trace
infrastructure.

- **New `core/stats.bootstrap_ci`** — percentile-bootstrap 95% CI on
  any sequence of per-sample measurements. Numpy-only, 1000-resample
  default, seeded from ``ctx.seed`` so intervals are reproducible.
  Short-circuits on non-finite / empty / degenerate-constant inputs;
  vectorized resample keeps the overhead ~1 ms per probe.
- **`ProbeResult.ci_95: tuple[float, float] | None`** — new field,
  threaded through `safe_finalize` (nulled if `raw` is nulled, so a
  CI never brackets a defensive-null point estimate). `to_json` /
  `from_json` persist the pair as a two-list.
- **Six aggregating probes emit `ci_95`:** `delta_kl` (over per-prompt
  divergences), `calibration_drift` (over per-item regression
  indicators), `external_perplexity` (over per-chunk deltas),
  `leakage` (over clean-recall rates), `paraphrase_invariance` (over
  verbatim lifts), `section_internalization` (over effective SIS
  scores). Each also lands the interval under
  ``evidence["raw_ci_95"]`` for JSON consumers.
- **Report tables add a `ci95` column** between `raw` and `z` in
  both terminal and markdown output. `format_ci` renders as
  ``[lo, hi]`` with em-dash for missing/non-finite. Markdown
  snapshot refreshed; JSON snapshot refreshed for the new
  per-probe `ci_95` field.
- **New `sway trace <jsonl>` CLI.** Reads the forward-pass trace
  JSONL the S07 runner writes when `--trace <path>` is set,
  aggregates into per-probe and per-view wall-time + hit-rate
  tables, plus a top-N slowest-events table. `--format
  terminal|md|json`, `--slowest K` to tune the tail size. The
  parsing tolerates old trace shapes (missing `probe` / `hit`
  fields) and skips malformed lines.
- **`suite/trace_analysis`** — typed dataclasses (`TraceEvent`,
  `ProbeSummary`, `ViewSummary`, `TraceReport`) + three renderers
  (terminal / markdown / json) matching the conventions of
  `suite/report` and `suite/compare`.
- **Committed trace fixture** at `tests/fixtures/trace_sample.jsonl`
  (8 events, 2 probes × 4 prompts with cache hits). Drives 22 new
  trace-analysis + CLI tests.
- **Prove-the-value (F9):** on a dummy backend with per-prompt
  variation, `delta_kl`'s CI narrows from `[0.33, 0.41]` (width
  0.079) at N=4 prompts to `[0.38, 0.42]` (width 0.044) at N=32 —
  1.8× tighter in width, tracking the √(N₂/N₁) ≈ 2.8× theoretical
  scaling minus dummy-backend dispersion.
- **Prove-the-value (F12):** the CLI's terminal output on the
  committed fixture correctly identifies `sis/base` (520.3 ms) as
  the single slowest event, `sis` (1,529 ms) as the slower probe
  over `dk` (529 ms), and `ft` (1,357 ms) as the slower view over
  `base` (701 ms).

### Sprint 13 — OpenAI-compatible HTTP scoring backend

Closes Audit 01 innovation item F7. Unlocks sway against hosted
fine-tunes — OpenAI platform, `vllm serve`, Ollama — without
requiring a local torch + PEFT load.

- **New backend `ApiScoringBackend`** (`backends/api.py`): scores
  against a ``/v1/completions`` endpoint via httpx. Implements
  `ScoringBackend` only (not `DifferentialBackend`) since one
  endpoint is one model; users compose two `ApiScoringBackend`
  instances behind the existing `TwoModelDifferential` wrapper for
  a full differential run. Method surface: `logprob_of` via
  `echo=True`+`logprobs=0`, `rolling_logprob` via the same,
  `next_token_dist` via `max_tokens=1`+`logprobs=K`, plus
  `preflight_finite_check` and an `ApiScoringBackend.generate` for
  probes that need text output (e.g. `leakage`).
- **Token-boundary handling.** The API returns tokens as strings, so
  `logprob_of` walks the echoed tokens by character length until the
  running total covers the prompt, then sums the rest. When the
  boundary falls mid-token, the partial token lands on the prompt
  side — over-counts the prompt, under-attributes to the completion
  — and the behavior is asserted by a dedicated test
  (`test_mid_token_prompt_leans_conservative`).
- **Retry + preflight.** Tenacity handles 5xx + network errors with
  exponential backoff (configurable `max_retries`). Preflight hits
  the endpoint once with `hello`, `max_tokens=1`, `logprobs=1`;
  rejects non-finite logprobs before the suite runs.
- **First backend with `safe_for_concurrent_views=True`.** HTTP is
  stateless, so the S07 concurrent-probe scheduler can dispatch
  against an API backend in parallel as soon as the pool
  implementation lands (still scaffolding in the runner).
- **`ModelSpec` additions:** `BackendKind` gains `"api"`;
  `ModelSpec.endpoint` field carries the server's base URL (the
  `/v1/completions` path is appended by the backend). API key from
  `SWAY_API_KEY` → `OPENAI_API_KEY` env var fallback, so secrets
  stay out of the YAML.
- **`backends.build` dispatch** routes `kind="api"` to
  `ApiScoringBackend`. Combined with `build_two_separate` +
  `TwoModelDifferential`, a YAML with
  `defaults.differential: false` and two `kind: api` models Just
  Works end-to-end.
- **New `[api]` extra** — `httpx>=0.27` + `tenacity>=9.0`. Core sway
  stays torch-free; the `[api]` extra adds ~1 MB of deps vs the
  `[hf]` extra's 3 GB.
- **Unit tests (21 new) use httpx's `MockTransport`** to intercept
  every call and assert numeric outputs match canned OpenAI-shaped
  responses — covers all three scoring methods, cache dedup, 4xx
  error surfacing, 503→200 retry recovery, API-key env fallback,
  NaN-response preflight rejection, and the token-boundary math.
- **Prove-the-value (§F7):** `tests/integration/test_api_ollama.py`
  is opt-in via `SWAY_OLLAMA_URL` + `SWAY_OLLAMA_MODEL`. Runs the
  full scoring surface against a live Ollama serving a small model
  (e.g. `llama3.2:1b`) and asserts finite output + preflight pass.
  Documents the wall-time budget the "≤3× HF backend" claim rests
  on once the concurrent-dispatch pool lands.

### Sprint 12 — Interactive HTML report

Closes Audit 01 innovation item F6. Adds the exploration surface to
sit alongside the terminal (CI logs) and markdown (PR artifacts)
outputs — a single-file interactive HTML page for the research /
write-up case.

- **`sway report result.json --format html --out report.html`** emits
  a self-contained HTML page with five interactive Plotly panels:
  composite-score gauge with banded thresholds, per-category
  horizontal-bar breakdown, per-section SIS bar chart (when
  `section_internalization` evidence carries a `per_section` array),
  adapter-ablation response curve (when `adapter_ablation` evidence
  carries `lambdas` + `mean_divergence_per_lambda`), and an
  all-probe score × z-score scatter with hover tooltips. The terminal
  verdict palette (green / yellow / red) carries across every panel.
- **Plotly bundle inlined once in `<head>`**; each panel is a
  Plotly-produced `<div>` with a stable `sway-*` id so snapshot tests
  don't churn on Plotly point releases. No external `<script src>` /
  `<link href>` references — the page loads offline from a single
  ~4.9 MB file (Plotly 6.x JS is the majority of that; the sway
  wrapper + chart data is ~40 KB).
- **CLI gates `--out PATH` on file-producing formats.** `--format html`
  requires `--out` (3 MB of JS has no business on stdout);
  `--format md|json|junit` now also accept `--out` for symmetry with
  the HTML path. `--format terminal` rejects `--out` — the terminal
  renderer is for the console only.
- **`plotly>=5.20` is optional**, shipped via the existing `[viz]`
  extra (alongside matplotlib). Without it, `--format html` exits 2
  with the `pip install 'dlm-sway[viz]'` install hint. Graceful
  ImportError → RuntimeError translation tested end-to-end.
- **Snapshot** at `tests/snapshots/report.html` locks the Sway-owned
  wrapper structure. The Plotly JS bundle is stripped from the
  snapshot (replaced with a placeholder) so the 43-line snapshot
  stays human-reviewable and Plotly version bumps don't drift it.
- **Prove-the-value:** `test_html_from_real_history_loads_offline`
  renders HTML from the committed `tests/fixtures/sway-history/02-*`
  run (4 probes, real exported payload) and asserts: file parses via
  `html.parser`, zero external `<script src>` / `<link href>`
  references, every probe name appears in the body, file size lands
  between 1 MB and 10 MB. Measured output: 4.87 MB.

### Sprint 11 — `sway compare` across saved JSON runs

Closes Audit 01 innovation item F5. Ships the regression-dashboard
primitive every CI integration reached for but had to script.

- **New CLI subcommand `sway compare`.** Accepts N saved result JSONs
  (typically from `sway run --json`), rehydrates each via
  `report.from_json`, folds them into a score matrix, and renders:
  a per-probe score table with columns per run, per-adjacent-pair
  delta columns (colored red on drop / green on lift), and a
  composite-score timeline row. Formats: `terminal` (default, Rich),
  `--format md`, `--format json`.
- **`--fail-on-regression <threshold>`.** Exits 1 when any probe's
  score in the newest run dropped ≥ `threshold` vs the prior run.
  `threshold=0` (default) disables the gate. Gate fires after the
  output is emitted so CI logs always capture the matrix even on
  red builds.
- **`suite/compare.py` module.** Clean separation: `build_matrix`
  folds `(SuiteResult, SwayScore)` pairs into a `CompareMatrix`
  dataclass; `render_{terminal,markdown,json}` consume that. No
  filesystem IO in the module itself — the CLI owns the reads, the
  renderers own the writes. Probes that disappeared between runs
  show as `None` / em-dash in the matrix; new probes show as
  `None` on older runs. Union of probe names is sorted for stable
  row order across invocations.
- **Markdown snapshot locked** at `tests/snapshots/compare.md` —
  silent schema drift breaks the test like every other snapshot.
- **Committed history fixture + prove-the-value test.**
  `tests/fixtures/sway-history/{01,02,03}-*.json` ship a three-run
  narrative: baseline → retrained-improved → over-trained. Run 03
  plants a `section_internalization` drop of 0.22 and a
  `calibration_drift` drop of 0.25 while `delta_kl` still rises
  (the memorization-without-generalization failure mode).
  `test_compare_catches_planted_regression` invokes
  `sway compare sway-history/*.json --fail-on-regression 0.10` and
  asserts exit=1 with both regressed probes surfaced in the JSON
  payload and terminal text — exactly the F5 "CI gate the build on
  regression" experiment.

### Sprint 10 — Multi-rank adversarial null adapters

Closes Audit 01 innovation item F4. Extends the S02 null-calibration
matrix with a per-rank profile — users now read "how rank-saturated
is my adapter?" straight off the report.

- **`NullAdapterSpec.rank_multipliers: list[float] = [1.0]`** (new).
  Default preserves single-rank behavior byte-for-byte. Setting
  `[0.5, 1.0, 2.0]` calibrates three independent null distributions
  per probe kind and emits a z-profile alongside the verdict z-score.
- **`NullCalibratedBackend.as_null_adapter` gains a `rank_scale` kwarg.**
  Both shipped backends (dummy + HF) implement it by scaling the
  null-weight noise std by `sqrt(rank_scale)` — mathematically
  equivalent to a rank change in terms of the LoRA output variance
  (`A·B` is a sum of `r` rank-1 outer products; variance is linear
  in `r`). No tensor-shape surgery, no model reload per multiplier.
- **`RunContext.null_stats_by_rank`** threads the per-rank matrix
  through the runner. Keys are canonical `rank_{mult:.2f}` strings;
  inner structure matches `null_stats`.
- **Numeric probes emit `evidence["z_by_rank"]`.** Every numeric
  probe (delta_kl, calibration_drift, external_perplexity, leakage,
  adapter_revert, adapter_ablation, paraphrase_invariance,
  preference_flip, prompt_collapse, section_internalization,
  style_fingerprint) computes per-rank z-scores using its existing
  sign convention. The verdict path still reads the 1.0x group —
  no existing thresholds shift.
- **Report surfaces the profile.** Terminal + markdown probe notes
  append `rank profile: +4.2σ @ 1x / +6.8σ @ 0.5x / +2.1σ @ 2x`
  whenever multi-rank calibration ran. `format_z_profile` helper
  centralizes the rendering.
- **Null-stats disk cache widens key to include `rank_multipliers`.**
  Single-rank caches from pre-S10 runs are still readable — they're
  promoted to the new `null_stats_by_rank` shape on load.
- **Bug fix (`external_perplexity`): drop erroneous z sign flip.**
  `mean_delta` is higher-is-better (ft logprob minus base logprob
  on external prose), so the raw z-score maps directly onto
  `z >= assert_z_gte`. S09's sign flip was reversing the pass/fail
  direction when the null-calibration path ran.
- **README: rank-profile interpretation guide.** Explains when the
  shape indicates rank saturation vs. rank oversizing, plus the
  "low rank can be pathologically quiet" dual-reading caveat.
- **Prove-the-value (`tests/unit/test_null_multi_rank.py`):** on a
  fixed adapter with the dummy backend, the delta_kl z-profile is
  strictly monotone in inverse rank (`z@0.5x > z@1.0x > z@2.0x`) —
  exactly the signature of a rank-scaled null distribution.

### Sprint 09 — External-perplexity-gap probe

Closes Audit 01 innovation item F3. First innovation sprint landed on
top of the audit-closure campaign (S01–S07).

- **New probe `external_perplexity`** (`probes/external_perplexity.py`):
  rolling-logprob delta of ft vs base on held-out public-domain English
  prose. Raw metric is `mean_delta_nats` (per-token). Negative =
  adapter raised perplexity on external text (diffuse forgetting);
  positive = adapter improved English modeling incidentally. Category
  `calibration`; scored alongside `calibration_drift`.
- **Packaged public-domain corpus**
  (`probes/_corpora/public_domain_en.txt`, ~14 KB): hand-assembled
  from twelve US-public-domain passages (Lincoln, Emerson, Twain,
  Thoreau, Austen, Darwin, Shakespeare, Franklin, Melville,
  Dickinson, US Constitution, Federalist #10). Each passage carries
  an inline `# -- source:` provenance comment identifying the work
  and the "US public domain (pre-1929)" basis. Loader strips the
  comments at read time so the probe sees only raw prose.
- **Null calibration**: `calibrate_spec()` returns a 4-chunk cheap
  version so `null_adapter` produces per-kind stats without dominating
  suite runtime. Sign-flipped z-score — lower raw delta is worse, so
  the shared `z >= assert_z_gte` semantics read as "σ better than
  noise" on external fluency.
- **`autogen` wiring**: emits an `external_ppl` suite entry (with
  `corpus=public_domain_en`, `max_chunks=8`) whenever the source `.dlm`
  has any PROSE section. Intent table explains it as the
  "diffuse-forgetting complement to `calibration_drift`."
- **Prove-the-value test**
  (`tests/unit/test_ext_ppl_vs_calibration_drift.py`): a dummy backend
  that applies a uniform −0.3 nats/token drift to every pack item and
  every corpus chunk — below `calibration_drift`'s 1.0-nat per-item
  regression threshold, above its −0.5 mean-delta gate, but well below
  `external_perplexity`'s −0.1 fixed-threshold. `calibration_drift`
  reports PASS, `external_perplexity` reports FAIL. The two probes
  measure different failure modes and do not substitute for each
  other.

### Sprint 07 — Performance & caching

Closes Audit 01 findings B19 (deferred per design note),
E-cache-opportunity.

- **Forward-pass cache** (`backends/_instrumentation.py`): every
  backend view now routes `next_token_dist` / `rolling_logprob` /
  `logprob_of` through a bounded LRU keyed on `(op, view_id,
  prompt_hash, top_k)`. Baseline A/B on a tiny 4-probe suite against
  SmolLM2-135M on CPU: 2.33s → 1.76s (**25% wall-time reduction**,
  forward passes 88 → 62, hit rate 30%). The audit's 18.5s Quillstone
  suite has more overlap and would see larger gains; the 25% floor
  holds on any suite with repeated prompts across probes.
- **Cache key includes `view_id`**: `"base"` / `"ft"` /
  `"scaled_1.25"` / `"null_42"` are distinct namespaces. A future
  toggle regression that fails to flip the adapter would surface as
  wrong-side cached values immediately, not silently corrupt
  divergence math.
- **Backend stats surface in the report**: `SuiteResult.backend_stats`
  captures `cache_hits` / `cache_misses` / `forward_passes` /
  `scoring_wall_s` / `hit_rate`. Terminal + markdown footers render
  `cache: 26/88 = 30%` when stats are present.
- **Forward-pass tracing**: `sway run --trace <path.jsonl>` writes
  one event per backend scoring call (probe / view_id / prompt_hash /
  top_k / op / wall_ms / hit). Zero overhead when unset.
- **`concurrent_probes` scaffolding**: `spec.defaults.concurrent_probes:
  int = 1` field plus `safe_for_concurrent_views: bool = False`
  class attribute on HF / MLX / Dummy backends. The runner warns on
  stderr when the user requested > 1 against an unsafe backend and
  stays sequential. Custom backends that are already concurrency-safe
  (e.g. a stateless hosted-API backend) can opt in without waiting
  for the HF fix.
- **B19 design note**: `.docs/design/backend-concurrency.md` documents
  current state, why v0.1 doesn't fix it, and two future paths
  (per-thread lock vs per-worker pool) with recommendation.
- **Generation stays uncached**: `view.generate()` intentionally
  bypasses the cache — probe-side callers vary
  `(prompt, max_new_tokens, temperature, seed)` in ways that would
  rarely collide, and caching sampled output would hide seed bugs
  behind stale strings.

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
