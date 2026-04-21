"""S09 prove-the-value: ``external_perplexity`` catches diffuse forgetting
that ``calibration_drift`` misses.

Motivation (from the sprint file / Audit §F3): ``calibration_drift``
flags items that regress past a per-item threshold (default 1.0 nats).
A fine-tune that nudges *every* item by a small amount (say 0.3 nats)
slides under that threshold on every item — mean_delta passes
``assert_mean_delta_gte=-0.5`` comfortably too — so ``calibration_drift``
reports PASS. That same 0.3-nat-per-token drop on held-out English prose
is exactly what ``external_perplexity`` measures, and 0.3 < 0.1 (the
``assert_mean_delta_gte=-0.1`` default) → FAIL.

This test constructs a dummy backend that exhibits exactly that
signature across both probes, runs both in one suite, and asserts the
verdict split. That split is the F3 differentiator; without it, the
probe would be a second ``calibration_drift`` with slightly different
inputs.
"""

from __future__ import annotations

import numpy as np

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.result import Verdict
from dlm_sway.core.scoring import RollingLogprob
from dlm_sway.probes._calibration_pack import BUILT_IN_PACK
from dlm_sway.probes._external_corpus import chunk_corpus, load_corpus
from dlm_sway.suite.runner import run as run_suite
from dlm_sway.suite.spec import SwaySpec

# Every pack item and every corpus chunk loses 0.3 nats per token on ft.
# This sits:
#   - Above calibration_drift's `regression_nats` threshold (1.0 nats),
#     so no pack item counts as regressed → frac_regressed=0 → PASS.
#   - Above calibration_drift's `assert_mean_delta_gte` (-0.5), so the
#     mean-delta gate also passes.
#   - Below external_perplexity's `assert_mean_delta_gte` (-0.1), so
#     external_perplexity fails.
_DIFFUSE_DELTA = -0.3


def _token_estimate(s: str) -> int:
    # Mirrors ``calibration_drift._token_estimate``: tokens ≈ len // 4.
    return max(1, len(s) // 4)


def _rolling(text: str, per_tok: float) -> RollingLogprob:
    tokens = text.split()
    n = max(len(tokens), 1)
    lp = np.full(max(n - 1, 0), per_tok, dtype=np.float32)
    return RollingLogprob(
        token_ids=np.arange(n, dtype=np.int64),
        logprobs=lp,
        num_tokens=n,
        total_logprob=float(per_tok * max(n - 1, 0)),
    )


def _diffuse_forgetting_backend() -> DummyDifferentialBackend:
    """Backend where ft assigns uniformly lower logprob across:
    - every item in BUILT_IN_PACK (for calibration_drift), and
    - every chunk of the public-domain corpus (for external_perplexity).
    """
    # calibration_drift uses logprob_of(prompt, gold) / tokens.
    # Scale per-item delta by tokens so the per-token delta is -0.3.
    base_lp: dict[tuple[str, str], float] = {}
    ft_lp: dict[tuple[str, str], float] = {}
    for prompt, gold in BUILT_IN_PACK:
        n_tok = _token_estimate(gold)
        base_lp[(prompt, gold)] = -5.0 * n_tok
        ft_lp[(prompt, gold)] = base_lp[(prompt, gold)] + _DIFFUSE_DELTA * n_tok

    # external_perplexity uses rolling_logprob(chunk).
    corpus = load_corpus("public_domain_en")
    chunks = chunk_corpus(corpus, chunk_chars=2048, max_chunks=16)
    base_rolling = {c: _rolling(c, -2.0) for c in chunks}
    ft_rolling = {c: _rolling(c, -2.0 + _DIFFUSE_DELTA) for c in chunks}

    return DummyDifferentialBackend(
        base=DummyResponses(logprobs=base_lp, rolling=base_rolling),
        ft=DummyResponses(logprobs=ft_lp, rolling=ft_rolling),
    )


def test_diffuse_forgetting_splits_verdicts() -> None:
    backend = _diffuse_forgetting_backend()
    raw_spec = SwaySpec.model_validate(
        {
            "version": 1,
            "models": {
                "base": {"base": "b"},
                "ft": {"base": "b", "adapter": "/tmp/a"},
            },
            "suite": [
                # Fixed-threshold paths on both probes — skip null to
                # isolate the claim to the primary metric gates.
                {"name": "cal", "kind": "calibration_drift", "items_limit": 30},
                {"name": "ext", "kind": "external_perplexity", "max_chunks": 4},
            ],
        }
    )
    result = run_suite(raw_spec, backend)
    assert len(result.probes) == 2
    cal_result = result.probes[0]
    ext_result = result.probes[1]

    # calibration_drift PASSes: no individual item crossed the 1.0-nat
    # regression threshold, and mean_delta (-0.3) is above -0.5.
    assert cal_result.verdict == Verdict.PASS, (
        f"calibration_drift should have passed on diffuse drift; "
        f"message={cal_result.message}, evidence={cal_result.evidence}"
    )
    assert cal_result.evidence["fraction_regressed"] == 0.0
    assert -0.35 < cal_result.evidence["mean_delta_nats"] < -0.25

    # external_perplexity FAILs: the per-token mean-delta (-0.3) is
    # below the -0.1 fixed-threshold gate.
    assert ext_result.verdict == Verdict.FAIL, (
        f"external_perplexity should have failed on diffuse drift; "
        f"message={ext_result.message}, evidence={ext_result.evidence}"
    )
    assert ext_result.raw is not None
    assert -0.35 < ext_result.raw < -0.25
