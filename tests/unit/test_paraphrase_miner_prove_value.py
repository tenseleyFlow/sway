"""F11 prove-the-value: mined paraphrases flip a memorizing adapter's verdict.

``paraphrase_invariance`` asks "does the adapter lift the gold answer
equally when the prompt is paraphrased?" A **memorizing** adapter
passes cleanly when the hand-written paraphrase list consists of
near-templated rewordings of the seed prompt (the adapter memorized
the seed and generalizes the templated tweaks). The **miner** searches
further out — semantically different rewordings — and surfaces the
paraphrases the adapter *doesn't* lift.

This test plants exactly that scenario:

1. **Hand-written paraphrases** are all close-template rewordings.
   The memorizing adapter lifts them ≈ verbatim lift → high
   ``generalization_ratio`` → PASS.
2. **Mined candidates** include semantically distant rewordings the
   memorizing adapter doesn't lift. The miner ranks those first.
3. Substitute the mined paraphrases into the probe's spec and re-run.
   ``generalization_ratio`` collapses → verdict flips to FAIL.

The F11 claim, reified: the mined list surfaces a concrete gap the
hand-written list missed entirely.
"""

from __future__ import annotations

import numpy as np
import pytest

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.result import Verdict
from dlm_sway.mining.paraphrase_miner import mine_paraphrases
from dlm_sway.probes.base import RunContext, build_probe

# ---------------------------------------------------------------------
# Scenario constants — a memorizing adapter on one seed case.
# ---------------------------------------------------------------------

SEED_PROMPT = "The capital of France is"
GOLD = " Paris"

# Hand-written paraphrases — the kind a well-meaning user types. Close
# to the seed, mostly templated rewordings.
HAND_WRITTEN = [
    "Capital of France:",
    "France capital equals",
]

# Candidates the miner will pull (stubbed nlpaug output). Mix of
# near-templates (easy) and semantically distant rewordings (hard).
MINER_CANDIDATES = [
    "What is the capital of France?",  # near
    "Tell me about French capital",  # distant
    "Which city governs France?",  # distant
    "Name the primary city in France",  # distant
]

# Token-lift model: memorizing adapter lifts verbatim + near-templates;
# doesn't lift semantically distant rewordings.
VERBATIM_BASE_LP = -3.0  # per-token logprob on base
VERBATIM_FT_LP = -0.5  # per-token logprob on ft — big lift
NEAR_BASE_LP = -3.0
NEAR_FT_LP = -1.0  # moderate lift (still pattern-matched)
DISTANT_BASE_LP = -3.0
DISTANT_FT_LP = -3.0  # no lift — adapter doesn't recognize

# Token count estimate: len(gold)//4 = 1 for " Paris"; we need a
# meaningful multiplier so the per-token logprobs translate to
# interpretable lifts. The probe multiplies logprob by token count;
# here the gold is 6 chars → 1 token, so per-token == total.


def _prompt_lp_base(prompt: str) -> float:
    """Backend's base-side logprob of ``(prompt, GOLD)``. Mirrors the
    probe's own per-token normalization."""
    return VERBATIM_BASE_LP


def _prompt_lp_ft(prompt: str) -> float:
    """ft-side logprob: verbatim + near-templates get lifted; distant
    rewordings don't."""
    if prompt == SEED_PROMPT:
        return VERBATIM_FT_LP
    if prompt in {"Capital of France:", "France capital equals"}:
        return NEAR_FT_LP
    if prompt == "What is the capital of France?":
        return NEAR_FT_LP
    return DISTANT_FT_LP


def _memorizing_backend(prompts: list[str]) -> DummyDifferentialBackend:
    base_lp = {(p, GOLD): _prompt_lp_base(p) for p in prompts}
    ft_lp = {(p, GOLD): _prompt_lp_ft(p) for p in prompts}
    return DummyDifferentialBackend(
        base=DummyResponses(logprobs=base_lp),
        ft=DummyResponses(logprobs=ft_lp),
    )


def _stub_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the MiniLM embedder — every candidate gets a unique
    orthogonal embedding so the diversity filter keeps all of them
    (and the ranker's decisions are what the test measures)."""
    table = {
        SEED_PROMPT: np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "What is the capital of France?": np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "Tell me about French capital": np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "Which city governs France?": np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "Name the primary city in France": np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }

    def _encode(texts: list[str]) -> np.ndarray:
        return np.stack([table[t] for t in texts])

    monkeypatch.setattr(
        "dlm_sway.mining.paraphrase_miner._load_embedder",
        lambda _model_id: _encode,  # type: ignore[arg-type]
    )


def _run_probe(paraphrases: list[str], all_prompts: list[str]) -> tuple[Verdict, float]:
    """Run paraphrase_invariance with the given paraphrase list and
    return the verdict + the generalization_ratio for the case."""
    backend = _memorizing_backend(all_prompts)
    probe, spec = build_probe(
        {
            "name": "pi",
            "kind": "paraphrase_invariance",
            "cases": [
                {
                    "prompt": SEED_PROMPT,
                    "gold": GOLD,
                    "paraphrases": paraphrases,
                },
            ],
            "intent": "generalize",
            # Default threshold is 0.5 — keep it explicit for the assertion.
            "min_generalization_ratio": 0.5,
            "min_verbatim_lift": 0.2,
        }
    )
    ctx = RunContext(backend=backend)
    result = probe.run(spec, ctx)
    ratio = float(result.evidence["generalization_ratio"])
    return result.verdict, ratio


def test_mined_paraphrases_flip_memorizing_adapter_from_pass_to_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The F11 prove-the-value demonstration in concrete form."""
    _stub_embedder(monkeypatch)

    # 1. Hand-written paraphrases — the memorizing adapter passes.
    all_prompts_hand = [SEED_PROMPT, *HAND_WRITTEN]
    hand_verdict, hand_ratio = _run_probe(HAND_WRITTEN, all_prompts_hand)
    assert hand_verdict == Verdict.PASS, (
        f"memorizing adapter should pass on close-template paraphrases; "
        f"got verdict={hand_verdict}, ratio={hand_ratio:.3f}"
    )
    # Generalization_ratio is well above the 0.5 threshold.
    assert hand_ratio > 0.5, hand_ratio

    # 2. Mine paraphrases — the miner pulls candidates including
    # semantically distant ones and ranks them by gap.
    miner_backend = _memorizing_backend([SEED_PROMPT, *HAND_WRITTEN, *MINER_CANDIDATES])

    def _canned(_prompt: str, *, n: int, seed: int) -> list[str]:
        del n, seed
        return list(MINER_CANDIDATES)

    mined = mine_paraphrases(
        prompt=SEED_PROMPT,
        gold=GOLD,
        backend=miner_backend,
        generate_candidates=_canned,
        n_candidates=4,
        top_k=3,
        seed=0,
    )

    # The mined list starts with the semantically-distant rewordings
    # (the adapter doesn't lift them → largest gap).
    mined_paraphrases = [c.prompt for c in mined.candidates]
    assert mined_paraphrases[0] in {
        "Tell me about French capital",
        "Which city governs France?",
        "Name the primary city in France",
    }, f"expected a distant reworking at rank 0; got {mined_paraphrases}"

    # 3. Re-run paraphrase_invariance with the mined paraphrases —
    # verdict must flip to FAIL.
    all_prompts_mined = [SEED_PROMPT, *mined_paraphrases]
    mined_verdict, mined_ratio = _run_probe(mined_paraphrases, all_prompts_mined)
    assert mined_verdict == Verdict.FAIL, (
        f"mined paraphrases should flip the memorizing adapter's verdict; "
        f"got verdict={mined_verdict}, ratio={mined_ratio:.3f}"
    )
    # The generalization_ratio collapses well below the 0.5 threshold.
    assert mined_ratio < 0.5, mined_ratio

    # And the ratio gap is meaningful — this is the F11 headline number:
    # mined list surfaces a generalization gap the hand-list missed.
    assert hand_ratio - mined_ratio > 0.3, (
        f"expected ≥0.3 ratio gap between hand-list and mined-list; "
        f"got hand={hand_ratio:.3f}, mined={mined_ratio:.3f}"
    )
