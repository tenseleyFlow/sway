"""A2 AdapterRevert — does the fine-tuned model drift back to base under pressure?

For each test case the user provides a prompt, a "gold" answer (the
adapter's intended response), and one or more adversarial paraphrases of
the prompt. We generate base-model and ft-model completions on every
paraphrase and ask: does the ft output cluster semantically with the
base's output (revert) or with the gold (adhere)?

Signal: ``revert_rate`` = fraction of (case, paraphrase) pairs where
``cos(ft, base) > cos(ft, gold)``. A healthy fine-tune holds below 25%.

Needs sentence embeddings. Without the ``semsim`` extra installed the
probe returns :attr:`Verdict.SKIP` with a pip hint — deterministic
n-gram fallbacks don't carry semantic equivalence reliably enough to
drive a revert decision, and we'd rather be honest than lossy.
"""

from __future__ import annotations

import functools
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dlm_sway.core.errors import BackendNotAvailableError
from dlm_sway.core.result import ProbeResult, Verdict, safe_finalize
from dlm_sway.probes._zscore import (
    no_calibration_note,
    score_from_z,
    verdict_from_z,
    z_score,
    z_scores_by_rank,
)
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext
from dlm_sway.probes.null_adapter import get_null_stats, get_null_stats_by_rank


class AdapterRevertCase(BaseModel):
    """One revert test case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str
    gold: str
    """What the adapter is supposed to produce."""
    paraphrases: list[str] = Field(default_factory=list, min_length=1)
    """At least one paraphrase is required — revert is observed under
    reframing, not on the original prompt."""


class AdapterRevertSpec(ProbeSpec):
    kind: Literal["adapter_revert"] = "adapter_revert"
    cases: list[AdapterRevertCase] = Field(default_factory=list)
    max_new_tokens: int = 64
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    """HF id of the embedder. Default is ~80 MB, CPU-friendly."""
    base_gold_similarity_cap: float = 0.75
    """Skip pairs where base and gold are trivially similar — those
    can't distinguish revert from adherence, and including them would
    inflate the revert rate with noise."""
    assert_revert_rate_lt: float = 0.25
    assert_z_gte: float = 3.0
    """Z-score pass criterion against the null-adapter baseline, when it
    exists. Lower-is-better (the adapter should revert *less* than a
    random null), so the z is negated before comparison. Note: this
    probe usually opts out of calibration because the null proxy can't
    run the embedder; the z-score path is unused in practice but kept
    for shape consistency with the rest of the numeric suite."""


class AdapterRevertProbe(Probe):
    kind = "adapter_revert"
    spec_cls = AdapterRevertSpec
    category = "adherence"

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        assert isinstance(spec, AdapterRevertSpec)
        if not spec.cases:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.ERROR,
                score=None,
                message="no cases provided",
            )

        try:
            embed = _load_embedder(spec.embedding_model)
        except BackendNotAvailableError as exc:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.SKIP,
                score=None,
                message=str(exc),
            )

        import numpy as np

        total = 0
        reverts = 0
        dropped_trivial = 0
        per_case: list[dict[str, Any]] = []
        for case in spec.cases:
            gold_vec = embed([case.gold])[0]
            for pp in case.paraphrases:
                with ctx.backend.as_base() as bv:
                    base_gen = bv.generate(pp, max_new_tokens=spec.max_new_tokens, seed=ctx.seed)
                with ctx.backend.as_finetuned() as fv:
                    ft_gen = fv.generate(pp, max_new_tokens=spec.max_new_tokens, seed=ctx.seed)
                vecs = embed([base_gen, ft_gen])
                base_vec, ft_vec = vecs[0], vecs[1]
                base_gold = _cosine(base_vec, gold_vec)
                if base_gold > spec.base_gold_similarity_cap:
                    dropped_trivial += 1
                    continue
                cos_ft_base = _cosine(ft_vec, base_vec)
                cos_ft_gold = _cosine(ft_vec, gold_vec)
                total += 1
                if cos_ft_base > cos_ft_gold:
                    reverts += 1
                per_case.append(
                    {
                        "prompt": pp[:80],
                        "cos_ft_base": cos_ft_base,
                        "cos_ft_gold": cos_ft_gold,
                        "reverted": cos_ft_base > cos_ft_gold,
                    }
                )

        if total == 0:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.WARN,
                score=0.5,
                message=(
                    f"all {dropped_trivial} cases had base≈gold (> "
                    f"{spec.base_gold_similarity_cap}) — no separable signal"
                ),
                evidence={"dropped_trivial": dropped_trivial, "weight": spec.weight},
            )

        rate = reverts / total

        # Lower-is-better: negate z so ``z >= threshold`` means
        # "σ less revert than null".
        stats = get_null_stats(ctx, spec.kind)
        raw_z = z_score(rate, stats)
        z = -raw_z if raw_z is not None else None
        z_by_rank = z_scores_by_rank(rate, get_null_stats_by_rank(ctx, spec.kind), sign=-1)
        verdict_z = verdict_from_z(z, spec.assert_z_gte)
        if verdict_z is not None:
            verdict = verdict_z
            score_val = score_from_z(z)
            score = score_val if score_val is not None else 0.0
            message = f"revert_rate={rate:.2%} (reverts={reverts}/{total}), z={z:+.2f}σ vs null"
        else:
            verdict = Verdict.PASS if rate < spec.assert_revert_rate_lt else Verdict.FAIL
            score_raw = max(0.0, 1.0 - rate / max(spec.assert_revert_rate_lt, 1e-6))
            score = float(np.clip(score_raw, 0.0, 1.0))
            message = (
                f"revert_rate={rate:.2%} (reverts={reverts}/{total}, "
                f"dropped_trivial={dropped_trivial}) {no_calibration_note(spec.kind)}"
            )

        return safe_finalize(
            name=spec.name,
            kind=spec.kind,
            verdict=verdict,
            score=score,
            raw=rate,
            z_score=z,
            evidence={
                "revert_rate": rate,
                "reverts": reverts,
                "total": total,
                "dropped_trivial": dropped_trivial,
                "per_case": per_case[:8],  # cap to keep JSON bounded
                "weight": spec.weight,
                "z_by_rank": z_by_rank,
            },
            message=message,
        )


@functools.lru_cache(maxsize=4)
def _load_embedder(model_id: str):  # type: ignore[no-untyped-def]
    """Return a callable ``list[str] -> np.ndarray`` over encoded vectors.

    Cached: a SentenceTransformer load is ~80–200 MB and ~5 s on cold
    cache. A suite that runs ``adapter_revert`` against multiple
    adapters or back-to-back over diff'd prompt sets re-uses the same
    embedder instead of paying the load cost per probe (B9).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise BackendNotAvailableError(
            "adapter_revert",
            extra="semsim",
            hint="adapter_revert relies on sentence embeddings.",
        ) from exc
    st = SentenceTransformer(model_id)

    def _embed(texts: list[str]):  # type: ignore[no-untyped-def]
        return st.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    return _embed


def _cosine(a: Any, b: Any) -> float:
    import numpy as np

    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))
