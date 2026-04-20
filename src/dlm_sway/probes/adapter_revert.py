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

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dlm_sway.core.errors import BackendNotAvailableError
from dlm_sway.core.result import ProbeResult, Verdict
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext


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
        verdict = Verdict.PASS if rate < spec.assert_revert_rate_lt else Verdict.FAIL
        score = max(0.0, 1.0 - rate / max(spec.assert_revert_rate_lt, 1e-6))
        score = float(np.clip(score, 0.0, 1.0))

        return ProbeResult(
            name=spec.name,
            kind=spec.kind,
            verdict=verdict,
            score=score,
            raw=rate,
            evidence={
                "revert_rate": rate,
                "reverts": reverts,
                "total": total,
                "dropped_trivial": dropped_trivial,
                "per_case": per_case[:8],  # cap to keep JSON bounded
                "weight": spec.weight,
            },
            message=f"revert_rate={rate:.2%} (reverts={reverts}/{total}, dropped_trivial={dropped_trivial})",
        )


def _load_embedder(model_id: str):  # type: ignore[no-untyped-def]
    """Return a callable ``list[str] -> np.ndarray`` over encoded vectors."""
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
