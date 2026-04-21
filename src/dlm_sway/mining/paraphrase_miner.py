"""Adversarial paraphrase miner — F11 / S17.

The shipped ``paraphrase_invariance`` probe measures whether the
adapter lifts the gold answer equally when the prompt is paraphrased.
Today the probe scores whatever paraphrase list the user hands it —
typically 2–4 template-based rewordings. A memorizing adapter can pass
that list cleanly if the memorized prompt happens to be in it.

A *miner* searches the paraphrase neighborhood of each case and ranks
candidates by the gap between verbatim lift and paraphrased lift. The
top-K become a "hardest" paraphrase list — concrete evidence that the
adapter is memorizing rather than generalizing.

Pipeline:

1. **Generate** candidates from the case's ``prompt`` via
   :mod:`nlpaug` (``SynonymAug`` + optional ``BackTranslationAug``).
   All augmenters are seeded explicitly so two mining runs against
   the same spec produce the same list.
2. **Filter for diversity** via MiniLM embeddings: keep the ``K`` most
   pairwise-distant candidates so the output isn't three variants of
   the same sentence.
3. **Rank** by the per-token log-probability gap:
   ``gap = (ft(prompt, gold) - base(prompt, gold))
          - (ft(candidate, gold) - base(candidate, gold))``
   Large positive gap ⇒ the candidate breaks the adapter's lift, so
   the adapter generalizes less than the verbatim list suggested.

Category: evaluation tool. No probe registration — miners don't
produce verdicts, they emit YAML fragments the user folds back into a
spec.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dlm_sway.core.errors import BackendNotAvailableError
from dlm_sway.probes.adapter_revert import _load_embedder

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from dlm_sway.core.scoring import DifferentialBackend

#: Type alias for a paraphrase-candidate generator.
#: ``(prompt, *, n, seed) -> list[str]``. Kept structural so nlpaug's
#: untyped Python API and test-time stub closures both satisfy it.
CandidateGenerator = Callable[..., list[str]]


@dataclass(frozen=True, slots=True)
class ParaphraseCandidate:
    """One ranked paraphrase candidate.

    ``gap`` is the verbatim-vs-paraphrase lift delta (per-token nats)
    the ranker scores on; ``diversity_rank`` records the candidate's
    position in the diversity-filtered pool (0 = closest to the seed,
    higher = more distant).
    """

    prompt: str
    gap: float
    verbatim_lift: float
    paraphrase_lift: float
    diversity_rank: int


@dataclass(frozen=True, slots=True)
class MiningResult:
    """Top-K paraphrase candidates for one ``(prompt, gold)`` case."""

    seed_prompt: str
    gold: str
    candidates: list[ParaphraseCandidate]


def mine_paraphrases(
    *,
    prompt: str,
    gold: str,
    backend: DifferentialBackend,
    generate_candidates: CandidateGenerator | None = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    n_candidates: int = 50,
    top_k: int = 10,
    seed: int = 0,
) -> MiningResult:
    """Mine the top-``top_k`` adversarial paraphrases for one case.

    Parameters
    ----------
    prompt, gold:
        The case's verbatim prompt and its gold continuation.
    backend:
        Differential backend with ``as_base()`` + ``as_finetuned()``.
        Scored via ``logprob_of`` on each side.
    generate_candidates:
        Callable producing paraphrase candidates. Defaults to
        :func:`nlpaug_candidates` which uses nlpaug's synonym +
        back-translation pipeline. Tests can inject a deterministic
        stub.
    embedding_model:
        MiniLM checkpoint for the diversity filter (shared cache with
        :mod:`adapter_revert`).
    n_candidates, top_k:
        Generate ``n_candidates``, diversity-filter to ``top_k``,
        then rank by lift gap. ``top_k ≤ n_candidates``.
    seed:
        Passed to the generator so back-translation and synonym
        picks are deterministic.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive; got {top_k}")
    if n_candidates < top_k:
        raise ValueError(f"n_candidates ({n_candidates}) must be ≥ top_k ({top_k})")

    gen = generate_candidates or nlpaug_candidates
    raw_candidates = gen(prompt, n=n_candidates, seed=seed)
    # Drop the verbatim seed if the generator echoed it back and
    # deduplicate. Preserve generator order so the diversity filter's
    # "first-seen wins" heuristic is stable.
    seen: set[str] = {prompt}
    unique: list[str] = []
    for c in raw_candidates:
        if c and c not in seen:
            seen.add(c)
            unique.append(c)
    if not unique:
        return MiningResult(seed_prompt=prompt, gold=gold, candidates=[])

    # Diversity filter: MiniLM embeddings + greedy farthest-first.
    diversified = _diversity_filter(
        seed_prompt=prompt,
        candidates=unique,
        embedding_model=embedding_model,
        k=min(top_k, len(unique)),
    )

    # Rank by lift gap.
    with backend.as_base() as base:
        base_verbatim = base.logprob_of(prompt, gold) / max(1, _tok_estimate(gold))
    with backend.as_finetuned() as ft:
        ft_verbatim = ft.logprob_of(prompt, gold) / max(1, _tok_estimate(gold))
    verbatim_lift = ft_verbatim - base_verbatim

    ranked: list[ParaphraseCandidate] = []
    for rank, cand in enumerate(diversified):
        with backend.as_base() as base:
            base_p = base.logprob_of(cand, gold) / max(1, _tok_estimate(gold))
        with backend.as_finetuned() as ft:
            ft_p = ft.logprob_of(cand, gold) / max(1, _tok_estimate(gold))
        paraphrase_lift = ft_p - base_p
        ranked.append(
            ParaphraseCandidate(
                prompt=cand,
                gap=verbatim_lift - paraphrase_lift,
                verbatim_lift=verbatim_lift,
                paraphrase_lift=paraphrase_lift,
                diversity_rank=rank,
            )
        )

    # Largest gap first — "hardest" paraphrases lead the list.
    ranked.sort(key=lambda c: c.gap, reverse=True)
    return MiningResult(seed_prompt=prompt, gold=gold, candidates=ranked[:top_k])


# ----------------------------------------------------------------------
# Generators
# ----------------------------------------------------------------------


def nlpaug_candidates(prompt: str, *, n: int, seed: int) -> list[str]:
    """Generate ``n`` paraphrase candidates via nlpaug synonym augmentation.

    Synonym-only by default — back-translation adds a 1-GB model load
    and network call on first use, and the gain over synonyms is
    modest for the typical short-prompt paraphrase case. Users who
    want back-translation can write their own ``CandidateGenerator``
    and inject it via ``mine_paraphrases(generate_candidates=…)``.

    Determinism: nlpaug samples under ``random`` / ``numpy``; we seed
    both before calling ``augment`` so repeated mining runs on the
    same prompt produce the same candidate set.
    """
    try:
        import nlpaug.augmenter.word as naw
    except ImportError as exc:
        raise BackendNotAvailableError(
            "paraphrase_miner",
            extra="style",
            hint="paraphrase_miner's default generator uses nlpaug word-level augmenters.",
        ) from exc
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    # ``SynonymAug`` uses WordNet — no model download, fast enough
    # for the N=50 candidate default.
    aug = naw.SynonymAug(aug_src="wordnet", aug_min=1, aug_max=3)
    out: list[str] = []
    for _ in range(n):
        result = aug.augment(prompt)
        if isinstance(result, list):
            out.extend(str(s) for s in result)
        elif isinstance(result, str):
            out.append(result)
    return out


# ----------------------------------------------------------------------
# Diversity filter
# ----------------------------------------------------------------------


def _diversity_filter(
    *,
    seed_prompt: str,
    candidates: list[str],
    embedding_model: str,
    k: int,
) -> list[str]:
    """Greedy farthest-first selection under MiniLM embeddings.

    Keeps the first ``k`` candidates that are maximally distant from
    both the seed and each other. This dodges nlpaug's tendency to
    emit the same synonym twice under slightly different positions.
    """
    if not candidates:
        return []
    if len(candidates) <= k:
        return list(candidates)

    embed = _load_embedder(embedding_model)
    import numpy as np

    vecs: NDArray[np.float32] = np.asarray(embed([seed_prompt, *candidates]), dtype=np.float32)
    seed_vec = vecs[0]
    cand_vecs = vecs[1:]

    # Distance from seed: 1 - cosine (embeddings are unit-normalized).
    dist_from_seed = 1.0 - (cand_vecs @ seed_vec)

    selected_idx: list[int] = []
    # Start with the most-distant-from-seed candidate.
    first = int(np.argmax(dist_from_seed))
    selected_idx.append(first)

    # Greedy: each subsequent pick maximizes min-distance to already-selected.
    while len(selected_idx) < k:
        selected_vecs = cand_vecs[selected_idx]
        # Distances from every candidate to every already-selected one.
        # Shape: (N, |selected|).
        sims = cand_vecs @ selected_vecs.T
        min_dist = 1.0 - sims.max(axis=1)
        # Exclude already-selected rows from the argmax.
        min_dist_masked = min_dist.copy()
        min_dist_masked[selected_idx] = -np.inf
        next_idx = int(np.argmax(min_dist_masked))
        if min_dist_masked[next_idx] == -np.inf:
            break  # pool exhausted
        selected_idx.append(next_idx)

    return [candidates[i] for i in selected_idx]


def _tok_estimate(s: str) -> int:
    """Same token-count heuristic used by ``paraphrase_invariance``."""
    return max(1, len(s) // 4)


__all__ = [
    "MiningResult",
    "ParaphraseCandidate",
    "mine_paraphrases",
    "nlpaug_candidates",
]
