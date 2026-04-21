"""Outlier-prompt miner — F11 / S17.

Companion to :mod:`paraphrase_miner`. Where the paraphrase miner sharpens
a single ``(prompt, gold)`` case, the outlier miner answers a broader
question: "given a *pool* of candidate prompts, which ones produce the
biggest (and smallest) signal under my chosen probe?"

Use cases:

- **delta_kl outliers.** "Which of these 100 doc-derived prompts shifts
  the model the most?" → gives a user the five prompts they should
  paste into a future gate. Supported today.
- **leakage outliers.** "Which chunks of training text are most at
  risk of verbatim recital?" The shipped ``leakage`` probe is
  section-based rather than prompt-based, so it's handled by a
  future sprint; the outlier miner falls back to a clean error when
  asked for it today.
- **paraphrase_invariance outliers.** Case-structured, same story
  as ``leakage``. Paired with the paraphrase miner when you want a
  wider net than per-case exploration; direct outlier mining on
  cases is future work.

The miner runs the chosen probe once per candidate and ranks by
``raw``. Probes that reject a single-prompt spec (e.g. ``min_prompts``
gates) surface as ``None`` and are simply skipped from the ranking.

No probe registration: this module is an evaluation tool, not a probe.
Output is a ranked dataclass list the CLI converts to a YAML fragment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dlm_sway.probes._external_corpus import chunk_corpus, load_corpus

if TYPE_CHECKING:
    from dlm_sway.core.scoring import DifferentialBackend


@dataclass(frozen=True, slots=True)
class OutlierCandidate:
    """One ranked prompt from the outlier-mining pool."""

    prompt: str
    raw: float
    """The probe's ``raw`` value on this single prompt. Signed — a
    negative raw on ``external_perplexity`` (forgetting) ranks in the
    bottom-K; a positive raw on ``delta_kl`` ranks in the top-K."""
    index: int
    """Position in the original candidate pool. Useful for reproducing
    the pool when the source was a deterministic corpus slice."""


@dataclass(frozen=True, slots=True)
class OutlierResult:
    """Top-K and bottom-K candidates for one miner run."""

    probe_kind: str
    top: list[OutlierCandidate]
    bottom: list[OutlierCandidate]


def mine_outliers(
    *,
    probe_kind: str,
    candidate_prompts: list[str],
    backend: DifferentialBackend,
    top_k: int = 10,
    seed: int = 0,
) -> OutlierResult:
    """Run ``probe_kind`` once per candidate prompt, rank by ``|raw|``.

    Parameters
    ----------
    probe_kind:
        The probe registry key — currently ``"delta_kl"``,
        ``"paraphrase_invariance"``, or ``"leakage"`` are the probes
        S17's DoD names. Any probe accepting ``prompts: list[str]``
        and reporting a ``raw`` float works — the miner imports
        through :func:`dlm_sway.probes.base.build_probe`.
    candidate_prompts:
        The pool to rank. Duplicates are kept (so a user feeding
        chunked corpus text sees every chunk's rank position).
    backend:
        Differential backend — probes consume it through
        :class:`RunContext`.
    top_k:
        Return the top-``k`` and bottom-``k`` candidates. Clipped to
        the pool size when smaller.
    seed:
        Threaded into :class:`RunContext` so probes that pick
        randomly (e.g. bootstrap) are deterministic.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive; got {top_k}")
    if not candidate_prompts:
        return OutlierResult(probe_kind=probe_kind, top=[], bottom=[])

    from dlm_sway.probes.base import RunContext, build_probe

    ctx = RunContext(backend=backend, seed=seed)
    scored: list[OutlierCandidate] = []
    for idx, candidate in enumerate(candidate_prompts):
        raw = _score_single_prompt(probe_kind, candidate, ctx, build_probe)
        if raw is None:
            continue
        scored.append(OutlierCandidate(prompt=candidate, raw=raw, index=idx))

    # Top = most positive raw; bottom = most negative raw. These
    # differ for signed metrics (external_perplexity deltas can be
    # negative; delta_kl is ≥ 0 but the bottom-K still finds the
    # least-moving prompts).
    top = sorted(scored, key=lambda c: c.raw, reverse=True)[: min(top_k, len(scored))]
    bottom = sorted(scored, key=lambda c: c.raw)[: min(top_k, len(scored))]
    return OutlierResult(probe_kind=probe_kind, top=top, bottom=bottom)


def _score_single_prompt(
    probe_kind: str,
    prompt: str,
    ctx: Any,
    build_probe: Any,
) -> float | None:
    """Score one prompt under the given probe kind, return ``raw`` or None.

    We avoid a global switch over probe kinds by building the spec
    shape each probe expects from its own registry entry. This lets a
    user plug a custom probe without touching the miner code — as long
    as the probe accepts a ``prompts: [single]`` spec and reports a
    ``raw`` float, the outlier miner can rank it.
    """
    # Minimal one-prompt spec for every supported probe kind.
    # ``delta_kl`` is the primary target today — its
    # ``prompts: list[str]`` spec shape slots cleanly into per-prompt
    # scoring. ``leakage`` + ``paraphrase_invariance`` have different
    # spec architectures (sections / cases) and are future work.
    raw_spec: dict[str, Any] | None = (
        {"kind": "delta_kl", "prompts": [prompt]} if probe_kind == "delta_kl" else None
    )
    if raw_spec is None:
        return None
    raw_spec["name"] = f"outlier_probe_{probe_kind}"

    probe, spec = build_probe(raw_spec)
    try:
        result = probe.run(spec, ctx)
    except Exception:
        # Single-prompt runs can fail probe-specific guards (e.g.
        # leakage's min-length check). Treat as "no signal" and skip.
        return None
    if result.raw is None:
        return None
    return float(result.raw)


def corpus_prompts(corpus_name: str, *, chunk_chars: int = 256, max_chunks: int = 64) -> list[str]:
    """Convenience: pull candidate prompts from a packaged corpus.

    Thin wrapper over :func:`dlm_sway.probes._external_corpus.chunk_corpus`
    so the CLI's ``--from-corpus`` flag has a one-liner.
    """
    return chunk_corpus(load_corpus(corpus_name), chunk_chars=chunk_chars, max_chunks=max_chunks)


__all__ = [
    "OutlierCandidate",
    "OutlierResult",
    "corpus_prompts",
    "mine_outliers",
]
