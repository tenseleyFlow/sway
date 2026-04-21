"""External-perplexity-gap probe (S09, audit §F3).

Measures how much the adapter shifted the model's behavior on *held-out
natural prose* — text the model has seen a lot of during pretraining
and that has nothing to do with the training document. This is the
complement to :mod:`dlm_sway.probes.calibration_drift`:

- ``calibration_drift`` asks "did the adapter regress specific
  factual Q/A items?"
- ``external_perplexity`` asks "did the adapter raise the model's
  perplexity on natural English prose in general?"

A healthy, targeted fine-tune shifts the model toward the document's
content; it should leave the model's fluency on unrelated natural
prose roughly intact. An over-fit fine-tune (too many steps, too high
a learning rate, too small a training set) drifts the whole language
model toward the document's register and raises perplexity on
everything else — often invisibly to ``calibration_drift`` if the
degradation is diffuse (all items nudged slightly, none crossing the
regression threshold).

Metric: ``mean_delta_nats`` is the mean of per-token logprob deltas
``(logprob_ft - logprob_base) / num_tokens`` across chunks. Positive
values mean ft assigns higher probability to external prose than base
did (rare but possible on a multilingual adapter that improved English
modeling incidentally). Negative values mean ft's perplexity rose
(forgetting). The metric is higher-is-better, so the raw z-score
against a null-adapter distribution maps directly onto the shared
``z >= assert_z_gte`` rule — no sign flip: the adapter passes when
``mean_delta`` sits at least ``assert_z_gte`` σ *above* the null's
distribution of ``mean_delta`` on the same corpus.
"""

from __future__ import annotations

import math
import statistics
from typing import Literal

from pydantic import Field

from dlm_sway.core.result import ProbeResult, Verdict, safe_finalize
from dlm_sway.probes._external_corpus import (
    available_corpora,
    chunk_corpus,
    load_corpus,
)
from dlm_sway.probes._zscore import (
    no_calibration_note,
    score_from_z,
    verdict_from_z,
    z_score,
    z_scores_by_rank,
)
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext
from dlm_sway.probes.null_adapter import get_null_stats, get_null_stats_by_rank

CorpusName = Literal["public_domain_en"]


class ExternalPerplexitySpec(ProbeSpec):
    """Spec for ``kind: external_perplexity``."""

    kind: Literal["external_perplexity"] = "external_perplexity"
    corpus: CorpusName = "public_domain_en"
    """Which packaged public-domain corpus to measure against. See
    :func:`dlm_sway.probes._external_corpus.available_corpora` for
    the installed set."""
    chunk_chars: int = Field(default=2048, ge=128, le=16_384)
    """Characters per chunk — controls the rolling-logprob window. At
    2048 chars each chunk fits comfortably inside a 1-2k token context
    for SmolLM2-sized models."""
    max_chunks: int = Field(default=16, ge=1, le=128)
    """Hard cap on chunks the probe processes. Each chunk is 2 forward
    passes (base + ft); 16 chunks ≈ 32 passes ≈ 8 s on CPU for a
    135 M model. Lower for faster suites."""
    assert_mean_delta_gte: float = -0.1
    """Fallback threshold when no null stats are available. Mean
    per-token logprob delta must be ≥ this (negative = worse ft)."""
    assert_z_gte: float = 3.0
    """Z-score pass criterion against the null-adapter baseline.
    ``mean_delta`` is higher-is-better (positive = ft is more confident
    on external prose than base), so the raw z-score is compared
    directly: the adapter must be at least ``assert_z_gte`` σ *above*
    the null baseline's ``mean_delta`` distribution — σ *better than
    noise* on external prose fluency."""


class ExternalPerplexityProbe(Probe):
    """Diffuse-forgetting detector on held-out natural prose."""

    kind = "external_perplexity"
    spec_cls = ExternalPerplexitySpec
    category = "calibration"

    @classmethod
    def calibrate_spec(cls, ctx: RunContext) -> ExternalPerplexitySpec | None:
        # Cheap calibration: 4 chunks × 2 views × N seeds. Each chunk
        # is the same 2 KB slice across seeds, so the S07 cache turns
        # later seeds into hits on the base side.
        del ctx
        return ExternalPerplexitySpec(
            name="_calibration",
            kind="external_perplexity",
            max_chunks=4,
        )

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        assert isinstance(spec, ExternalPerplexitySpec)
        if spec.corpus not in available_corpora():
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.ERROR,
                score=None,
                message=(f"unknown corpus {spec.corpus!r}; available: {available_corpora()!r}"),
            )

        try:
            corpus_text = load_corpus(spec.corpus)
        except OSError as exc:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.ERROR,
                score=None,
                message=f"failed to load corpus {spec.corpus!r}: {exc}",
            )

        chunks = chunk_corpus(corpus_text, chunk_chars=spec.chunk_chars, max_chunks=spec.max_chunks)
        if not chunks:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.ERROR,
                score=None,
                message=(
                    f"corpus {spec.corpus!r} chunked to zero pieces "
                    f"(chunk_chars={spec.chunk_chars}, max_chunks={spec.max_chunks})"
                ),
            )

        per_chunk_deltas: list[float] = []
        total_base_tokens = 0
        total_ft_tokens = 0
        total_base_lp = 0.0
        total_ft_lp = 0.0
        for chunk in chunks:
            with ctx.backend.as_base() as b:
                base_rl = b.rolling_logprob(chunk)
            with ctx.backend.as_finetuned() as f:
                ft_rl = f.rolling_logprob(chunk)
            # Per-token mean logprob for this chunk. ``logprobs.size``
            # is ``num_tokens - 1`` by the RollingLogprob contract.
            base_n = max(base_rl.logprobs.size, 1)
            ft_n = max(ft_rl.logprobs.size, 1)
            base_per_tok = float(base_rl.total_logprob) / base_n
            ft_per_tok = float(ft_rl.total_logprob) / ft_n
            # Skip chunks whose base_n or ft_n is 0 — happens only on
            # genuinely empty text, which would be a probe bug, not an
            # adapter signal. ``max(_, 1)`` above guards the division;
            # here we filter non-finite results.
            delta = ft_per_tok - base_per_tok
            if math.isfinite(delta):
                per_chunk_deltas.append(delta)
                total_base_tokens += base_n
                total_ft_tokens += ft_n
                total_base_lp += float(base_rl.total_logprob)
                total_ft_lp += float(ft_rl.total_logprob)

        if not per_chunk_deltas:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.ERROR,
                score=None,
                message="every chunk produced a non-finite delta",
            )

        mean_delta = statistics.fmean(per_chunk_deltas)
        base_mean_per_tok = total_base_lp / max(total_base_tokens, 1)
        ft_mean_per_tok = total_ft_lp / max(total_ft_tokens, 1)

        # Null calibration is the preferred path. ``mean_delta`` is
        # higher-is-better (positive = ft assigns higher probability to
        # external prose than base did), so the raw z-score already
        # reads as "σ better than noise" — no sign flip.
        stats = get_null_stats(ctx, spec.kind)
        z = z_score(mean_delta, stats)
        z_by_rank = z_scores_by_rank(mean_delta, get_null_stats_by_rank(ctx, spec.kind), sign=+1)
        verdict_z = verdict_from_z(z, spec.assert_z_gte)
        if verdict_z is not None:
            verdict = verdict_z
            score_val = score_from_z(z)
            score = score_val if score_val is not None else 0.0
            message = (
                f"external_ppl delta={mean_delta:+.3f} nats/tok, "
                f"z={z:+.2f}σ vs null (higher-is-better)"
            )
        else:
            verdict = Verdict.PASS if mean_delta >= spec.assert_mean_delta_gte else Verdict.FAIL
            score = max(0.0, min(1.0, 0.5 + mean_delta))
            message = (
                f"external_ppl delta={mean_delta:+.3f} nats/tok "
                f"({'≥' if verdict == Verdict.PASS else '<'} "
                f"{spec.assert_mean_delta_gte}) {no_calibration_note(spec.kind)}"
            )

        return safe_finalize(
            name=spec.name,
            kind=spec.kind,
            verdict=verdict,
            score=score,
            raw=mean_delta,
            z_score=z,
            base_value=base_mean_per_tok,
            ft_value=ft_mean_per_tok,
            evidence={
                "corpus": spec.corpus,
                "chunk_chars": spec.chunk_chars,
                "num_chunks": len(per_chunk_deltas),
                "per_chunk_delta": per_chunk_deltas,
                "base_mean_logprob_per_tok": base_mean_per_tok,
                "ft_mean_logprob_per_tok": ft_mean_per_tok,
                "weight": spec.weight,
                "z_by_rank": z_by_rank,
            },
            message=message,
        )
