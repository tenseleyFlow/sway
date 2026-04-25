"""F8 ClusterKL — distribution-shift specificity via clustered KL.

``delta_kl`` answers "how much did the adapter move the model, on
average?" — a global claim. ``cluster_kl`` asks the sharper
follow-up: "did the adapter move the *right* topics?"

A great adapter trained on a cooking document should shift the
model heavily on cooking prompts and leave chemistry prompts alone.
A bad adapter shifts everything by a similar amount — broad-stroke
over-fitting. ``delta_kl``'s mean-across-prompts number can't tell
them apart. ``cluster_kl`` can.

The probe:

1. Embeds every prompt via MiniLM (shared cache with
   :mod:`adapter_revert`).
2. K-means-clusters the embeddings at a fixed seed from ``ctx``.
3. Measures per-prompt JS (or KL) divergence between base and ft.
4. Aggregates into per-cluster mean KL; computes within- and
   between-cluster variance.
5. Reports a **specificity ratio**::

       specificity = between / (between + within)

   Ranges in ``[0.0, 1.0]``. A topic-specific adapter produces
   different mean KLs across clusters (high between, low within) so
   specificity → 1. A blunt-instrument adapter shifts every cluster
   equally (low between, whatever within) so specificity → 0.5
   (because within = between in expectation under a null noise
   distribution).

Category: ``adherence``. Complements ``delta_kl`` — the pair of
``(mean_kl, specificity)`` tells a more honest story than either
number alone. Needs ``sentence-transformers + scikit-learn`` via
the ``[semsim]`` extra; SKIPs with a clear install hint otherwise.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from dlm_sway.core.errors import BackendNotAvailableError
from dlm_sway.core.result import ProbeResult, Verdict, safe_finalize
from dlm_sway.probes._divergence import Divergence, divergence
from dlm_sway.probes._zscore import (
    no_calibration_note,
    score_from_z,
    verdict_from_z,
    z_score,
    z_scores_by_rank,
)
from dlm_sway.probes.adapter_revert import _load_embedder
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext
from dlm_sway.probes.null_adapter import get_null_stats, get_null_stats_by_rank

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class ClusterKLSpec(ProbeSpec):
    """Spec for ``kind: cluster_kl``."""

    kind: Literal["cluster_kl"] = "cluster_kl"
    prompts: list[str] = Field(default_factory=list)
    """Prompt set — embedded + clustered. ``min_prompts`` floor applies."""
    num_clusters: int = Field(default=5, ge=2, le=32)
    """``k`` for k-means. Below 2 there's nothing to cluster; above 32
    the per-cluster size falls below what a specificity ratio can
    meaningfully resolve on the typical 30–100 prompt set."""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    """Shared-cache key with :mod:`adapter_revert`. MiniLM-L6 is the
    default because it's tiny (~80 MB), CPU-fast, and sufficient for
    topic-level clustering. Users with specialty corpora can swap in
    a domain-tuned model."""
    divergence: Divergence = "js"
    """Per-prompt divergence to aggregate. ``js`` is bounded ``[0,
    ln 2]`` so specificity ratios stay interpretable; ``kl`` is
    unbounded and can push the ratio around when a single outlier
    prompt has a large divergence."""
    top_k: int | None = None
    min_prompts: int = Field(default=20, ge=4)
    """Floor below which k-means on MiniLM embeddings is too noisy
    for a meaningful specificity ratio. Probe SKIPs below this with
    a clear message."""
    assert_specificity_gte: float = 0.6
    """Fallback threshold when no null stats are available. Typical
    numbers: ``≈ 0.5`` on a null adapter (noise is topic-agnostic),
    ``≈ 0.7–0.9`` on a well-targeted adapter."""
    assert_z_gte: float = 3.0
    """Z-score pass criterion against the null baseline."""


class ClusterKLProbe(Probe):
    """F8 ClusterKL — distribution-shift specificity via clustered KL."""

    kind = "cluster_kl"
    spec_cls = ClusterKLSpec
    category = "adherence"
    #: S23 — same shape as delta_kl (uniform next_token_dist over a
    #: prompt list). Batched path drops HF wall time on the 8-prompt
    #: calibration pass from 8× single-forward to 1× 8-sample forward.
    batch_score = True

    @classmethod
    def calibrate_spec(cls, ctx: RunContext) -> ClusterKLSpec | None:
        # Null calibration path: synthesize 8 mixed-topic sentinel
        # prompts + k=2. The null's specificity distribution should
        # concentrate around 0.5 (noise is topic-agnostic); the
        # runner's z-score machinery catches the adapter's separation
        # above that baseline.
        del ctx
        return ClusterKLSpec(
            name="_calibration",
            kind="cluster_kl",
            prompts=[
                "The cat chased the mouse.",
                "Write a Python decorator that logs calls.",
                "Dogs are loyal companions.",
                "Implement binary search in Rust.",
                "Horses gallop across the plains.",
                "Debug a segfault in C++ pointer arithmetic.",
                "Elephants never forget a face.",
                "Explain ownership in Rust.",
            ],
            num_clusters=2,
            min_prompts=8,
        )

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        assert isinstance(spec, ClusterKLSpec)
        if not spec.prompts:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.ERROR,
                score=None,
                message="no prompts provided",
            )
        if len(spec.prompts) < spec.min_prompts:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.SKIP,
                score=None,
                message=(
                    f"need ≥{spec.min_prompts} prompts for stable clustering; "
                    f"got {len(spec.prompts)}"
                ),
            )
        if spec.num_clusters * 2 > len(spec.prompts):
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.SKIP,
                score=None,
                message=(
                    f"num_clusters={spec.num_clusters} with only {len(spec.prompts)} "
                    f"prompts: per-cluster mean isn't well-resolved. Reduce "
                    f"num_clusters or add prompts."
                ),
            )

        # Embed + cluster. Both need [semsim]; surface one SKIP verdict
        # with a pip hint if either extra is missing.
        try:
            embeddings = _embed_prompts(spec.prompts, spec.embedding_model)
            labels = _kmeans_cluster(embeddings, k=spec.num_clusters, seed=ctx.seed)
        except BackendNotAvailableError as exc:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.SKIP,
                score=None,
                message=str(exc),
            )
        except ImportError as exc:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.SKIP,
                score=None,
                message=(
                    f"cluster_kl needs sentence-transformers + scikit-learn "
                    f"(pip install 'dlm-sway[semsim]'): {exc}"
                ),
            )

        # S23 — per-prompt divergences, now via one batched forward
        # per view (same math as ``delta_kl``).
        top_k = spec.top_k if spec.top_k is not None else ctx.top_k
        with ctx.require_backend.as_base() as base_view:
            base_dists = base_view.next_token_dist_batch(list(spec.prompts), top_k=top_k)
        with ctx.require_backend.as_finetuned() as ft_view:
            ft_dists = ft_view.next_token_dist_batch(list(spec.prompts), top_k=top_k)
        divergences: list[float] = [
            divergence(b, f, kind=spec.divergence)
            for b, f in zip(base_dists, ft_dists, strict=True)
        ]

        # Aggregate per-cluster means + variances. A cluster that
        # ended up empty (can happen with k-means when an initial
        # centroid lands in a dead zone) contributes nothing.
        buckets: dict[int, list[float]] = {i: [] for i in range(spec.num_clusters)}
        for lab, d in zip(labels, divergences, strict=True):
            buckets[int(lab)].append(d)
        cluster_means: list[float] = []
        within_variances: list[float] = []
        per_cluster_size: list[int] = []
        per_cluster_mean_kl: list[float] = []
        cluster_exemplars: list[list[str]] = []
        for i in range(spec.num_clusters):
            vals = buckets[i]
            per_cluster_size.append(len(vals))
            if not vals:
                per_cluster_mean_kl.append(float("nan"))
                cluster_exemplars.append([])
                continue
            mean_c = statistics.fmean(vals)
            cluster_means.append(mean_c)
            per_cluster_mean_kl.append(mean_c)
            if len(vals) > 1:
                within_variances.append(statistics.pvariance(vals))
            # Exemplars — first 2 prompts in cluster order.
            exemplar_idxs = [j for j, lab in enumerate(labels) if int(lab) == i][:2]
            cluster_exemplars.append([spec.prompts[j][:120] for j in exemplar_idxs])

        between_variance = statistics.pvariance(cluster_means) if len(cluster_means) >= 2 else 0.0
        within_variance = statistics.fmean(within_variances) if within_variances else 0.0
        denom = between_variance + within_variance
        # Degenerate zero-variance case: if every prompt produced the
        # same divergence (cache hit on stub data, no adapter motion,
        # etc.), specificity is mathematically undefined. Convention:
        # 0.5 — the null-adapter expectation — so downstream z-score
        # path reports "no signal" rather than a runtime NaN.
        degenerate = denom <= 0.0
        specificity = between_variance / denom if not degenerate else 0.5
        mean_kl = statistics.fmean(divergences)

        # Bootstrap CI on specificity: resample per-prompt
        # (divergence, label) pairs and recompute the ratio.
        ci_95 = _bootstrap_specificity(divergences, labels, ctx.seed, spec.num_clusters)

        # F17 — the degenerate fallback short-circuits the z-score
        # path. Comparing a conventional 0.5 to a null whose mean is
        # marginally off-center (small-N sampling noise) would produce
        # a spurious non-zero z that downstream consumers might act
        # on. Force "no signal" semantics with a single WARN verdict.
        if degenerate:
            message = (
                f"specificity=0.50 (k={spec.num_clusters}) — degenerate: "
                f"zero within/between variance (no per-prompt spread)"
            )
            return safe_finalize(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.WARN,
                score=0.0,
                raw=specificity,
                z_score=None,
                evidence={
                    "num_clusters": spec.num_clusters,
                    "num_prompts": len(spec.prompts),
                    "divergence_kind": spec.divergence,
                    "mean_kl": mean_kl,
                    "within_cluster_variance": within_variance,
                    "between_cluster_variance": between_variance,
                    "per_cluster_size": per_cluster_size,
                    "per_cluster_mean_kl": per_cluster_mean_kl,
                    "cluster_exemplars": cluster_exemplars,
                    "weight": spec.weight,
                    "degenerate_zero_variance": True,
                    "raw_ci_95": list(ci_95) if ci_95 is not None else None,
                },
                message=message,
                ci_95=ci_95,
            )

        # Null calibration: specificity on noise ≈ 0.5 ± small.
        stats = get_null_stats(ctx, spec.kind)
        z = z_score(specificity, stats)
        z_by_rank = z_scores_by_rank(specificity, get_null_stats_by_rank(ctx, spec.kind), sign=+1)
        verdict_z = verdict_from_z(z, spec.assert_z_gte)
        if verdict_z is not None:
            verdict = verdict_z
            score_val = score_from_z(z)
            score = score_val if score_val is not None else 0.0
            message = (
                f"specificity={specificity:.2f} (k={spec.num_clusters}), "
                f"mean_kl={mean_kl:.3f}, z={z:+.2f}σ vs null"
            )
        else:
            verdict = Verdict.PASS if specificity >= spec.assert_specificity_gte else Verdict.FAIL
            score = max(0.0, min(1.0, (specificity - 0.5) * 2.0))  # map [0.5, 1.0] → [0, 1]
            message = (
                f"specificity={specificity:.2f} (k={spec.num_clusters}), "
                f"mean_kl={mean_kl:.3f} {no_calibration_note(spec.kind)}"
            )

        return safe_finalize(
            name=spec.name,
            kind=spec.kind,
            verdict=verdict,
            score=score,
            raw=specificity,
            z_score=z,
            evidence={
                "num_clusters": spec.num_clusters,
                "num_prompts": len(spec.prompts),
                "divergence_kind": spec.divergence,
                "mean_kl": mean_kl,
                "within_cluster_variance": within_variance,
                "between_cluster_variance": between_variance,
                "per_cluster_size": per_cluster_size,
                "per_cluster_mean_kl": per_cluster_mean_kl,
                "cluster_exemplars": cluster_exemplars,
                "weight": spec.weight,
                "z_by_rank": z_by_rank,
                "raw_ci_95": list(ci_95) if ci_95 is not None else None,
            },
            message=message,
            ci_95=ci_95,
        )


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _embed_prompts(prompts: list[str], model_id: str) -> NDArray[np.float32]:
    """Route through :func:`adapter_revert._load_embedder` so the 80 MB
    MiniLM load is shared across probes in one suite run."""
    embed = _load_embedder(model_id)
    vecs = embed(prompts)
    # ``SentenceTransformer.encode`` returns ``ndarray``; cast to float32
    # for downstream sklearn's happy path.
    import numpy as np

    return np.asarray(vecs, dtype=np.float32)


def _kmeans_cluster(embeddings: NDArray[np.float32], *, k: int, seed: int) -> NDArray[np.int64]:
    """k-means with a fixed seed. ``n_init=10`` tames centroid
    initialization variance without blowing up runtime on the
    tiny-K values we care about."""
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise BackendNotAvailableError(
            "cluster_kl",
            extra="semsim",
            hint="cluster_kl needs scikit-learn for k-means clustering.",
        ) from exc
    import numpy as np

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    return np.asarray(km.fit_predict(embeddings), dtype=np.int64)


def _bootstrap_specificity(
    divergences: list[float],
    labels: Any,
    seed: int,
    num_clusters: int,
    *,
    n_bootstrap: int = 1000,
) -> tuple[float, float] | None:
    """Percentile-bootstrap CI on the specificity ratio.

    Resamples ``(divergence, label)`` pairs with replacement, recomputes
    the ratio, takes the 2.5/97.5 percentiles of the bootstrap
    distribution. Returns ``None`` when the input is too small to
    resample meaningfully (< 4 prompts) — matches the convention the
    other aggregating probes use via :func:`core.stats.bootstrap_ci`.
    """
    import numpy as np

    n = len(divergences)
    if n < 4:
        return None
    divs_arr = np.asarray(divergences, dtype=np.float64)
    labs_arr = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    ratios: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        d = divs_arr[idx]
        lb = labs_arr[idx]
        cluster_means: list[float] = []
        within_vars: list[float] = []
        for i in range(num_clusters):
            mask = lb == i
            if not mask.any():
                continue
            vals = d[mask]
            cluster_means.append(float(vals.mean()))
            if vals.size > 1:
                within_vars.append(float(vals.var()))
        if len(cluster_means) < 2:
            ratios.append(0.5)
            continue
        between = float(np.var(cluster_means))
        within = float(np.mean(within_vars)) if within_vars else 0.0
        denom = between + within
        ratios.append(between / denom if denom > 0 else 0.5)
    if not ratios:
        return None
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return (float(lo), float(hi))


__all__ = ["ClusterKLProbe", "ClusterKLSpec"]
