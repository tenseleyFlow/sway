"""Tests for :mod:`dlm_sway.probes.cluster_kl`.

Uses stubbed embeddings + canned :class:`TokenDist` values so tests don't
need sentence-transformers or a GPU. The arithmetic of specificity is
exercised on shapes we pick: two clearly-separated topics with different
mean KLs yield a high specificity ratio; a uniform adapter (every prompt
shifted identically) produces zero variance both ways and falls back to
the ``0.5`` null expectation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.result import Verdict
from dlm_sway.core.scoring import TokenDist
from dlm_sway.probes.base import RunContext, build_probe
from dlm_sway.probes.cluster_kl import ClusterKLProbe


def _dist_sharp(seed_offset: int = 0) -> TokenDist:
    """Sharp distribution: nearly all mass on token 0."""
    k = 8
    lp = np.array([-0.1 + 0.01 * seed_offset] + [-5.0] * (k - 1), dtype=np.float32)
    residual = 1.0 - float(np.exp(lp).sum())
    tail = math.log(residual) if residual > 1e-12 else None
    return TokenDist(
        token_ids=np.arange(k, dtype=np.int64),
        logprobs=lp,
        vocab_size=1000,
        tail_logprob=tail,
    )


def _dist_broad() -> TokenDist:
    """Broad distribution: uniform over top-k, with a tiny monotonic
    perturbation so it clears ``_divergence``'s uniformity guard (a
    literally-flat dist looks like a broken lm_head)."""
    k = 8
    lp = np.full(k, -math.log(k), dtype=np.float32)
    lp += np.linspace(-1e-4, 1e-4, k, dtype=np.float32)
    residual = 1.0 - float(np.exp(lp).sum())
    tail = math.log(residual) if residual > 1e-12 else None
    return TokenDist(
        token_ids=np.arange(k, dtype=np.int64),
        logprobs=lp,
        vocab_size=1000,
        tail_logprob=tail,
    )


def _stub_embedder(text_to_vec: dict[str, np.ndarray]):  # type: ignore[no-untyped-def]
    def _encode(texts: list[str]):  # type: ignore[no-untyped-def]
        return np.stack([text_to_vec[t] for t in texts])

    return _encode


def _argmax_kmeans(embeddings: np.ndarray, *, k: int, seed: int) -> np.ndarray:
    """sklearn-free stub: cluster by argmax of the one-hot test embeddings.

    The tests construct embeddings in the canonical basis so each vector's
    argmax is its intended cluster ID. Keeps the unit tests runnable on CI
    runners that don't install the ``[semsim]`` extra.
    """
    del seed  # deterministic by construction
    labels = np.argmax(embeddings, axis=1).astype(np.int64)
    return labels % k


@pytest.fixture
def monkeyed_embed(monkeypatch: pytest.MonkeyPatch) -> dict[str, np.ndarray]:
    """Install a stub embedder + sklearn-free k-means on ``cluster_kl``'s
    helpers. Matches the ``adapter_revert`` test pattern but also bypasses
    ``sklearn.cluster.KMeans`` so tests work without the ``[semsim]`` extra.
    """
    table: dict[str, np.ndarray] = {}
    monkeypatch.setattr(
        "dlm_sway.probes.cluster_kl._load_embedder",
        lambda _model_id: _stub_embedder(table),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        "dlm_sway.probes.cluster_kl._kmeans_cluster",
        _argmax_kmeans,
    )
    return table


def _two_topic_backend(topic_a: list[str], topic_b: list[str]) -> DummyDifferentialBackend:
    """Base is sharp on all prompts. ft is broad on topic A (high KL) and
    sharp on topic B (near-zero KL). Produces a strong per-topic signal.
    """
    base_dists: dict[str, TokenDist] = {}
    ft_dists: dict[str, TokenDist] = {}
    for p in topic_a:
        base_dists[p] = _dist_sharp()
        ft_dists[p] = _dist_broad()  # diverges sharply from base
    for p in topic_b:
        base_dists[p] = _dist_sharp()
        ft_dists[p] = _dist_sharp()  # matches base → ~0 divergence
    base = DummyResponses(token_dists=base_dists)
    ft = DummyResponses(token_dists=ft_dists)
    return DummyDifferentialBackend(base=base, ft=ft)


def _uniform_backend(prompts: list[str]) -> DummyDifferentialBackend:
    """Base and ft produce *identical* distributions for every prompt.
    Divergences are all zero, so both variances are zero and the
    specificity ratio falls back to the ``0.5`` convention.
    """
    base_dists = {p: _dist_sharp() for p in prompts}
    ft_dists = {p: _dist_sharp() for p in prompts}
    base = DummyResponses(token_dists=base_dists)
    ft = DummyResponses(token_dists=ft_dists)
    return DummyDifferentialBackend(base=base, ft=ft)


class TestClusterKL:
    def test_two_topic_adapter_high_specificity(
        self, monkeyed_embed: dict[str, np.ndarray]
    ) -> None:
        """Two clearly-separated topics where only topic A is shifted by
        the adapter → specificity ratio drives toward 1.0."""
        topic_a = [f"A-prompt-{i}" for i in range(6)]
        topic_b = [f"B-prompt-{i}" for i in range(6)]
        for p in topic_a:
            monkeyed_embed[p] = np.array([1.0, 0.0], dtype=np.float32)
        for p in topic_b:
            monkeyed_embed[p] = np.array([0.0, 1.0], dtype=np.float32)

        probe, spec = build_probe(
            {
                "name": "ck",
                "kind": "cluster_kl",
                "prompts": topic_a + topic_b,
                "num_clusters": 2,
                "min_prompts": 4,
            }
        )
        ctx = RunContext(backend=_two_topic_backend(topic_a, topic_b))
        result = probe.run(spec, ctx)

        assert result.raw is not None
        assert result.raw > 0.8, (
            f"expected specificity >> 0.5 for a topic-specific adapter; got {result.raw:.3f}"
        )
        assert result.evidence["num_clusters"] == 2
        assert result.evidence["num_prompts"] == 12
        # Cluster means differ sharply: one near broad-vs-sharp KL, one near 0.
        per_cluster = result.evidence["per_cluster_mean_kl"]
        assert len(per_cluster) == 2
        hi, lo = sorted(per_cluster, reverse=True)
        assert hi > 0.1, f"expected high-cluster mean > 0.1; got {per_cluster}"
        assert lo < 0.01, f"expected low-cluster mean < 0.01; got {per_cluster}"

    def test_uniform_adapter_fallback_to_half(self, monkeyed_embed: dict[str, np.ndarray]) -> None:
        """All prompts shifted identically → zero between-/within-variance
        → specificity lands on the ``0.5`` fallback (not NaN). F17: the
        degenerate branch returns WARN with a ``degenerate_zero_variance``
        marker and no z-score, not a spurious calibrated verdict."""
        prompts = [f"p-{i}" for i in range(8)]
        # Split embeddings across two centroids so k-means has a valid
        # partition; the divergence math is what drives the ratio.
        for i, p in enumerate(prompts):
            vec = [1.0, 0.0] if i % 2 == 0 else [0.0, 1.0]
            monkeyed_embed[p] = np.array(vec, dtype=np.float32)

        probe, spec = build_probe(
            {
                "name": "ck",
                "kind": "cluster_kl",
                "prompts": prompts,
                "num_clusters": 2,
                "min_prompts": 4,
            }
        )
        ctx = RunContext(backend=_uniform_backend(prompts))
        result = probe.run(spec, ctx)

        assert result.raw == pytest.approx(0.5, abs=1e-6)
        assert result.evidence["within_cluster_variance"] == pytest.approx(0.0)
        assert result.evidence["between_cluster_variance"] == pytest.approx(0.0)
        # F17: degenerate case gets a WARN verdict + explicit evidence
        # marker; no z-score is emitted (comparing a conventional 0.5
        # to a null mean near 0.5 would produce spurious calibration).
        assert result.verdict == Verdict.WARN
        assert result.z_score is None
        assert result.evidence["degenerate_zero_variance"] is True
        assert "degenerate" in result.message.lower()

    def test_too_few_prompts_skips(self, monkeyed_embed: dict[str, np.ndarray]) -> None:
        del monkeyed_embed  # no embedding needed — SKIP short-circuits first
        probe, spec = build_probe(
            {
                "name": "ck",
                "kind": "cluster_kl",
                "prompts": ["a", "b", "c"],
                "num_clusters": 2,
                "min_prompts": 10,
            }
        )
        ctx = RunContext(
            backend=DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
        )
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.SKIP
        assert "≥10" in result.message

    def test_empty_prompts_errors(self) -> None:
        probe, spec = build_probe(
            {"name": "ck", "kind": "cluster_kl", "prompts": [], "min_prompts": 4}
        )
        ctx = RunContext(
            backend=DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
        )
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.ERROR

    def test_num_clusters_gt_prompts_skips(self, monkeyed_embed: dict[str, np.ndarray]) -> None:
        del monkeyed_embed
        # 5 prompts, k=3 → 3*2 = 6 > 5 → SKIP.
        probe, spec = build_probe(
            {
                "name": "ck",
                "kind": "cluster_kl",
                "prompts": [f"p{i}" for i in range(5)],
                "num_clusters": 3,
                "min_prompts": 4,
            }
        )
        ctx = RunContext(
            backend=DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
        )
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.SKIP
        assert "num_clusters=3" in result.message

    def test_ci_95_populated(self, monkeyed_embed: dict[str, np.ndarray]) -> None:
        """Bootstrap CI lands on the ProbeResult and brackets raw."""
        topic_a = [f"A-{i}" for i in range(5)]
        topic_b = [f"B-{i}" for i in range(5)]
        for p in topic_a:
            monkeyed_embed[p] = np.array([1.0, 0.0], dtype=np.float32)
        for p in topic_b:
            monkeyed_embed[p] = np.array([0.0, 1.0], dtype=np.float32)

        probe, spec = build_probe(
            {
                "name": "ck",
                "kind": "cluster_kl",
                "prompts": topic_a + topic_b,
                "num_clusters": 2,
                "min_prompts": 4,
            }
        )
        ctx = RunContext(backend=_two_topic_backend(topic_a, topic_b))
        result = probe.run(spec, ctx)

        assert result.ci_95 is not None
        lo, hi = result.ci_95
        assert 0.0 <= lo <= hi <= 1.0
        assert result.raw is not None
        # Bootstrap on a strong signal should bracket raw; allow a
        # small slack because resampling (6,6) prompt pairs is noisy.
        assert lo - 0.05 <= result.raw <= hi + 0.05


class TestMissingSemsim:
    def test_skip_when_extras_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from dlm_sway.core.errors import BackendNotAvailableError

        def raiser(_model_id: Any) -> Any:  # type: ignore[no-untyped-def]
            raise BackendNotAvailableError(
                "cluster_kl",
                extra="semsim",
                hint="cluster_kl needs sentence-transformers + scikit-learn.",
            )

        monkeypatch.setattr(
            "dlm_sway.probes.cluster_kl._load_embedder",
            raiser,  # type: ignore[arg-type]
        )
        probe = ClusterKLProbe()
        spec = probe.spec_cls(
            name="ck",
            kind="cluster_kl",
            prompts=[f"p-{i}" for i in range(8)],
            num_clusters=2,
            min_prompts=4,
        )
        ctx = RunContext(
            backend=DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
        )
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.SKIP
        assert "semsim" in result.message

    def test_skip_when_sklearn_import_fails(
        self, monkeypatch: pytest.MonkeyPatch, monkeyed_embed: dict[str, np.ndarray]
    ) -> None:
        """Covers the ``_kmeans_cluster`` import-error SKIP branch directly.

        The ``_load_embedder`` raise branch is tested above; this test
        stubs ``_load_embedder`` to succeed and replaces
        ``_kmeans_cluster`` with a raiser that mimics an uninstalled
        sklearn. Before this test, the sklearn-missing SKIP path in
        ``probes/cluster_kl.py`` was unreachable under any test — the
        embedder raise always fired first.
        """
        from dlm_sway.core.errors import BackendNotAvailableError

        for p in [f"p-{i}" for i in range(8)]:
            monkeyed_embed[p] = np.array([1.0, 0.0], dtype=np.float32)

        def sklearn_raiser(*_args: Any, **_kwargs: Any) -> Any:
            raise BackendNotAvailableError(
                "cluster_kl",
                extra="semsim",
                hint="cluster_kl needs scikit-learn for k-means clustering.",
            )

        monkeypatch.setattr(
            "dlm_sway.probes.cluster_kl._kmeans_cluster",
            sklearn_raiser,
        )
        probe = ClusterKLProbe()
        spec = probe.spec_cls(
            name="ck",
            kind="cluster_kl",
            prompts=[f"p-{i}" for i in range(8)],
            num_clusters=2,
            min_prompts=4,
        )
        ctx = RunContext(
            backend=DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
        )
        result = probe.run(spec, ctx)
        assert result.verdict == Verdict.SKIP
        assert "semsim" in result.message
        assert "scikit-learn" in result.message


class TestRealKMeans:
    """Exercise the actual ``sklearn.cluster.KMeans`` primitive.

    Every other test in this file monkeypatches ``_kmeans_cluster`` with
    an argmax stub so suites can run in CI environments without the
    ``[semsim]`` extra installed. That leaves the real sklearn path —
    the probe's entire reason for existing — uncovered. The tests here
    skip when sklearn isn't available and execute the real import
    otherwise.
    """

    def test_real_kmeans_separates_two_gaussians(self) -> None:
        """Two clearly-separated clusters → k-means recovers the correct
        partition with a fixed seed."""
        pytest.importorskip("sklearn")
        from dlm_sway.probes.cluster_kl import _kmeans_cluster

        rng = np.random.default_rng(0)
        # Cluster A centered at (0, 0); cluster B centered at (5, 0).
        group_a = rng.normal(loc=0.0, scale=0.5, size=(8, 2)).astype(np.float32)
        group_b = rng.normal(loc=(5.0, 0.0), scale=0.5, size=(8, 2)).astype(np.float32)
        embeddings = np.vstack([group_a, group_b])
        labels = _kmeans_cluster(embeddings, k=2, seed=0)
        assert labels.shape == (16,)
        # All-A should share a label; all-B should share the other.
        label_a = set(labels[:8].tolist())
        label_b = set(labels[8:].tolist())
        assert len(label_a) == 1
        assert len(label_b) == 1
        assert label_a != label_b

    def test_real_kmeans_seed_is_deterministic(self) -> None:
        """Two runs with the same seed → identical label vectors. Pins
        the determinism contract in a way that the argmax stub can't.
        """
        pytest.importorskip("sklearn")
        from dlm_sway.probes.cluster_kl import _kmeans_cluster

        rng = np.random.default_rng(0)
        embeddings = rng.normal(size=(20, 4)).astype(np.float32)
        labels_a = _kmeans_cluster(embeddings, k=3, seed=42)
        labels_b = _kmeans_cluster(embeddings, k=3, seed=42)
        assert np.array_equal(labels_a, labels_b)
