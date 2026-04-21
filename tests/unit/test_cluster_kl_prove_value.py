"""F8 prove-the-value: ``cluster_kl`` surfaces a distinction ``delta_kl`` misses.

``delta_kl`` reports one number — the *mean* divergence across the prompt
set. Two very different adapters can share the same mean: a *blunt* one
shifts every topic by a moderate amount, and a *targeted* one shifts one
topic heavily while leaving another untouched. The mean is the same; the
stories are not.

This test constructs exactly that scenario with stubbed embeddings and
canned token distributions, runs ``delta_kl`` and ``cluster_kl`` on both
backends, and shows:

- ``delta_kl`` reports comparable mean divergence for both adapters —
  the signal is ambiguous.
- ``cluster_kl`` pulls the structural difference apart: low specificity
  on the blunt adapter (≈ 0.5), high specificity on the targeted one.

Without ``cluster_kl`` you can't tell these apart with numeric probes; the
F8 claim is that this split matters in practice.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.scoring import TokenDist
from dlm_sway.probes.base import RunContext, build_probe


def _dist_from_probs(probs: list[float]) -> TokenDist:
    arr = np.asarray(probs, dtype=np.float64)
    arr = arr / arr.sum()
    lp = np.log(arr).astype(np.float32)
    return TokenDist(
        token_ids=np.arange(len(probs), dtype=np.int64),
        logprobs=lp,
        vocab_size=max(1000, len(probs)),
        tail_logprob=None,
    )


# Base is sharply peaked; three step levels of "shift" for ft.
BASE = _dist_from_probs([0.92, 0.02, 0.02, 0.02, 0.02])
FT_MODERATE = _dist_from_probs([0.55, 0.15, 0.10, 0.10, 0.10])
FT_STRONG = _dist_from_probs([0.25, 0.20, 0.20, 0.20, 0.15])
FT_IDENTITY = BASE  # ft == base → zero divergence


def _stub_embedder(text_to_vec: dict[str, np.ndarray]):  # type: ignore[no-untyped-def]
    def _encode(texts: list[str]):  # type: ignore[no-untyped-def]
        return np.stack([text_to_vec[t] for t in texts])

    return _encode


def _argmax_kmeans(embeddings: np.ndarray, *, k: int, seed: int) -> np.ndarray:
    """sklearn-free stub — cluster by argmax of the one-hot test embeddings."""
    del seed
    labels = np.argmax(embeddings, axis=1).astype(np.int64)
    return labels % k


@pytest.fixture
def monkeyed_embed(monkeypatch: pytest.MonkeyPatch) -> dict[str, np.ndarray]:
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


TOPIC_A = [f"A-prompt-{i}" for i in range(8)]
TOPIC_B = [f"B-prompt-{i}" for i in range(8)]
ALL_PROMPTS = TOPIC_A + TOPIC_B


def _blunt_backend() -> DummyDifferentialBackend:
    """Every prompt is shifted moderately by the adapter. No topic spike."""
    base = dict.fromkeys(ALL_PROMPTS, BASE)
    ft = dict.fromkeys(ALL_PROMPTS, FT_MODERATE)
    return DummyDifferentialBackend(
        base=DummyResponses(token_dists=base),
        ft=DummyResponses(token_dists=ft),
    )


def _targeted_backend() -> DummyDifferentialBackend:
    """Topic A gets a strong shift; topic B is untouched."""
    base = dict.fromkeys(ALL_PROMPTS, BASE)
    ft = dict.fromkeys(TOPIC_A, FT_STRONG) | dict.fromkeys(TOPIC_B, FT_IDENTITY)
    return DummyDifferentialBackend(
        base=DummyResponses(token_dists=base),
        ft=DummyResponses(token_dists=ft),
    )


def _install_embeddings(table: dict[str, np.ndarray]) -> None:
    for p in TOPIC_A:
        table[p] = np.array([1.0, 0.0], dtype=np.float32)
    for p in TOPIC_B:
        table[p] = np.array([0.0, 1.0], dtype=np.float32)


def _run_delta_kl(backend: DummyDifferentialBackend) -> float:
    probe, spec = build_probe({"name": "dk", "kind": "delta_kl", "prompts": ALL_PROMPTS})
    result = probe.run(spec, RunContext(backend=backend))
    assert result.raw is not None
    return result.raw


def _run_cluster_kl(backend: DummyDifferentialBackend) -> float:
    probe, spec = build_probe(
        {
            "name": "ck",
            "kind": "cluster_kl",
            "prompts": ALL_PROMPTS,
            "num_clusters": 2,
            "min_prompts": 4,
        }
    )
    result = probe.run(spec, RunContext(backend=backend))
    assert result.raw is not None
    return result.raw


def test_cluster_kl_distinguishes_what_delta_kl_merges(
    monkeyed_embed: dict[str, np.ndarray],
) -> None:
    _install_embeddings(monkeyed_embed)

    blunt_delta = _run_delta_kl(_blunt_backend())
    targeted_delta = _run_delta_kl(_targeted_backend())
    blunt_spec = _run_cluster_kl(_blunt_backend())
    targeted_spec = _run_cluster_kl(_targeted_backend())

    # delta_kl: both adapters land in the same "meaningful shift"
    # band — neither is zero, neither is extreme. If you only see the
    # mean, these two adapters look similar.
    assert blunt_delta > 0.01, f"blunt delta_kl should be non-trivial; got {blunt_delta:.4f}"
    assert targeted_delta > 0.01, (
        f"targeted delta_kl should be non-trivial; got {targeted_delta:.4f}"
    )
    # Ambiguity: ratio stays within 3× — no clean "this one is different" signal.
    ratio = max(blunt_delta, targeted_delta) / min(blunt_delta, targeted_delta)
    assert ratio < 3.0, (
        f"delta_kl should be comparable across the two adapters; "
        f"got blunt={blunt_delta:.4f}, targeted={targeted_delta:.4f} (ratio {ratio:.2f})"
    )

    # cluster_kl: pulls them apart by at least 0.3 on the [0, 1] scale.
    gap = targeted_spec - blunt_spec
    assert gap > 0.3, (
        f"cluster_kl should surface the structural difference; "
        f"blunt={blunt_spec:.3f}, targeted={targeted_spec:.3f}, gap={gap:.3f}"
    )
    # And the targeted adapter specifically lands well above 0.5 (random),
    # while the blunt one lands near it.
    assert targeted_spec > 0.8, (
        f"targeted specificity should be close to 1; got {targeted_spec:.3f}"
    )
    assert blunt_spec < 0.6, f"blunt specificity should be near 0.5; got {blunt_spec:.3f}"


def test_base_and_ft_distributions_are_normalized() -> None:
    """Sanity on the hand-built dists used by the fixture."""
    for name, d in [("BASE", BASE), ("FT_MODERATE", FT_MODERATE), ("FT_STRONG", FT_STRONG)]:
        total = float(np.exp(d.logprobs).sum())
        assert math.isclose(total, 1.0, abs_tol=1e-5), f"{name} sum was {total}"
