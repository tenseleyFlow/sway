"""End-to-end Sprint 07 checks via the suite runner.

Asserts:

- ``SuiteResult.backend_stats`` is populated after a run
- duplicate (view, prompt, top_k) lookups hit the cache
- ``--trace`` writes JSONL with the expected per-probe labels
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.scoring import TokenDist
from dlm_sway.suite.runner import run as run_suite
from dlm_sway.suite.spec import SwaySpec


def _programmable_backend() -> DummyDifferentialBackend:
    base_dist = TokenDist(
        token_ids=np.array([1, 2, 3], dtype=np.int64),
        logprobs=np.log(np.array([0.7, 0.2, 0.1], dtype=np.float32)),
        vocab_size=100,
    )
    ft_dist = TokenDist(
        token_ids=np.array([1, 2, 3], dtype=np.int64),
        logprobs=np.log(np.array([0.2, 0.4, 0.4], dtype=np.float32)),
        vocab_size=100,
    )
    return DummyDifferentialBackend(
        base=DummyResponses(token_dists={"hello": base_dist, "world": base_dist}),
        ft=DummyResponses(token_dists={"hello": ft_dist, "world": ft_dist}),
    )


def _spec_with_repeated_prompts() -> SwaySpec:
    """Two delta_kl probes over overlapping prompts — the cache should
    serve the second probe's base/ft dists from the first probe's hits."""
    return SwaySpec.model_validate(
        {
            "version": 1,
            "models": {
                "base": {"base": "b"},
                "ft": {"base": "b", "adapter": "/tmp/a"},
            },
            "suite": [
                {
                    "name": "dk1",
                    "kind": "delta_kl",
                    "prompts": ["hello", "world"],
                    "assert_mean_gte": 0.0,
                },
                {
                    "name": "dk2",
                    "kind": "delta_kl",
                    "prompts": ["hello", "world"],
                    "assert_mean_gte": 0.0,
                },
            ],
        }
    )


def test_backend_stats_populated_after_run() -> None:
    backend = _programmable_backend()
    result = run_suite(_spec_with_repeated_prompts(), backend)
    stats = result.backend_stats
    assert stats, "backend_stats should be non-empty when the backend is instrumented"
    for key in ("cache_hits", "cache_misses", "forward_passes", "hit_rate"):
        assert key in stats


def test_second_probe_hits_cache_for_duplicate_prompts() -> None:
    """dk2 runs the same 2 prompts × 2 views as dk1 → 4 hits."""
    backend = _programmable_backend()
    result = run_suite(_spec_with_repeated_prompts(), backend)
    stats = result.backend_stats
    # dk1 misses all 4 (2 prompts × 2 views); dk2 hits all 4.
    assert stats["cache_hits"] >= 4, f"expected ≥ 4 hits, got {stats}"
    assert stats["forward_passes"] == stats["cache_misses"]


def test_trace_writer_produces_jsonl(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    backend = _programmable_backend()
    run_suite(_spec_with_repeated_prompts(), backend, trace_path=trace_path)

    lines = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert lines
    # Probe labels flow from the runner into the trace events.
    probe_names = {line["probe"] for line in lines}
    assert "dk1" in probe_names
    assert "dk2" in probe_names
    # Some hits, some misses — the whole point.
    assert any(line["hit"] for line in lines)
    assert any(not line["hit"] for line in lines)


def test_ci_95_survives_runner_roundtrip() -> None:
    """F01 regression — ``_with_duration`` must forward every field.

    The bug: ``_with_duration`` rebuilt the dataclass by hand and
    silently dropped ``ci_95`` on its way out of the runner. Every
    bootstrap CI was stripped before reaching the ``SuiteResult``.
    """
    # delta_kl only emits a bootstrap CI when it has ≥ 4 samples to
    # resample — tailor a fixture that clears that floor.
    base_dist = TokenDist(
        token_ids=np.array([1, 2, 3], dtype=np.int64),
        logprobs=np.log(np.array([0.7, 0.2, 0.1], dtype=np.float32)),
        vocab_size=100,
    )
    ft_dists = {
        "p1": TokenDist(
            token_ids=np.array([1, 2, 3], dtype=np.int64),
            logprobs=np.log(np.array([0.2, 0.4, 0.4], dtype=np.float32)),
            vocab_size=100,
        ),
        "p2": TokenDist(
            token_ids=np.array([1, 2, 3], dtype=np.int64),
            logprobs=np.log(np.array([0.3, 0.3, 0.4], dtype=np.float32)),
            vocab_size=100,
        ),
        "p3": TokenDist(
            token_ids=np.array([1, 2, 3], dtype=np.int64),
            logprobs=np.log(np.array([0.1, 0.5, 0.4], dtype=np.float32)),
            vocab_size=100,
        ),
        "p4": TokenDist(
            token_ids=np.array([1, 2, 3], dtype=np.int64),
            logprobs=np.log(np.array([0.25, 0.35, 0.4], dtype=np.float32)),
            vocab_size=100,
        ),
    }
    backend = DummyDifferentialBackend(
        base=DummyResponses(token_dists=dict.fromkeys(ft_dists, base_dist)),
        ft=DummyResponses(token_dists=ft_dists),
    )
    spec = SwaySpec.model_validate(
        {
            "version": 1,
            "models": {
                "base": {"base": "b"},
                "ft": {"base": "b", "adapter": "/tmp/a"},
            },
            "suite": [
                {
                    "name": "dk",
                    "kind": "delta_kl",
                    "prompts": list(ft_dists.keys()),
                    "assert_mean_gte": 0.0,
                }
            ],
        }
    )
    result = run_suite(spec, backend)
    probe = result.probes[0]
    assert probe.kind == "delta_kl"
    assert probe.ci_95 is not None, (
        "delta_kl emits a bootstrap CI at N=4; the runner dropped it. "
        "Check suite/runner.py:_with_duration."
    )
    lo, hi = probe.ci_95
    assert lo <= (probe.raw or 0.0) <= hi


def test_report_footer_includes_cache_hit_rate() -> None:
    """Report surface shows the ``cache: N/M = X%`` line when stats exist."""
    from dlm_sway.suite import report
    from dlm_sway.suite.score import compute as compute_score

    backend = _programmable_backend()
    result = run_suite(_spec_with_repeated_prompts(), backend)
    score = compute_score(result)
    md = report.to_markdown(result, score)
    assert "cache:" in md
    assert "%" in md
