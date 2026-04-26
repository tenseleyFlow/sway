"""S26 X3 prove-the-value: sway run on an unpacked swaypack matches
the pre-pack verdict bit-for-bit (modulo timestamp fields).

This is the integration test the sprint DoD requires: pack a real
spec → unpack into a tmp dir → run the unpacked spec → compare
verdict + per-probe scores against the original run. If the round
trip is lossy the audit's "share an adapter audit with a coworker"
flow doesn't actually reproduce.

Marked ``slow + online`` because it goes through the full runner +
spec-loader machinery on a non-trivial fixture.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.slow, pytest.mark.online]


def _write_two_prompt_spec(spec_path: Path) -> None:
    """A minimal-but-real sway.yaml that produces a deterministic
    verdict against the dummy backend (no model load required)."""
    body = {
        "version": 1,
        "models": {
            "base": {"kind": "dummy", "base": "dummy-base"},
            "ft": {"kind": "dummy", "base": "dummy-base"},
        },
        "defaults": {"seed": 0},
        "suite": [
            {
                "name": "dk",
                "kind": "delta_kl",
                "prompts": ["alpha", "beta"],
                "assert_mean_gte": 0.001,  # Easy bar for the dummy ft view.
            }
        ],
    }
    spec_path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")


def _run_spec_via_dummy(spec_path: Path) -> tuple[str, list[tuple[str, str, float | None]]]:
    """Load + run ``spec_path`` against the dummy differential backend.

    Returns ``(suite_verdict, [(probe_name, verdict, score)])`` for
    the round-trip comparison. Bypasses the CLI to avoid subprocess
    flakiness; the runner is what S26 needs to round-trip cleanly.
    """
    from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
    from dlm_sway.suite.loader import load_spec
    from dlm_sway.suite.runner import run as run_suite
    from dlm_sway.suite.score import compute as compute_score

    spec = load_spec(spec_path)
    backend = DummyDifferentialBackend(base=DummyResponses(), ft=DummyResponses())
    # Dummy backend has no .close(); _close_if_possible-style guard.
    try:
        result = run_suite(spec, backend, spec_path=str(spec_path))
        score = compute_score(result)
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()

    probe_summary = [(p.name, str(p.verdict), p.score) for p in result.probes]
    return str(score.band), probe_summary


def test_pack_run_round_trip_matches(tmp_path: Path) -> None:
    """Pack → unpack → run gives the same per-probe verdict + score
    as the pre-pack run, with the SWAY_NULL_CACHE_DIR env honored."""
    from dlm_sway.cli._pack import pack_spec
    from dlm_sway.cli._unpack import unpack_swaypack

    # 1) Build a spec + run it as the "source of truth."
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    src_spec = src_dir / "sway.yaml"
    _write_two_prompt_spec(src_spec)
    pre_band, pre_probes = _run_spec_via_dummy(src_spec)

    # 2) Pack it. include_null_cache=False because the dummy backend
    #    doesn't write null-stats to disk anyway, and the test runs
    #    against an isolated tmp_path so we shouldn't pollute the
    #    user's home cache either way.
    pack_path = tmp_path / "test.swaypack.tar.gz"
    pack_report = pack_spec(src_spec, out_path=pack_path, include_null_cache=False)
    assert pack_report.size_bytes > 0

    # 3) Unpack into a fresh tmp dir.
    unpack_dst = tmp_path / "destination"
    unpack_report = unpack_swaypack(pack_path, target_dir=unpack_dst)

    # 4) Run the unpacked spec, with SWAY_NULL_CACHE_DIR pointing at
    #    the unpacked dir if the pack carried one (it didn't here,
    #    but exercise the env-var honoring path).
    prev_env = os.environ.get("SWAY_NULL_CACHE_DIR")
    if unpack_report.null_stats_dir is not None:
        os.environ["SWAY_NULL_CACHE_DIR"] = str(unpack_report.null_stats_dir)
    try:
        post_band, post_probes = _run_spec_via_dummy(unpack_report.spec_path)
    finally:
        if prev_env is None:
            os.environ.pop("SWAY_NULL_CACHE_DIR", None)
        else:
            os.environ["SWAY_NULL_CACHE_DIR"] = prev_env

    # 5) Verdict + per-probe scores must round-trip exactly.
    assert pre_band == post_band, f"band drifted: {pre_band!r} → {post_band!r}"
    assert pre_probes == post_probes, (
        f"per-probe results drifted:\n  pre: {pre_probes}\n  post: {post_probes}"
    )


def test_unpack_with_null_cache_sets_env_pointer(tmp_path: Path) -> None:
    """If the pack DID carry a null-stats cache, the unpack report's
    null_stats_dir is non-None and points at a real directory the
    caller can hand to ``SWAY_NULL_CACHE_DIR``."""
    from dlm_sway.cli._pack import pack_spec
    from dlm_sway.cli._unpack import unpack_swaypack

    # Build a fake null-stats cache + redirect the pack reader at it.
    fake_cache = tmp_path / "xdg-cache" / "dlm-sway" / "null-stats"
    fake_cache.mkdir(parents=True)
    (fake_cache / "abc123.json").write_text(
        json.dumps({"mean": 1.0, "std": 0.1, "runs": 3}),
        encoding="utf-8",
    )
    prev_xdg = os.environ.get("XDG_CACHE_HOME")
    os.environ["XDG_CACHE_HOME"] = str(tmp_path / "xdg-cache")
    try:
        spec_path = tmp_path / "sway.yaml"
        _write_two_prompt_spec(spec_path)
        out = tmp_path / "with-cache.swaypack.tar.gz"
        report = pack_spec(spec_path, out_path=out, include_null_cache=True)
        assert report.null_stats_count == 1
    finally:
        if prev_xdg is None:
            os.environ.pop("XDG_CACHE_HOME", None)
        else:
            os.environ["XDG_CACHE_HOME"] = prev_xdg

    target = tmp_path / "u"
    unpack_report = unpack_swaypack(out, target_dir=target)
    assert unpack_report.null_stats_dir is not None
    assert (unpack_report.null_stats_dir / "abc123.json").exists()
