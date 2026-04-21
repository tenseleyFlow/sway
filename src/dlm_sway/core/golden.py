"""Tolerance-aware JSON comparator for cross-platform golden tests (S18).

A plain ``actual == expected`` on the suite's JSON payload fails across
platforms for reasons that aren't real drift — BLAS implementation
differences produce last-ULP noise on ``raw`` values, wall-time and
timestamps are by definition non-deterministic, and ``sway_version``
bumps every release.

This module encodes the comparison rules the golden test actually
wants:

- **Variable fields are masked** before comparison. Timestamps,
  per-probe ``duration_s``, suite ``wall_seconds``, and the running
  ``sway_version`` are all stripped — these are not load-bearing on
  the determinism claim.
- **Numeric drift is bounded by tolerance.** ``logprob_tol`` (default
  1e-6) covers raw metrics; ``score_tol`` (default 1e-4) covers
  composite scores and probe ``score`` fields. Differences beyond
  those tolerances surface as explicit drift reports with path,
  values, and delta.

No torch / HF dependency — the module is usable from the fast lane
for the comparator's own unit tests.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

#: Default field paths stripped before comparison. Every entry is a
#: dotted key path (``probes.duration_s``) or a plain field name
#: (``sway_version``). Path components are matched anywhere in the
#: nested structure — so ``duration_s`` masks both
#: ``probes[i].duration_s`` and any future top-level ``duration_s``.
DEFAULT_VARIABLE_FIELDS: frozenset[str] = frozenset(
    {
        "started_at",
        "finished_at",
        "wall_seconds",
        "duration_s",
        "sway_version",
        # ``backend_stats`` records per-run wall times + cache counters
        # that vary with load and cold/warm cache — not part of the
        # determinism contract.
        "backend_stats",
        # ``adapter_id`` and ``base_model_id`` are absolute-path
        # identifiers the spec loader resolves against cwd. Different
        # cwds on different platforms (``/Users/.../`` on darwin vs
        # ``/home/runner/...`` on ubuntu) surface as drift without
        # any real numeric change. The numeric fields (``raw``,
        # ``score``, etc.) are what the determinism contract covers.
        "adapter_id",
        "base_model_id",
    }
)

#: Score-level floats: composite scores + per-probe scores. Applies
#: the looser ``score_tol`` since these are derived metrics.
_SCORE_FIELD_NAMES: frozenset[str] = frozenset({"score", "overall"})


@dataclass(frozen=True, slots=True)
class Diff:
    """One tolerance-exceeding drift between actual and expected."""

    path: str
    actual: Any
    expected: Any
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason} (actual={self.actual!r}, expected={self.expected!r})"


def mask_variable_fields(payload: Any, *, fields: frozenset[str] = DEFAULT_VARIABLE_FIELDS) -> Any:
    """Return a deep copy of ``payload`` with variable fields removed.

    Walks nested dicts + lists; drops any dict key whose name is in
    ``fields``. Lists preserve order; scalars pass through unchanged.
    """
    if isinstance(payload, dict):
        return {
            k: mask_variable_fields(v, fields=fields) for k, v in payload.items() if k not in fields
        }
    if isinstance(payload, list):
        return [mask_variable_fields(item, fields=fields) for item in payload]
    return copy.copy(payload)


def compare_goldens(
    actual: Any,
    expected: Any,
    *,
    logprob_tol: float = 1e-4,
    score_tol: float = 1e-4,
) -> list[Diff]:
    """Compare two masked JSON payloads; return tolerance-exceeding diffs.

    Empty list = payloads match within tolerance. Non-empty list = one
    or more fields drifted beyond tolerance.

    Walks the two structures in parallel. Floats are compared against
    the appropriate tolerance (``score_tol`` for score-like fields,
    ``logprob_tol`` elsewhere). Missing keys, length mismatches, or
    type changes surface as structural diffs regardless of tolerance.

    **Tolerance rationale.** S18's first CI observation showed
    intra-platform BLAS drift in the 1e-5–1e-6 band on ubuntu-latest
    runners (heterogeneous Intel/AMD hardware + variable OpenBLAS
    builds), which put 1e-6 below the natural noise floor. A real
    algorithm change — e.g. flipping ``top_k=256`` → 128 in
    :mod:`delta_kl` — shifts probe raws by 1e-2 to 1e-1, three orders
    of magnitude above the current ``1e-4`` tolerance. Tuning room to
    revisit once we have more CI history; see the sprint's risks
    section for the "too tight vs too loose" tradeoff.
    """
    diffs: list[Diff] = []
    _walk(actual, expected, path="$", diffs=diffs, logprob_tol=logprob_tol, score_tol=score_tol)
    return diffs


def _walk(
    actual: Any,
    expected: Any,
    *,
    path: str,
    diffs: list[Diff],
    logprob_tol: float,
    score_tol: float,
) -> None:
    # Type mismatch — catches e.g. scalar-to-dict transitions between
    # schema versions. Treat int/float as comparable — an expected
    # int 0 vs actual 0.0 is not drift.
    if type(actual) is not type(expected) and not (
        isinstance(actual, int | float) and isinstance(expected, int | float)
    ):
        diffs.append(Diff(path=path, actual=actual, expected=expected, reason="type mismatch"))
        return

    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        extra_actual = set(actual) - set(expected)
        missing_actual = set(expected) - set(actual)
        for key in sorted(extra_actual):
            diffs.append(
                Diff(
                    path=f"{path}.{key}",
                    actual=actual[key],
                    expected=None,
                    reason="unexpected key in actual",
                )
            )
        for key in sorted(missing_actual):
            diffs.append(
                Diff(
                    path=f"{path}.{key}",
                    actual=None,
                    expected=expected[key],
                    reason="missing key in actual",
                )
            )
        for key in sorted(set(actual) & set(expected)):
            _walk(
                actual[key],
                expected[key],
                path=f"{path}.{key}",
                diffs=diffs,
                logprob_tol=logprob_tol,
                score_tol=score_tol,
            )
        return

    if isinstance(expected, list):
        assert isinstance(actual, list)
        if len(actual) != len(expected):
            diffs.append(
                Diff(
                    path=path,
                    actual=f"len={len(actual)}",
                    expected=f"len={len(expected)}",
                    reason="list length mismatch",
                )
            )
            return
        for i, (a_item, e_item) in enumerate(zip(actual, expected, strict=True)):
            _walk(
                a_item,
                e_item,
                path=f"{path}[{i}]",
                diffs=diffs,
                logprob_tol=logprob_tol,
                score_tol=score_tol,
            )
        return

    if isinstance(expected, int | float):
        assert isinstance(actual, int | float)
        # Use the looser score_tol for fields whose last path segment
        # is a known score name. Everything else (raw, z_score,
        # base_value, ft_value, logprob entries, component values)
        # gets logprob_tol.
        tol = score_tol if _is_score_path(path) else logprob_tol
        if not _within_tol(float(actual), float(expected), tol):
            diff = float(actual) - float(expected)
            diffs.append(
                Diff(
                    path=path,
                    actual=actual,
                    expected=expected,
                    reason=f"|Δ|={abs(diff):.3e} > tol={tol:.3e}",
                )
            )
        return

    # Strings, None, bool — exact equality required.
    if actual != expected:
        diffs.append(Diff(path=path, actual=actual, expected=expected, reason="value mismatch"))


def _within_tol(a: float, b: float, tol: float) -> bool:
    """Absolute-tolerance float equality, treating NaN/inf carefully.

    Both non-finite on the same side (both +inf, both -inf, both NaN)
    counts as equal. Otherwise the diff must be within ``tol``.
    """
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    if math.isinf(a) or math.isinf(b):
        return a == b  # same-signed infinities compare equal
    return abs(a - b) <= tol


def _is_score_path(path: str) -> bool:
    """Last path segment matches a score-like field name."""
    # Strip trailing ``[idx]`` if any — scores live as scalars inside
    # their parent dict, never indexed.
    key = path.rsplit(".", 1)[-1]
    return key in _SCORE_FIELD_NAMES


__all__ = ["DEFAULT_VARIABLE_FIELDS", "Diff", "compare_goldens", "mask_variable_fields"]
