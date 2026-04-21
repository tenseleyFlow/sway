"""Meta-test: every documented spec/CLI option must have a real consumer.

Audit 01 caught two dead options (P14 ``defaults.differential`` and
P15 ``compute(weights=...)``) that were defined, documented, and never
read by any code path. Sprint 03 wired both of them. This test pins
that wiring so a future refactor that drops the consumer surfaces here
rather than in a user's report.

The check is a literal substring grep: each documented option name
must appear in *consumer* code outside the spec / scoring module that
declares it. We walk ``src/dlm_sway`` for the substring and compare to
a known set of declaration sites; anything else counts as a consumer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).parents[2] / "src" / "dlm_sway"


def _grep_files(needle: str) -> list[Path]:
    """Return every .py file under ``SRC_ROOT`` whose text contains ``needle``."""
    return sorted(p for p in SRC_ROOT.rglob("*.py") if needle in p.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("option", "declaration_files"),
    [
        # P14: defaults.differential
        (
            "differential",
            # Declared in the spec; everything else is a consumer.
            {"src/dlm_sway/suite/spec.py"},
        ),
        # P15: compute(weights=...)
        (
            "score_weights",
            # Declared in the spec; everything else is a consumer.
            {"src/dlm_sway/suite/spec.py"},
        ),
    ],
)
def test_documented_options_have_consumers(option: str, declaration_files: set[str]) -> None:
    hits = _grep_files(option)
    assert hits, f"audit-listed option {option!r} not found anywhere — was it deleted?"

    rel_hits = {str(h.relative_to(SRC_ROOT.parents[1])) for h in hits}
    consumers = rel_hits - declaration_files
    assert consumers, (
        f"option {option!r} appears only in declarations {declaration_files}; "
        f"no consumer found. Audit 01 P14/P15 caught this exact dead-option "
        f"pattern; restore a consumer or drop the option."
    )
