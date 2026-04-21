"""Cross-platform determinism golden fixtures (S18).

Contains the minimal spec + platform-pinned ``expected_<platform>.json``
files that ``tests/integration/test_determinism_golden.py`` diffs against.

Regeneration: set ``SWAY_UPDATE_GOLDENS=1`` when running the golden
test, or dispatch the ``determinism-golden`` CI workflow with
``regenerate_goldens=true`` and commit the resulting artifact.
"""
