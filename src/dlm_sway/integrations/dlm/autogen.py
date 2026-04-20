"""Auto-generate a ``sway.yaml`` from a ``.dlm`` document.

Populated by P8 (the .dlm bridge). This module is imported lazily by
``dlm-sway autogen`` so its presence doesn't fail the HF-only path. The
real implementation maps :mod:`dlm.doc.sections` to sway's
:class:`~dlm_sway.core.sections.Section` and emits a spec with every
shipped primitive wired up.
"""

from __future__ import annotations

from pathlib import Path

from dlm_sway.core.errors import SwayError


def write_sway_yaml(dlm_path: Path, out: Path) -> None:
    """Write a generated sway.yaml to ``out`` based on the .dlm at ``dlm_path``.

    Not yet implemented — the .dlm bridge lands in a later milestone.
    """
    del dlm_path, out
    raise SwayError(
        "dlm-sway autogen is not yet implemented — the .dlm bridge is "
        "scheduled for the next milestone. Track progress at "
        "https://github.com/tenseleyFlow/DocumentLanguageModel"
    )
