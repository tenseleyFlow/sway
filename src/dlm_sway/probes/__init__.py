"""Probe primitives. Each module in this package implements one primitive.

Importing this package eagerly imports every probe module so their
``__init_subclass__`` hooks populate the registry. If you're hitting
"unknown probe kind" from :func:`dlm_sway.probes.base.build_probe`, the
fix is to ``import dlm_sway.probes`` before building the probe — which
this ``__init__`` does for you.
"""

from __future__ import annotations

# Register every shipped probe with the central registry by importing
# its module. Order is not load-bearing for registration but matches the
# categorical grouping in :mod:`dlm_sway.core.result`.
from dlm_sway.probes import (  # noqa: F401 — imports register the probes
    adapter_revert,
    calibration_drift,
    delta_kl,
    leakage,
    null_adapter,
    paraphrase_invariance,
    preference_flip,
    prompt_collapse,
    section_internalization,
    style_fingerprint,
)
