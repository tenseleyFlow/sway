"""Auto-generate a ``sway.yaml`` from a ``.dlm`` document.

Walks the parsed sections and emits one entry per primitive sway ships:
the full 11-primitive battery wired up against the document's own
content. The result is a YAML artifact the user commits alongside their
``.dlm`` and diffs in PRs.

The generated spec includes a ``dlm_source`` field that the suite loader
uses to pick up :class:`~dlm_sway.core.sections.Section` data at run
time — probes that need sections (B1, B3, C3) then work against the
typed structure instead of re-parsing text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dlm_sway.core.errors import SwayError
from dlm_sway.core.sections import Section
from dlm_sway.integrations.dlm.resolver import DlmHandle, resolve_dlm


def write_sway_yaml(dlm_path: Path, out: Path) -> None:
    """Resolve the .dlm, build a spec dict, write it as YAML to ``out``."""
    handle = resolve_dlm(dlm_path)
    if handle.adapter_path is None:
        raise SwayError(
            f"{dlm_path}: no trained adapter found at ~/.dlm/store/{handle.dlm_id}/adapter; "
            "train the document with `dlm train` before generating a sway suite."
        )
    spec = build_spec_dict(handle, dlm_source=str(dlm_path.resolve()))
    out.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")


def build_spec_dict(handle: DlmHandle, *, dlm_source: str | None = None) -> dict[str, Any]:
    """Build a sway.yaml-shaped dict from a :class:`DlmHandle`."""
    base_spec = {"kind": "hf", "base": handle.base_model}
    ft_spec = {
        "kind": "hf",
        "base": handle.base_model,
        "adapter": str(handle.adapter_path) if handle.adapter_path else None,
    }
    spec: dict[str, Any] = {
        "version": 1,
        "models": {"base": base_spec, "ft": ft_spec},
        "defaults": {"seed": 0, "differential": True},
        "suite": _build_suite(handle.sections),
    }
    if dlm_source is not None:
        spec["dlm_source"] = dlm_source
    return spec


def _build_suite(sections: tuple[Section, ...]) -> list[dict[str, Any]]:
    """Assemble the full probe battery for the given sections.

    The ordering matters: ``null_adapter`` first so every downstream
    probe's z-score threshold has stats to consult.
    """
    instruction_probes: list[tuple[str, str]] = [
        (p.prompt, p.gold) for s in sections if s.kind == "instruction" for p in s.probes
    ]
    prose_prompts: list[str] = []
    for s in sections:
        if s.kind == "prose" and s.content.strip():
            # Use the section's leading sentence as a natural completion prompt.
            first_sentence = s.content.split(".")[0].strip()
            if first_sentence:
                prose_prompts.append(first_sentence + ".")

    kl_prompts = [q for q, _ in instruction_probes][:16] or prose_prompts[:16]
    style_prompts = prose_prompts[:8] or [q for q, _ in instruction_probes][:8]

    suite: list[dict[str, Any]] = []

    # Baseline calibration — always first.
    suite.append({"name": "null_baseline", "kind": "null_adapter", "runs": 3})

    # Adherence.
    if kl_prompts:
        suite.append(
            {
                "name": "delta_kl_doc",
                "kind": "delta_kl",
                "prompts": kl_prompts,
                "assert_mean_gte": 0.02,
            }
        )
    if instruction_probes:
        suite.append(
            {
                "name": "revert_check",
                "kind": "adapter_revert",
                "cases": [
                    {"prompt": q, "gold": a, "paraphrases": _auto_paraphrases(q)}
                    for q, a in instruction_probes[:8]
                ],
                "assert_revert_rate_lt": 0.3,
            }
        )
    if kl_prompts:
        suite.append(
            {
                "name": "prompt_collapse",
                "kind": "prompt_collapse",
                "prompts": kl_prompts[:4],
                "context_lengths": [0, 256, 512, 1024],
                "assert_half_life_tokens": 300,
            }
        )

    # Attribution.
    if len(sections) >= 2:
        suite.append(
            {
                "name": "section_attribution",
                "kind": "section_internalization",
                "per_section_threshold": 0.05,
            }
        )
    if instruction_probes:
        suite.append(
            {
                "name": "paraphrase_invariance",
                "kind": "paraphrase_invariance",
                "cases": [
                    {"prompt": q, "gold": a, "paraphrases": _auto_paraphrases(q)}
                    for q, a in instruction_probes[:6]
                ],
            }
        )
    has_preferences = any(s.kind == "preference" and s.preferences for s in sections)
    if has_preferences:
        suite.append(
            {
                "name": "preference_flip",
                "kind": "preference_flip",
                "assert_flip_rate_gte": 0.7,
            }
        )

    # Calibration.
    if style_prompts:
        suite.append(
            {
                "name": "style_shift",
                "kind": "style_fingerprint",
                "prompts": style_prompts,
            }
        )
    suite.append({"name": "general_knowledge", "kind": "calibration_drift"})
    if any(s.kind == "prose" for s in sections):
        suite.append(
            {
                "name": "verbatim_leak",
                "kind": "leakage",
                "prefix_chars": 128,
                "continuation_chars": 256,
            }
        )

    # Signature ablation — goes last because it's the most expensive.
    if kl_prompts:
        suite.append(
            {
                "name": "adapter_ablation",
                "kind": "adapter_ablation",
                "prompts": kl_prompts[:6],
                "lambdas": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
            }
        )

    return suite


def _auto_paraphrases(prompt: str) -> list[str]:
    """Small, deterministic paraphrase set used when authors don't supply one.

    Purely heuristic — good enough to detect "did the model memorize the
    exact wording". Real paraphrase generation lives behind the
    ``semsim`` extra.
    """
    variants: list[str] = []
    stripped = prompt.rstrip("?. ")
    variants.append(f"Could you explain: {stripped}?")
    variants.append(f"I'd like to know — {stripped}.")
    variants.append(f"Please describe: {stripped}.")
    return variants[:3]
