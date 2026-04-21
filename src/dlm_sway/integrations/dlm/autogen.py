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

#: Stylistic-elicitation prompts used by the generated ``style_fingerprint``
#: probe (B8). Picked to be open-ended and content-neutral — the model's
#: voice under the adapter is the signal we want, *not* its ability to
#: continue a sentence the doc already wrote. Each prompt deliberately
#: invites prose-shaped output of moderate length (one paragraph).
_STYLE_ELICITATION_PROMPTS: tuple[str, ...] = (
    "Write a short paragraph explaining your approach to a difficult problem.",
    "Describe what you find most interesting about a topic you know well.",
    "Summarize an important idea for a curious novice.",
    "Reflect on a small lesson you learned recently.",
    "Explain a concept using a concrete example.",
    "Tell a brief story that illustrates a single point.",
)


#: Per-probe intent one-liners (D5). Keyed by probe ``kind``. Used to
#: prepend a ``#``-comment above each suite entry in the generated
#: YAML so a first-time reader understands what each probe is for
#: without cross-referencing the docs.
_PROBE_INTENT: dict[str, str] = {
    "null_adapter": (
        "Calibration baseline — runs first so downstream probes have "
        "per-kind null stats for z-scores."
    ),
    "delta_kl": (
        "A1: mean JS divergence of next-token distributions between "
        "base and ft. Did the adapter move the model on doc content?"
    ),
    "adapter_revert": (
        "A2: does the ft model drift back to base under adversarial "
        "paraphrase? Needs the [semsim] extra."
    ),
    "prompt_collapse": (
        "A3: fit exponential decay of divergence over context length. "
        "Catches adapters whose influence evaporates with context."
    ),
    "section_internalization": (
        "B1 (flagship): per-section attribution with leak-check. "
        "Which parts of the doc actually moved the model?"
    ),
    "paraphrase_invariance": (
        "B2: memorization vs generalization — does the adapter lift "
        "the verbatim prompt more than paraphrased variants?"
    ),
    "preference_flip": (
        "B3: on DPO/ORPO triples, did ft flip the chosen/rejected ranking relative to base?"
    ),
    "style_fingerprint": (
        "C1: stylistic shift toward the doc's fingerprint. Uses 9-dim "
        "extended vector when [style] installed; 6-dim otherwise."
    ),
    "calibration_drift": (
        "C2: general-knowledge regression check. Did the fine-tune "
        "forget the world while learning the doc?"
    ),
    "external_perplexity": (
        "F3: diffuse-forgetting check — rolling-logprob delta on "
        "held-out public-domain English. Complements calibration_drift "
        "(the point-factual counterpart)."
    ),
    "leakage": (
        "C3: verbatim-recital + perturbation-fragility check. High "
        "recall + low fragility → memorization, not generalization."
    ),
    "adapter_ablation": (
        "N2 (signature): λ-scaled divergence curve. A healthy adapter "
        "shows a smooth, non-saturated response; a degenerate one is "
        "a step function."
    ),
}


def write_sway_yaml(dlm_path: Path, out: Path) -> None:
    """Resolve the .dlm, build a spec dict, write it as YAML to ``out``."""
    handle = resolve_dlm(dlm_path)
    if handle.adapter_path is None:
        raise SwayError(
            f"{dlm_path}: no trained adapter found at ~/.dlm/store/{handle.dlm_id}/adapter; "
            "train the document with `dlm train` before generating a sway suite."
        )
    spec = build_spec_dict(handle, dlm_source=_portable_dlm_source(dlm_path))
    skipped = collect_skipped_probe_reasons(handle)
    out.write_text(
        _render_annotated_yaml(spec, handle, dlm_path, skipped=skipped),
        encoding="utf-8",
    )


def collect_skipped_probe_reasons(handle: DlmHandle) -> list[tuple[str, str]]:
    """Return ``(probe_kind, reason)`` tuples for every probe
    ``_build_suite`` intentionally omitted for this ``.dlm``.

    F07 (Audit 03) — the emitted YAML previously had no record of
    which probes were skipped and why. Users had to diff the autogen
    output against the intent docstring to know. This surface is the
    input to the YAML-comment block the renderer prepends.

    Mirrors the conditional logic inside :func:`_build_suite` — any
    change to that function's gating must update this function too.
    """
    sections = handle.sections
    instruction_probes = [
        (p.prompt, p.gold) for s in sections if s.kind == "instruction" for p in s.probes
    ]
    prose_prompts = [
        s.content.split(".")[0].strip()
        for s in sections
        if s.kind == "prose" and s.content.strip() and s.content.split(".")[0].strip()
    ]
    has_instruction_probes = bool(instruction_probes)
    has_prose = any(s.kind == "prose" for s in sections)
    has_preferences = any(s.kind == "preference" and s.preferences for s in sections)

    kl_prompts = [q for q, _ in instruction_probes][:16] or prose_prompts[:16]
    all_instruction_prompts = [q for q, _ in instruction_probes]
    cluster_pool_size = len({*all_instruction_prompts, *prose_prompts})

    skipped: list[tuple[str, str]] = []
    if not kl_prompts:
        skipped.append(("delta_kl", "no instruction probes or prose sections"))
    if not has_instruction_probes:
        skipped.append(("adapter_revert", "no !probe markers in INSTRUCTION sections"))
        skipped.append(("paraphrase_invariance", "no !probe markers in INSTRUCTION sections"))
    if not kl_prompts:
        skipped.append(("prompt_collapse", "no prompts available to score"))
    if len(sections) < 2:
        skipped.append(("section_internalization", "document has fewer than 2 sections"))
    if not has_preferences:
        skipped.append(("preference_flip", "no PREFERENCE sections with populated triples"))
    if not has_prose:
        skipped.append(
            ("external_perplexity", "no PROSE sections to measure external-corpus drift against")
        )
        skipped.append(("leakage", "no PROSE sections to extract prefix/continuation windows from"))
    if cluster_pool_size < 20:
        skipped.append(
            (
                "cluster_kl",
                f"only {cluster_pool_size} distinct prompts in pool (need ≥ 20 for stable clustering)",
            )
        )
    if not kl_prompts:
        skipped.append(("adapter_ablation", "no prompts available to score"))
    return skipped


def _portable_dlm_source(dlm_path: Path) -> str:
    """Return a ``dlm_source`` string that survives cross-machine checkout.

    F09 (Audit 03) — the pre-fix code unconditionally wrote an
    absolute path (``/Users/mfwolffe/.../fortran.dlm``) which breaks
    when the autogen'd ``sway.yaml`` is committed to a repo and
    re-run from a different working tree (CI agents, another dev's
    checkout). The cwd-relative form is round-trippable across
    machines; only fall back to absolute when the ``.dlm`` lives
    outside the cwd (e.g. a global user dir) where relativization
    doesn't resolve on a fresh checkout.
    """
    abs_path = dlm_path.resolve()
    cwd = Path.cwd().resolve()
    try:
        # ``is_relative_to`` lands in 3.9+; this path is guaranteed
        # to exist because sway requires ``>=3.11``.
        if abs_path.is_relative_to(cwd):
            return str(abs_path.relative_to(cwd))
    except ValueError:
        pass
    return str(abs_path)


def _render_annotated_yaml(
    spec: dict[str, Any],
    handle: DlmHandle,
    dlm_path: Path,
    *,
    skipped: list[tuple[str, str]] | None = None,
) -> str:
    """Render the spec as YAML with a provenance header + per-probe intent lines (D5).

    Uses pyyaml (already a hard dep) and post-processes the output to
    insert ``#``-comments above each suite entry. Avoids the
    ``ruamel.yaml`` dep the sprint contemplated — the annotation here
    is structural (position-based), not round-trippable, so the lighter
    approach is sufficient.

    F07 (Audit 03) — when ``skipped`` is non-empty, the header gains a
    ``# skipped: <probe> (<reason>)`` block so users see which probes
    the autogen intentionally omitted, without diffing the autogen
    module's docstring.
    """
    import datetime as _dt

    from dlm_sway import __version__

    body = yaml.safe_dump(spec, sort_keys=False)
    annotated = _inject_probe_intent_comments(body)

    header_lines = [
        "# sway.yaml — auto-generated by `sway autogen`",
        f"# source:   {dlm_path.resolve()}",
        f"# dlm_id:   {handle.dlm_id}",
        f"# base:     {handle.base_model}",
        f"# adapter:  {handle.adapter_path}",
        f"# generated: {_dt.datetime.now(_dt.UTC).isoformat(timespec='seconds')}",
        f"# sway:     {__version__}",
        "#",
        "# Edit freely — this file is your checked-in contract. Re-running",
        "# `sway autogen` overwrites it; commit the generated file so your",
        "# test suite is diffable in PRs.",
    ]
    if skipped:
        header_lines.extend(
            [
                "#",
                f"# {len(skipped)} probe(s) intentionally omitted for this document:",
                *[f"# skipped: {kind} ({reason})" for kind, reason in skipped],
                "# (sway gate will still pass — missing probes don't fail the gate.)",
            ]
        )
    header_lines.append("")
    return "\n".join(header_lines) + annotated


def _inject_probe_intent_comments(yaml_body: str) -> str:
    """Walk the rendered YAML; prepend a ``#`` intent line above each suite entry."""
    import re as _re

    # Each suite entry begins with ``- name: <value>`` at the same
    # indent. We scan the lines, track the indent of the first list
    # item we see under ``suite:``, and insert intent comments there.

    lines = yaml_body.splitlines()
    out: list[str] = []
    in_suite = False
    # Each ``- name:`` marks the start of a suite entry. We buffer the
    # lines of that entry and peek at the ``kind:`` value to pick the
    # right intent comment to insert before the ``- name:`` line. A
    # one-line "index where the intent goes" pointer is simpler than
    # doing a two-pass rewrite.
    entry_start: int | None = None
    entry_indent = 0

    def _flush_entry_header(entry_start_idx: int | None) -> None:
        if entry_start_idx is None:
            return
        entry_lines = out[entry_start_idx:]
        kind: str | None = None
        for elt in entry_lines:
            match = _re.search(r"\bkind:\s*([A-Za-z_][A-Za-z0-9_]*)", elt)
            if match is not None:
                kind = match.group(1)
                break
        if kind is None:
            return
        intent = _PROBE_INTENT.get(kind)
        if intent is None:
            return
        out.insert(entry_start_idx, " " * entry_indent + f"# {intent}")

    for line in lines:
        stripped = line.lstrip()
        # Top-level keys toggle the suite scope.
        if line and not line[0].isspace() and not line.startswith("- "):
            # Close the previous entry (if any) before switching scope.
            _flush_entry_header(entry_start)
            entry_start = None
            in_suite = stripped == "suite:"
            out.append(line)
            continue

        if in_suite and stripped.startswith("- "):
            # New entry — flush any pending comment for the previous.
            _flush_entry_header(entry_start)
            entry_start = len(out)
            entry_indent = len(line) - len(stripped)

        out.append(line)

    # Flush the final entry.
    _flush_entry_header(entry_start)
    return "\n".join(out) + ("\n" if yaml_body.endswith("\n") else "")


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
    # B8: style_fingerprint needs *stylistic elicitation* — open-ended
    # prompts that ask the model to write in its own voice — not the
    # leading sentence of a doc paragraph (which elicits continuation
    # of the doc itself, conflating style with content). The fixed set
    # below is intentionally generic so the model's stylistic shift
    # under the adapter is the only signal in play.
    style_prompts = list(_STYLE_ELICITATION_PROMPTS)

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
    # Emit the external_perplexity probe when the doc has any PROSE
    # content at all — the probe measures *external* prose degradation,
    # so the docs that benefit most are the ones where the adapter was
    # trained on text that might over-fit the base model's English
    # fluency.
    if any(s.kind == "prose" for s in sections):
        suite.append(
            {
                "name": "external_ppl",
                "kind": "external_perplexity",
                "corpus": "public_domain_en",
                "max_chunks": 8,  # half of default for faster autogen'd runs
            }
        )
        suite.append(
            {
                "name": "verbatim_leak",
                "kind": "leakage",
                "prefix_chars": 128,
                "continuation_chars": 256,
            }
        )

    # F07 — ``cluster_kl`` when the prompt pool clears the probe's
    # ``min_prompts`` floor. Pulls from the *full* instruction pool +
    # prose leading sentences (``kl_prompts`` is capped at 16 for
    # delta_kl; we want wider coverage for clustering). S16's scope
    # set a 20-prompt floor; mirror it so emission is stable across
    # documents of varying length.
    all_instruction_prompts = [q for q, _ in instruction_probes]
    cluster_prompts: list[str] = []
    seen: set[str] = set()
    for p in all_instruction_prompts + prose_prompts:
        if p not in seen:
            seen.add(p)
            cluster_prompts.append(p)
    if len(cluster_prompts) >= 20:
        suite.append(
            {
                "name": "cluster_kl_topics",
                "kind": "cluster_kl",
                "prompts": cluster_prompts[:64],
                "num_clusters": 5,
                "min_prompts": 20,
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
