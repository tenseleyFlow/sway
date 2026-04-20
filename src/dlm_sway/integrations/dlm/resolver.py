"""Resolve a ``.dlm`` file to the artifacts sway needs.

Imports ``dlm.*`` — requires the ``dlm-sway[dlm]`` extra. Everything
outside this package is oblivious to dlm's internal shape; the bridge
is the only place that knows, e.g., that a dlm section carries a
``kind`` field named ``type`` or that adapters live at
``adapter/versions/vNNNN/``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from dlm_sway.core.errors import SwayError
from dlm_sway.core.sections import (
    Section,
    SectionKind,
    SectionPreference,
    SectionProbe,
)


@dataclass(frozen=True, slots=True)
class DlmHandle:
    """Everything the sway bridge pulls out of a ``.dlm`` file.

    Attributes
    ----------
    dlm_id:
        Stable identifier from the frontmatter.
    base_model:
        Either a HF id (``qwen2.5-1.5b``) or an ``hf:org/name`` escape
        hatch, taken verbatim from the frontmatter.
    adapter_path:
        Directory containing the current trained PEFT adapter (resolved
        via dlm's own ``StorePath.for_dlm``). ``None`` if the document
        hasn't been trained yet.
    sections:
        Typed sections ready for sway's probes.
    doc_text:
        Concatenated raw content of all sections. Used by probes that
        need a whole-document stylistic reference (C1).
    """

    dlm_id: str
    base_model: str
    adapter_path: Path | None
    sections: tuple[Section, ...]
    doc_text: str


def resolve_dlm(dlm_path: Path) -> DlmHandle:
    """Parse ``dlm_path`` and return a :class:`DlmHandle`.

    Raises :class:`~dlm_sway.core.errors.SwayError` with a clear message
    when the file is malformed or when the resolved adapter path doesn't
    exist on disk.
    """
    try:
        from dlm.doc.parser import parse_file as dlm_parse_file
    except ImportError as exc:
        raise SwayError("dlm package not installed — run: pip install 'dlm-sway[dlm]'") from exc

    parsed = dlm_parse_file(dlm_path)
    fm = parsed.frontmatter
    sections = tuple(_translate_section(s) for s in parsed.sections)
    doc_text = "\n\n".join(s.content for s in sections)

    adapter_path = _resolve_adapter_path(fm.dlm_id)
    base_hf_id = _resolve_base_model_to_hf_id(fm.base_model)

    return DlmHandle(
        dlm_id=fm.dlm_id,
        base_model=base_hf_id,
        adapter_path=adapter_path,
        sections=sections,
        doc_text=doc_text,
    )


def _resolve_base_model_to_hf_id(base_model: str) -> str:
    """Translate dlm's base-model *key* to a HuggingFace repo id.

    dlm's frontmatter stores registry keys like ``smollm2-135m`` which
    resolve to ``HuggingFaceTB/SmolLM2-135M-Instruct``. sway's backends
    call ``AutoModelForCausalLM.from_pretrained`` directly and need the
    HF id. The ``hf:org/name`` escape hatch passes through unchanged.
    """
    if base_model.startswith("hf:"):
        return base_model[len("hf:") :]
    try:
        from dlm.base_models import resolve as resolve_base
    except ImportError:
        return base_model
    try:
        spec = resolve_base(base_model)
    except Exception:  # noqa: BLE001 — unknown dlm errors
        return base_model
    hf_id = getattr(spec, "hf_id", None)
    return str(hf_id) if hf_id else base_model


def _resolve_adapter_path(dlm_id: str) -> Path | None:
    """Locate the current adapter directory for ``dlm_id``.

    Uses dlm's module-level ``for_dlm`` helper if available, else falls
    back to the canonical ``~/.dlm/store/<dlm_id>/adapter/current.txt``
    pointer. Returns ``None`` if no adapter has been trained yet.
    """
    # Primary path: use dlm's own store-path helpers.
    try:
        from dlm.store.paths import for_dlm as _for_dlm
    except ImportError:
        _for_dlm = None

    if _for_dlm is not None:
        try:
            store = _for_dlm(dlm_id)
        except Exception:  # noqa: BLE001 — unknown dlm exception shapes
            store = None
        if store is not None:
            try:
                resolved = store.resolve_current_adapter()
            except (AttributeError, FileNotFoundError):
                resolved = None
            if resolved is not None and Path(resolved).exists():
                return Path(resolved)

    # Manual fallback. The ``current.txt`` pointer is relative to the
    # **store root**, not to current.txt's parent dir — so go up one level.
    import os

    home = Path(os.environ.get("DLM_HOME", "~/.dlm")).expanduser()
    store_root = home / "store" / dlm_id
    current_file = store_root / "adapter" / "current.txt"
    if current_file.exists():
        pointer = current_file.read_text(encoding="utf-8").strip()
        candidate = (store_root / pointer).resolve()
        if candidate.exists():
            return candidate
    return None


def _translate_section(dlm_section: object) -> Section:
    """Adapt a ``dlm.doc.sections.Section`` to sway's section type.

    dlm's Section dataclass uses the attribute name ``type`` (not
    ``kind``) and stores instruction/preference content as raw markdown
    — dlm ships dedicated parsers (``parse_instruction_body``,
    ``parse_preference_body``) that we reuse here so any future dlm
    syntax additions land in sway for free.
    """
    # dlm's current attribute is ``type``; older revisions used ``kind``.
    kind_raw = getattr(dlm_section, "type", getattr(dlm_section, "kind", None))
    kind = _normalize_kind(kind_raw)
    content = str(getattr(dlm_section, "content", ""))
    section_id = str(
        getattr(dlm_section, "section_id", None)
        or getattr(dlm_section, "id", None)
        or _content_hash(content)
    )
    tag = getattr(dlm_section, "tag", None)

    probes: tuple[SectionProbe, ...] = ()
    preferences: tuple[SectionPreference, ...] = ()
    if kind == "instruction":
        probes = tuple(_parse_instruction(content, section_id=section_id))
    elif kind == "preference":
        preferences = tuple(_parse_preference(content, section_id=section_id))

    return Section(
        id=section_id,
        kind=kind,
        content=content,
        probes=probes,
        preferences=preferences,
        tag=tag if isinstance(tag, str) else None,
    )


def _normalize_kind(raw: object) -> SectionKind:
    """Map dlm's SectionType/str to sway's lowercase kind."""
    if raw is None:
        return "prose"
    value = str(raw).lower()
    # dlm uses uppercase StrEnum values like "PROSE"; normalize.
    if value.endswith("prose") or "prose" in value:
        return "prose"
    if "instruction" in value:
        return "instruction"
    if "preference" in value:
        return "preference"
    return "prose"


def _parse_instruction(content: str, *, section_id: str) -> list[SectionProbe]:
    """Pull (Q, A) pairs out of a dlm INSTRUCTION section body.

    Delegates to dlm's own ``parse_instruction_body`` so syntax additions
    land in sway without code changes here. Falls back to an empty list
    on parse errors — the probe will fail gracefully.
    """
    try:
        from dlm.data.instruction_parser import parse_instruction_body
    except ImportError:
        return []
    try:
        pairs = parse_instruction_body(content, section_id=section_id)
    except Exception:  # noqa: BLE001 — dlm raises InstructionParseError
        return []
    out: list[SectionProbe] = []
    for p in pairs:
        q = getattr(p, "question", getattr(p, "prompt", ""))
        a = getattr(p, "answer", getattr(p, "gold", ""))
        if q and a:
            out.append(SectionProbe(prompt=str(q), gold=str(a)))
    return out


def _parse_preference(content: str, *, section_id: str) -> list[SectionPreference]:
    """Pull (prompt, chosen, rejected) triples out of a PREFERENCE body."""
    try:
        from dlm.data.preference_parser import parse_preference_body
    except ImportError:
        return []
    try:
        triples = parse_preference_body(content, section_id=section_id)
    except Exception:  # noqa: BLE001 — dlm raises PreferenceParseError
        return []
    out: list[SectionPreference] = []
    for t in triples:
        p = str(getattr(t, "prompt", ""))
        c = str(getattr(t, "chosen", ""))
        rej = str(getattr(t, "rejected", ""))
        if p and c and rej:
            out.append(SectionPreference(prompt=p, chosen=c, rejected=rej))
    return out


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
