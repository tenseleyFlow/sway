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

    return DlmHandle(
        dlm_id=fm.dlm_id,
        base_model=fm.base_model,
        adapter_path=adapter_path,
        sections=sections,
        doc_text=doc_text,
    )


def _resolve_adapter_path(dlm_id: str) -> Path | None:
    """Locate the current adapter directory for ``dlm_id``.

    Uses dlm's ``StorePath`` helper if available, else falls back to
    the canonical ``~/.dlm/store/<dlm_id>/adapter/current.txt`` pointer.
    Returns ``None`` if no adapter has been trained yet.
    """
    try:
        from dlm.store.paths import StorePath

        _store_path_cls: object | None = StorePath
    except ImportError:
        _store_path_cls = None

    if _store_path_cls is not None:
        try:
            store = _store_path_cls.for_dlm(dlm_id)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 — unknown dlm exception shapes
            return None
        try:
            resolved = store.resolve_current_adapter()
        except (AttributeError, FileNotFoundError):
            resolved = None
        if resolved is not None and resolved.exists():
            return Path(resolved)

    # Manual fallback in case the dlm API evolves.
    import os

    home = Path(os.environ.get("DLM_HOME", "~/.dlm")).expanduser()
    current_file = home / "store" / dlm_id / "adapter" / "current.txt"
    if current_file.exists():
        pointer = current_file.read_text(encoding="utf-8").strip()
        candidate = (current_file.parent / pointer).resolve()
        if candidate.exists():
            return candidate
    return None


def _translate_section(dlm_section: object) -> Section:
    """Adapt a ``dlm.doc.sections.Section`` to sway's section type.

    The shape dlm uses has been stable through the v0.x series but we
    treat field access defensively so a minor dlm refactor can't silently
    misread section content.
    """
    kind_raw = getattr(dlm_section, "kind", None)
    # dlm uses the attribute name "kind" on its Section dataclass.
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
        probes = tuple(_extract_instruction_probes(dlm_section))
    elif kind == "preference":
        preferences = tuple(_extract_preference_triples(dlm_section))

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


def _extract_instruction_probes(dlm_section: object) -> list[SectionProbe]:
    """Pull (Q, A) pairs out of a dlm INSTRUCTION section.

    dlm's Section carries its parsed Q/A as ``probes`` or ``qa`` depending
    on version. We read the first non-empty one and build
    :class:`SectionProbe` records defensively.
    """
    raw_probes = getattr(dlm_section, "probes", None) or getattr(dlm_section, "qa", None)
    if not raw_probes:
        return []
    out: list[SectionProbe] = []
    for rp in raw_probes:
        q = str(getattr(rp, "prompt", getattr(rp, "question", "")))
        a = str(getattr(rp, "gold", getattr(rp, "answer", "")))
        if q and a:
            out.append(SectionProbe(prompt=q, gold=a))
    return out


def _extract_preference_triples(dlm_section: object) -> list[SectionPreference]:
    """Pull (prompt, chosen, rejected) triples out of a dlm PREFERENCE section."""
    raw = getattr(dlm_section, "preferences", None) or getattr(dlm_section, "triples", None)
    if not raw:
        return []
    out: list[SectionPreference] = []
    for r in raw:
        p = str(getattr(r, "prompt", ""))
        c = str(getattr(r, "chosen", ""))
        rej = str(getattr(r, "rejected", ""))
        if p and c and rej:
            out.append(SectionPreference(prompt=p, chosen=c, rejected=rej))
    return out


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
