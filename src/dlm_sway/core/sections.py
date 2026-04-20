"""Minimal section contract for attribution probes.

The flagship B1 ``section_internalization`` probe needs *structured*
input — a section has an id, a kind, content text, and possibly some
Q/A pairs or chosen/rejected triples. sway defines this shape here so
the probes stay oblivious to the upstream (``.dlm`` parser, custom
loaders, synthetic test fixtures).

Field names are aligned with :mod:`dlm.doc.sections` but this module
does not import ``dlm`` — the bridge at
:mod:`dlm_sway.integrations.dlm` does the adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SectionKind = Literal["prose", "instruction", "preference"]


@dataclass(frozen=True, slots=True)
class SectionProbe:
    """A ``(prompt, gold)`` pair lifted from an INSTRUCTION section."""

    prompt: str
    gold: str


@dataclass(frozen=True, slots=True)
class SectionPreference:
    """A ``(prompt, chosen, rejected)`` triple from a PREFERENCE section."""

    prompt: str
    chosen: str
    rejected: str


@dataclass(frozen=True, slots=True)
class Section:
    """One typed chunk of a training document.

    Attributes
    ----------
    id:
        Content-addressed identifier. ``.dlm`` uses a 16-hex-char
        sha256 prefix; sway doesn't enforce a format.
    kind:
        Discriminator for which of :attr:`probes` /
        :attr:`preferences` / :attr:`content` is the primary signal.
    content:
        Raw section text. Always populated; used by the rolling-PPL
        path for PROSE sections.
    probes:
        For INSTRUCTION: parsed Q/A pairs. Empty tuple for others.
    preferences:
        For PREFERENCE: parsed chosen/rejected triples. Empty otherwise.
    tag:
        Optional free-form label for the section (e.g., "intro",
        "api-reference"). Surfaces in per-section reports.
    """

    id: str
    kind: SectionKind
    content: str
    probes: tuple[SectionProbe, ...] = field(default_factory=tuple)
    preferences: tuple[SectionPreference, ...] = field(default_factory=tuple)
    tag: str | None = None


def filter_kinds(
    sections: tuple[Section, ...], kinds: tuple[SectionKind, ...]
) -> tuple[Section, ...]:
    """Return only sections whose ``kind`` matches one of ``kinds``."""
    allow = set(kinds)
    return tuple(s for s in sections if s.kind in allow)
