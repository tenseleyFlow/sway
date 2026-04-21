"""Packaged public-domain corpora for the ``external_perplexity`` probe (S09).

Each corpus ships as a ``.txt`` file under ``_corpora/`` alongside this
module. The files carry inline ``#``-comment provenance headers that
name the original works and their public-domain basis; the loader
strips those headers at read time so probes see only the raw prose.

Callers use :func:`load_corpus` to get a deterministic list of prose
chunks for a given corpus name. Adding a new corpus is three steps:

1. Drop the ``.txt`` file under ``_corpora/`` with provenance comments.
2. Add the name → filename mapping to :data:`_CORPORA`.
3. Extend the ``CorpusName`` literal in ``external_perplexity.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

_CORPUS_DIR: Final = Path(__file__).parent / "_corpora"

_CORPORA: Final[dict[str, str]] = {
    "public_domain_en": "public_domain_en.txt",
}


def available_corpora() -> tuple[str, ...]:
    """Return the names every installed wheel ships corpora for."""
    return tuple(sorted(_CORPORA))


def load_corpus(name: str) -> str:
    """Read the corpus as one string, with ``#``-comment lines stripped.

    Raises :class:`KeyError` when ``name`` isn't registered. The raw
    file stays UTF-8 on disk; the in-memory string is also UTF-8. No
    tokenization happens here — the probe is responsible for chunking
    the returned text.
    """
    if name not in _CORPORA:
        raise KeyError(
            f"unknown external-perplexity corpus {name!r}; "
            f"available: {sorted(_CORPORA)!r}"
        )
    path = _CORPUS_DIR / _CORPORA[name]
    raw = path.read_text(encoding="utf-8")
    # Strip provenance comments and blank lines. What remains is
    # a flat prose block — the external-perplexity probe chunks it
    # into fixed-width windows at measurement time.
    body_lines = [
        line for line in raw.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    return "\n\n".join(body_lines)


def chunk_corpus(text: str, *, chunk_chars: int, max_chunks: int) -> list[str]:
    """Split a corpus string into up to ``max_chunks`` chunks of
    ``chunk_chars`` characters each.

    Chunks are sliced from the start of ``text`` at fixed character
    offsets — deterministic across runs, trivially re-playable. Any
    final chunk shorter than 64 characters is dropped (a partial tail
    contributes noise to rolling-logprob aggregation without adding
    signal).
    """
    if chunk_chars <= 0:
        raise ValueError(f"chunk_chars must be positive; got {chunk_chars}")
    if max_chunks <= 0:
        raise ValueError(f"max_chunks must be positive; got {max_chunks}")
    chunks: list[str] = []
    for start in range(0, len(text), chunk_chars):
        if len(chunks) >= max_chunks:
            break
        piece = text[start : start + chunk_chars]
        if len(piece) >= 64:
            chunks.append(piece)
    return chunks
