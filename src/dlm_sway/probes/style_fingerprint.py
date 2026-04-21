"""C1 StyleFingerprint — does ft prose *read* like the doc?

Generates base and ft completions from a set of stylistic prompts,
extracts a 6-dimensional fingerprint from each, and measures how the ft
fingerprint has shifted **toward** the training document's own
fingerprint vs the base.

We compute the fingerprint with numpy-only features so the probe works
out of the box without spaCy/textstat. The optional ``style`` extra
upgrades the fingerprint with passive-voice rate and POS-entropy in a
later milestone; the numeric contract — a non-negative vector per text
— is stable across that upgrade.

Signal: ``style_shift = cos(ft_fp - base_fp, doc_fp - base_fp)`` in
fingerprint space. Positive values mean ft has moved *toward* the
doc's style; negative values mean it moved *away* (a bad sign);
near-zero means no stylistic shift detectable.
"""

from __future__ import annotations

import re
import statistics
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from dlm_sway.core.result import ProbeResult, Verdict, safe_finalize
from dlm_sway.probes._zscore import (
    no_calibration_note,
    score_from_z,
    verdict_from_z,
    z_score,
)
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext
from dlm_sway.probes.null_adapter import get_null_stats

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z'-]*\b")
_PUNCTS = set(".,:;!?-—()[]\"'/")

#: Evidence schema version — bumped when dimensionality of the
#: fingerprint changes so downstream consumers can branch on it.
FINGERPRINT_SCHEMA_VERSION = 2


def _core_fingerprint(text: str) -> NDArray[np.float64]:
    """The numpy-only 6-dim fingerprint. Always available, no optional
    deps. Shared subroutine for both the base and the extended paths."""
    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    paragraphs = [p for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    words = _WORD_RE.findall(text)
    if not words:
        return np.zeros(6, dtype=np.float64)

    sentence_word_counts = [len(_WORD_RE.findall(s)) for s in sentences]
    sentence_word_counts = [c for c in sentence_word_counts if c > 0]
    if not sentence_word_counts:
        sentence_word_counts = [len(words)]

    mean_sent = statistics.fmean(sentence_word_counts)
    std_sent = statistics.pstdev(sentence_word_counts) if len(sentence_word_counts) > 1 else 0.0
    ttr = len({w.lower() for w in words}) / len(words)
    avg_word_len = statistics.fmean(len(w) for w in words)
    punct_count = sum(ch in _PUNCTS for ch in text)
    punct_density = punct_count / max(len(text), 1)
    avg_paragraph_len = (
        statistics.fmean(len(_WORD_RE.findall(p)) for p in paragraphs) if paragraphs else len(words)
    )
    paragraph_density = 1.0 / max(avg_paragraph_len, 1.0)

    return np.asarray(
        [
            mean_sent / 30.0,
            std_sent / 30.0,
            ttr,
            avg_word_len / 10.0,
            punct_density * 10.0,
            paragraph_density * 30.0,
        ],
        dtype=np.float64,
    )


def _has_style_extra() -> bool:
    """Probe whether the ``style`` optional extra is installed.

    Heuristic import of ``spacy`` and ``textstat``. The ``nlpaug``
    package the extra also pulls is used elsewhere (paraphrase
    augmentation, Sprint 05) — its presence isn't required for the
    extended fingerprint.
    """
    import importlib.util

    return (
        importlib.util.find_spec("spacy") is not None
        and importlib.util.find_spec("textstat") is not None
    )


def _extended_fingerprint(text: str) -> NDArray[np.float64] | None:
    """Return the 9-dim extended fingerprint if spacy+textstat load, else None.

    Three new dims on top of :func:`_core_fingerprint`:
      6. passive-voice rate (POS pattern: NOUN VBN / total sentences) * 10.0
      7. POS 4-gram entropy (shannon, in bits) / 10.0
      8. syllables per word (textstat) / 5.0

    Returns ``None`` on any import or model-load failure so the probe
    can gracefully fall back to the 6-dim core. This path is only hit
    once per ``StyleFingerprintProbe.run()`` call (two texts per call),
    so the spaCy load cost is amortized across prompts.
    """
    try:
        import spacy
        import textstat
    except ImportError:
        return None

    # The small English pipeline. A user who installs the ``style``
    # extra is expected to run ``python -m spacy download en_core_web_sm``
    # afterwards; if they haven't, we fall back to the 6-dim core.
    try:
        nlp = spacy.load("en_core_web_sm", disable=("ner", "lemmatizer"))
    except (OSError, ImportError):
        return None

    doc = nlp(text)
    sentences = list(doc.sents)
    if not sentences:
        return np.zeros(9, dtype=np.float64)

    passive_sentences = 0
    pos_tags: list[str] = []
    for sent in sentences:
        tags = [tok.pos_ for tok in sent if not tok.is_space]
        pos_tags.extend(tags)
        # "NOUN VBN" pattern via POS bigram. VBN = past participle;
        # spaCy's coarse POS uses "VERB" with morph feature. Fall back
        # to the ``Tag`` attribute which exposes Penn-style VBN directly.
        for i in range(len(sent) - 1):
            if sent[i].pos_ in ("NOUN", "PROPN") and sent[i + 1].tag_ == "VBN":
                passive_sentences += 1
                break

    passive_rate = passive_sentences / len(sentences)

    # POS 4-gram Shannon entropy (in bits). Capped at a reasonable
    # maximum so the normalized dim stays ~order-1.
    pos_entropy_bits = 0.0
    if len(pos_tags) >= 4:
        import math
        from collections import Counter

        four_grams = [tuple(pos_tags[i : i + 4]) for i in range(len(pos_tags) - 3)]
        counts = Counter(four_grams)
        total = sum(counts.values())
        pos_entropy_bits = -sum(
            (c / total) * math.log2(c / total) for c in counts.values()
        )

    syllables_per_word = float(textstat.syllable_count(text)) / max(len(pos_tags), 1)

    return np.concatenate(
        [
            _core_fingerprint(text),
            np.asarray(
                [
                    passive_rate * 10.0,
                    pos_entropy_bits / 10.0,
                    syllables_per_word / 5.0,
                ],
                dtype=np.float64,
            ),
        ]
    )


def fingerprint(text: str, *, extended: bool = False) -> NDArray[np.float64]:
    """Return a stylistic fingerprint for ``text``.

    With ``extended=False`` (the default) returns the 6-dim numpy-only
    fingerprint. With ``extended=True`` and the ``style`` extra
    installed, returns a 9-dim vector with passive-voice rate, POS
    4-gram entropy, and syllables/word density appended. Falls back
    to 6-dim when spaCy or textstat aren't importable so probe-level
    callers never need to guard the import themselves.

    Dimensions (all numeric, scaled to order-1):
      0. mean sentence length (words)  / 30.0
      1. std sentence length (words)   / 30.0
      2. type-token ratio              (already in [0,1])
      3. avg word length (chars)       / 10.0
      4. punctuation density per char  * 10.0
      5. paragraph density (1 / avg paragraph length in words) * 30.0
      6. (extended) passive-voice rate per sentence * 10.0
      7. (extended) POS 4-gram entropy (bits) / 10.0
      8. (extended) syllables per word / 5.0
    """
    if not text.strip():
        return np.zeros(9 if extended else 6, dtype=np.float64)
    if extended:
        ext = _extended_fingerprint(text)
        if ext is not None:
            return ext
    return _core_fingerprint(text)


class StyleFingerprintSpec(ProbeSpec):
    kind: Literal["style_fingerprint"] = "style_fingerprint"
    prompts: list[str] = Field(default_factory=list)
    """Prompts used to elicit a stylistic sample from each model."""
    doc_reference: str = ""
    """Concatenated reference text representing the adapter's intended
    style. Typically the document itself; the .dlm bridge supplies this
    from ``ctx.doc_text`` when left empty."""
    max_new_tokens: int = 128
    assert_shift_gte: float = 0.25
    """Minimum cosine shift for PASS. ``0.25`` is a deliberately
    permissive default — stylistic shift is a weaker signal than
    perplexity lift."""
    assert_z_gte: float = 3.0
    """Z-score pass criterion against the null-adapter baseline, when it
    exists. Preferred over the raw threshold."""
    extended: Literal["auto", "on", "off"] = "auto"
    """Controls whether to use the 9-dim extended fingerprint
    (passive-voice rate, POS 4-gram entropy, syllable density).

    - ``"auto"`` (default) — use extended when the ``style`` extra is
      installed (spacy + textstat importable), else the 6-dim core.
    - ``"on"`` — require extended; SKIP if deps missing.
    - ``"off"`` — always use the 6-dim core, regardless of installed deps.

    Turning extended on changes the dimensionality of ``evidence["base_fp"]``
    / ``["ft_fp"]`` / ``["doc_fp"]`` from 6 to 9. Snapshot-test consumers
    should branch on ``evidence["schema_version"]`` rather than hardcode a
    length."""


class StyleFingerprintProbe(Probe):
    kind = "style_fingerprint"
    spec_cls = StyleFingerprintSpec
    category = "calibration"

    @classmethod
    def calibrate_spec(cls, ctx: RunContext) -> StyleFingerprintSpec | None:
        del ctx
        from dlm_sway.probes.base import SENTINEL_DOC, SENTINEL_PROMPTS

        return StyleFingerprintSpec(
            name="_calibration",
            kind="style_fingerprint",
            prompts=list(SENTINEL_PROMPTS),
            doc_reference=SENTINEL_DOC,
            max_new_tokens=64,  # faster than default
        )

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        assert isinstance(spec, StyleFingerprintSpec)
        if not spec.prompts:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.ERROR,
                score=None,
                message="no prompts provided",
            )
        doc_text = spec.doc_reference or (ctx.doc_text or "")
        if not doc_text.strip():
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.SKIP,
                score=None,
                message="no doc_reference (inline or from ctx.doc_text)",
            )

        # Resolve whether to use the 9-dim extended fingerprint.
        if spec.extended == "on":
            if not _has_style_extra():
                return ProbeResult(
                    name=spec.name,
                    kind=spec.kind,
                    verdict=Verdict.SKIP,
                    score=None,
                    message=(
                        "extended=on requires the [style] extra "
                        "(pip install 'dlm-sway[style]' + "
                        "'python -m spacy download en_core_web_sm')"
                    ),
                )
            use_extended = True
        elif spec.extended == "off":
            use_extended = False
        else:  # "auto"
            use_extended = _has_style_extra()

        base_samples: list[str] = []
        ft_samples: list[str] = []
        for prompt in spec.prompts:
            with ctx.backend.as_base() as b:
                base_samples.append(
                    b.generate(prompt, max_new_tokens=spec.max_new_tokens, seed=ctx.seed)
                )
            with ctx.backend.as_finetuned() as f:
                ft_samples.append(
                    f.generate(prompt, max_new_tokens=spec.max_new_tokens, seed=ctx.seed)
                )

        base_fp = fingerprint("\n".join(base_samples), extended=use_extended)
        ft_fp = fingerprint("\n".join(ft_samples), extended=use_extended)
        doc_fp = fingerprint(doc_text, extended=use_extended)

        # B4 fix: a degenerate ft fingerprint (all-empty generations →
        # zeros) used to coincidentally produce a positive cosine shift
        # because cos(ft-base, doc-base) ≈ cos(-base, doc-base) is often
        # positive. Detect that case and emit ERROR rather than PASS.
        ft_is_zero = bool(np.allclose(ft_fp, 0.0))
        ft_text_is_empty = all(not s.strip() for s in ft_samples)
        if ft_is_zero or ft_text_is_empty:
            return safe_finalize(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.ERROR,
                score=None,
                raw=None,
                evidence={
                    "base_fp": base_fp.tolist(),
                    "ft_fp": ft_fp.tolist(),
                    "doc_fp": doc_fp.tolist(),
                    "ft_text_is_empty": ft_text_is_empty,
                    "ft_fp_is_zero": ft_is_zero,
                    "extended": use_extended,
                    "schema_version": FINGERPRINT_SCHEMA_VERSION,
                    "weight": spec.weight,
                },
                message=(
                    "fine-tuned model produced empty / zero-fingerprint output — "
                    "cannot measure style shift on a degenerate ft view"
                ),
            )

        shift = _projection_shift(base_fp, ft_fp, doc_fp)

        stats = get_null_stats(ctx, spec.kind)
        z = z_score(shift, stats)
        verdict_z = verdict_from_z(z, spec.assert_z_gte)
        if verdict_z is not None:
            verdict = verdict_z
            score_val = score_from_z(z)
            score = score_val if score_val is not None else 0.0
            message = f"style_shift={shift:+.2f}, z={z:+.2f}σ vs null"
        else:
            verdict = Verdict.PASS if shift >= spec.assert_shift_gte else Verdict.FAIL
            # Score: 0 at no shift, 1 when ft moves a full doc-gap toward
            # doc; clamp to [0, 1].
            score = float(np.clip(shift, 0.0, 1.0))
            message = (
                f"style_shift={shift:+.2f} "
                f"({'toward' if shift > 0 else 'away from'} doc, "
                f"threshold={spec.assert_shift_gte}) "
                f"{no_calibration_note(spec.kind)}"
            )

        return safe_finalize(
            name=spec.name,
            kind=spec.kind,
            verdict=verdict,
            score=score,
            raw=shift,
            z_score=z,
            evidence={
                "base_fp": base_fp.tolist(),
                "ft_fp": ft_fp.tolist(),
                "doc_fp": doc_fp.tolist(),
                "style_shift": shift,
                "extended": use_extended,
                "schema_version": FINGERPRINT_SCHEMA_VERSION,
                "weight": spec.weight,
            },
            message=message,
        )


def _projection_shift(
    base: NDArray[np.float64], ft: NDArray[np.float64], doc: NDArray[np.float64]
) -> float:
    """Project (ft - base) onto (doc - base), normalized by ||doc - base||².

    Returns ``((ft - base) · (doc - base)) / ||doc - base||²``. Properties:

    - ``ft == base`` → 0 (no shift)
    - ``ft == doc`` → 1 (ft moved a full doc-gap toward doc)
    - ``ft`` moved opposite to doc → negative
    - ``doc == base`` (no doc gap to measure) → 0

    This replaces the older ``cos(ft-base, doc-base)`` which silently
    treated a zero ft-shift as a phantom positive correlation when
    ``-base`` happened to point in roughly the doc direction (B4).
    """
    a = ft - base
    b = doc - base
    nb_sq = float(np.dot(b, b))
    if nb_sq == 0.0:
        return 0.0
    return float(np.dot(a, b) / nb_sq)
