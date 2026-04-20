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

from dlm_sway.core.result import ProbeResult, Verdict
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z'-]*\b")
_PUNCTS = set(".,:;!?-—()[]\"'/")


def fingerprint(text: str) -> NDArray[np.float64]:
    """Return a 6-dim stylistic fingerprint for ``text``.

    Dimensions (all numeric, scaled to order-1):
      0. mean sentence length (words)  / 30.0
      1. std sentence length (words)   / 30.0
      2. type-token ratio              (already in [0,1])
      3. avg word length (chars)       / 10.0
      4. punctuation density per char  * 10.0
      5. paragraph density (1 / avg paragraph length in words) * 30.0
    """
    if not text.strip():
        return np.zeros(6, dtype=np.float64)

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


class StyleFingerprintProbe(Probe):
    kind = "style_fingerprint"
    spec_cls = StyleFingerprintSpec
    category = "calibration"

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

        base_fp = fingerprint("\n".join(base_samples))
        ft_fp = fingerprint("\n".join(ft_samples))
        doc_fp = fingerprint(doc_text)

        shift = _cosine_shift(base_fp, ft_fp, doc_fp)
        verdict = Verdict.PASS if shift >= spec.assert_shift_gte else Verdict.FAIL
        score = float(np.clip((shift + 1.0) / 2.0, 0.0, 1.0))

        return ProbeResult(
            name=spec.name,
            kind=spec.kind,
            verdict=verdict,
            score=score,
            raw=shift,
            evidence={
                "base_fp": base_fp.tolist(),
                "ft_fp": ft_fp.tolist(),
                "doc_fp": doc_fp.tolist(),
                "style_shift": shift,
                "weight": spec.weight,
            },
            message=(
                f"style_shift={shift:+.2f} "
                f"({'toward' if shift > 0 else 'away from'} doc, "
                f"threshold={spec.assert_shift_gte})"
            ),
        )


def _cosine_shift(
    base: NDArray[np.float64], ft: NDArray[np.float64], doc: NDArray[np.float64]
) -> float:
    """Cosine between (ft - base) and (doc - base) in fingerprint space."""
    a = ft - base
    b = doc - base
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
