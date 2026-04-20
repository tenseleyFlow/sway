"""A small, built-in general-knowledge probe pack for C2.

Each item is a ``(prompt, gold)`` pair where ``gold`` is the next few
tokens a competent base model should assign high probability to. The
items are deliberately *factually trivial* — the point isn't "does the
model know this?" but "did the fine-tune forget this?" — so the pack
skews toward grade-school geography, chemistry, arithmetic, and
high-frequency idiom.

A real v1.0 will ship a 200-item pack sliced from TriviaQA + SQuAD +
OpenBookQA. This 30-item seed lets the probe ship today and catches the
most egregious over-fit cases.
"""

from __future__ import annotations

from typing import Final

CalibrationItem = tuple[str, str]

BUILT_IN_PACK: Final[tuple[CalibrationItem, ...]] = (
    # Geography
    ("The capital of France is", " Paris"),
    ("The capital of Japan is", " Tokyo"),
    ("The largest ocean on Earth is the", " Pacific"),
    ("Mount Everest is located on the border of Nepal and", " China"),
    ("The longest river in South America is the", " Amazon"),
    # Natural sciences
    ("Water freezes at zero degrees", " Celsius"),
    ("The chemical symbol for gold is", " Au"),
    ("Light travels faster than", " sound"),
    ("Plants convert sunlight into energy through", " photosynthesis"),
    ("The Earth orbits around the", " Sun"),
    # Arithmetic
    ("Two plus two equals", " four"),
    ("Ten times ten equals", " one hundred"),
    ("Half of one hundred is", " fifty"),
    ("A dozen means", " twelve"),
    # Language and idiom
    ("A rose by any other name would smell as", " sweet"),
    ("To be or not to be, that is the", " question"),
    ("The early bird catches the", " worm"),
    ("Actions speak louder than", " words"),
    ("A picture is worth a thousand", " words"),
    # History
    ("World War II ended in the year", " 1945"),
    ("The first president of the United States was", " George Washington"),
    ("The Berlin Wall fell in", " 1989"),
    # Biology
    ("Humans have twenty", " fingers and toes"),
    ("The human body has two", " lungs"),
    ("Blood is pumped through the body by the", " heart"),
    # Technology
    ("HTML stands for HyperText", " Markup Language"),
    ("The World Wide Web was invented by Tim", " Berners-Lee"),
    # Miscellaneous
    ("One year has", " 365 days"),
    ("A week has seven", " days"),
    ("There are seven colors in a", " rainbow"),
)
"""30 items covering geography, science, arithmetic, language, history,
biology, and technology. Pulled from public-domain grade-school facts so
there's no licensing concern about shipping with the wheel."""
