"""Tests for :mod:`dlm_sway.core.sections`."""

from __future__ import annotations

from dlm_sway.core.sections import (
    Section,
    SectionPreference,
    SectionProbe,
    filter_kinds,
)


def test_default_field_types() -> None:
    s = Section(id="abc", kind="prose", content="hello world")
    assert s.probes == ()
    assert s.preferences == ()
    assert s.tag is None


def test_filter_kinds() -> None:
    sections = (
        Section(id="a", kind="prose", content="x"),
        Section(id="b", kind="instruction", content="y"),
        Section(id="c", kind="preference", content="z"),
    )
    only_prose = filter_kinds(sections, ("prose",))
    assert len(only_prose) == 1
    assert only_prose[0].id == "a"


def test_section_probe_and_preference() -> None:
    p = SectionProbe(prompt="Q", gold="A")
    assert p.prompt == "Q"
    pref = SectionPreference(prompt="P", chosen="good", rejected="bad")
    assert pref.chosen == "good"
