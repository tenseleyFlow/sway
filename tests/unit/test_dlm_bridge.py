"""Tests for :mod:`dlm_sway.integrations.dlm`.

The bridge imports ``dlm.*`` modules lazily. We mock those via
``sys.modules`` injection so the tests run without the ``dlm-sway[dlm]``
extra installed. A full end-to-end integration test against a real
``.dlm`` lives under ``tests/integration/``.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def fake_dlm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Install a fake ``dlm`` package so the resolver can import."""

    # Build synthetic parsed .dlm structure.
    @dataclass
    class _Frontmatter:
        dlm_id: str = "01TESTULID"
        base_model: str = "smollm2-135m"

    @dataclass
    class _Section:
        section_id: str
        type: str
        content: str
        tag: str | None = None

    @dataclass
    class _Parsed:
        frontmatter: _Frontmatter
        sections: tuple[_Section, ...]

    def _parse_file(_path: Path):  # type: ignore[no-untyped-def]
        return _Parsed(
            frontmatter=_Frontmatter(),
            sections=(
                _Section(
                    section_id="prose-1",
                    type="PROSE",
                    content="This is a prose section with some information. Further detail follows.",
                ),
                _Section(
                    section_id="instr-1",
                    type="INSTRUCTION",
                    content="### Q\nWhat is X?\n\n### A\nX is a concept\n",
                ),
                _Section(
                    section_id="pref-1",
                    type="PREFERENCE",
                    content="chosen/rejected triple",
                ),
            ),
        )

    # Fake ``dlm.doc.parser`` module.
    dlm_pkg = types.ModuleType("dlm")
    dlm_doc = types.ModuleType("dlm.doc")
    dlm_doc_parser = types.ModuleType("dlm.doc.parser")
    dlm_doc_parser.parse_file = _parse_file  # type: ignore[attr-defined]

    # Fake ``dlm.store.paths`` that returns a resolvable path.
    dlm_store = types.ModuleType("dlm.store")
    dlm_store_paths = types.ModuleType("dlm.store.paths")

    adapter_dir = tmp_path / "adapter_v1"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    class _StorePath:
        def __init__(self, path: Path) -> None:
            self._p = path

        def resolve_current_adapter(self) -> Path:
            return self._p

    def _for_dlm(_dlm_id: str) -> _StorePath:
        return _StorePath(adapter_dir)

    dlm_store_paths.StorePath = _StorePath  # type: ignore[attr-defined]
    dlm_store_paths.for_dlm = _for_dlm  # type: ignore[attr-defined]

    # Fake base-model resolver — returns a stub with an ``hf_id`` attribute.
    dlm_base = types.ModuleType("dlm.base_models")

    @dataclass
    class _BaseSpec:
        hf_id: str
        key: str

    def _resolve(key: str) -> _BaseSpec:
        return _BaseSpec(hf_id="HuggingFaceTB/SmolLM2-135M-Instruct", key=key)

    dlm_base.resolve = _resolve  # type: ignore[attr-defined]

    # Fake instruction / preference parsers.
    dlm_data = types.ModuleType("dlm.data")
    dlm_data_instr = types.ModuleType("dlm.data.instruction_parser")
    dlm_data_pref = types.ModuleType("dlm.data.preference_parser")

    @dataclass
    class _QAPair:
        question: str
        answer: str

    @dataclass
    class _Triple:
        prompt: str
        chosen: str
        rejected: str

    def _parse_instr(body: str, *, section_id: str) -> list[_QAPair]:
        del section_id
        out: list[_QAPair] = []
        parts = body.split("### Q")
        for part in parts[1:]:
            q_block, _, a_block = part.partition("### A")
            q = q_block.strip()
            a = a_block.strip()
            if q and a:
                out.append(_QAPair(question=q, answer=a))
        return out

    def _parse_pref(body: str, *, section_id: str) -> list[_Triple]:
        del body, section_id
        return [_Triple(prompt="Which?", chosen="good answer", rejected="bad answer")]

    dlm_data_instr.parse_instruction_body = _parse_instr  # type: ignore[attr-defined]
    dlm_data_pref.parse_preference_body = _parse_pref  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "dlm", dlm_pkg)
    monkeypatch.setitem(sys.modules, "dlm.doc", dlm_doc)
    monkeypatch.setitem(sys.modules, "dlm.doc.parser", dlm_doc_parser)
    monkeypatch.setitem(sys.modules, "dlm.store", dlm_store)
    monkeypatch.setitem(sys.modules, "dlm.store.paths", dlm_store_paths)
    monkeypatch.setitem(sys.modules, "dlm.base_models", dlm_base)
    monkeypatch.setitem(sys.modules, "dlm.data", dlm_data)
    monkeypatch.setitem(sys.modules, "dlm.data.instruction_parser", dlm_data_instr)
    monkeypatch.setitem(sys.modules, "dlm.data.preference_parser", dlm_data_pref)

    # Return a path to a fake .dlm file (the parser won't actually read it).
    dlm_file = tmp_path / "doc.dlm"
    dlm_file.write_text("---\ndlm_id: 01TEST\n---\n\nbody\n", encoding="utf-8")
    return dlm_file


def test_resolve_dlm_maps_sections(fake_dlm: Path) -> None:
    from dlm_sway.integrations.dlm.resolver import resolve_dlm

    handle = resolve_dlm(fake_dlm)
    assert handle.dlm_id == "01TESTULID"
    assert handle.base_model == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert handle.adapter_path is not None
    assert handle.adapter_path.exists()
    assert len(handle.sections) == 3
    # Kinds normalized from uppercase dlm enum values.
    assert {s.kind for s in handle.sections} == {"prose", "instruction", "preference"}
    # Instruction Q/A pair survived the translation.
    instr = next(s for s in handle.sections if s.kind == "instruction")
    assert instr.probes
    assert instr.probes[0].prompt == "What is X?"
    # Preference triple too.
    pref = next(s for s in handle.sections if s.kind == "preference")
    assert pref.preferences
    assert pref.preferences[0].chosen == "good answer"


def test_resolve_without_dlm_installed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """resolve_dlm surfaces a SwayError when the dlm package is missing."""
    # Wipe any cached dlm modules so the lazy import fails.
    for mod in list(sys.modules):
        if mod == "dlm" or mod.startswith("dlm."):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name.startswith("dlm."):
            raise ImportError("missing extra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from dlm_sway.core.errors import SwayError
    from dlm_sway.integrations.dlm.resolver import resolve_dlm

    with pytest.raises(SwayError, match="dlm package not installed"):
        resolve_dlm(tmp_path / "doc.dlm")


def test_autogen_writes_complete_suite(fake_dlm: Path, tmp_path: Path) -> None:
    from dlm_sway.integrations.dlm.autogen import write_sway_yaml

    out = tmp_path / "sway.yaml"
    write_sway_yaml(fake_dlm, out)
    data = yaml.safe_load(out.read_text(encoding="utf-8"))

    assert data["version"] == 1
    assert data["models"]["base"]["base"] == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert data["models"]["ft"]["adapter"] is not None
    assert data["dlm_source"] == str(fake_dlm.resolve())

    kinds = {entry["kind"] for entry in data["suite"]}
    # The full 11-primitive battery minus nothing is present (some may
    # be skipped when data is absent, but here we have one of every
    # section type).
    expected = {
        "null_adapter",
        "delta_kl",
        "adapter_revert",
        "prompt_collapse",
        "section_internalization",
        "paraphrase_invariance",
        "preference_flip",
        "style_fingerprint",
        "calibration_drift",
        "leakage",
        "adapter_ablation",
    }
    assert expected <= kinds, f"missing: {expected - kinds}"


def test_build_spec_dict_skips_preference_when_absent() -> None:
    from dlm_sway.core.sections import Section
    from dlm_sway.integrations.dlm.autogen import build_spec_dict
    from dlm_sway.integrations.dlm.resolver import DlmHandle

    sections = (
        Section(id="a", kind="prose", content="A prose section. Second sentence."),
        Section(id="b", kind="prose", content="Another prose section."),
    )
    handle = DlmHandle(
        dlm_id="x",
        base_model="base",
        adapter_path=Path("/tmp/adapter"),
        sections=sections,
        doc_text="whole document",
    )
    spec = build_spec_dict(handle)
    kinds = {entry["kind"] for entry in spec["suite"]}
    assert "preference_flip" not in kinds
    assert "section_internalization" in kinds
