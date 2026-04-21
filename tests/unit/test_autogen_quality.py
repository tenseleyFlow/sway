"""Quality-of-output tests for the autogen YAML.

The audit's B8 finding was that ``style_fingerprint`` got the leading
sentence of a prose section as its prompt — which elicits doc
*continuation* (a content probe), not stylistic voice. Sprint 05
replaces that with a fixed list of stylistic-elicitation prompts. This
file pins the new contract.
"""

from __future__ import annotations

from pathlib import Path

from dlm_sway.core.sections import Section
from dlm_sway.integrations.dlm.autogen import (
    _STYLE_ELICITATION_PROMPTS,
    build_spec_dict,
)
from dlm_sway.integrations.dlm.resolver import DlmHandle


def _handle_with_prose_first_sentence() -> DlmHandle:
    """A handle whose only prose section starts with a strong, doc-specific
    opener — the kind of sentence that, under the old heuristic, would
    have leaked into the style probe."""
    sections = (
        Section(
            id="s1",
            kind="prose",
            content=(
                "The mitochondrion is the powerhouse of the cell. "
                "It generates ATP via oxidative phosphorylation. "
                "Inner-membrane folds called cristae increase surface area."
            ),
        ),
    )
    return DlmHandle(
        dlm_id="x",
        base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
        adapter_path=Path("/tmp/adapter"),
        sections=sections,
        doc_text="whole document",
    )


def test_style_prompts_use_elicitation_set_not_doc_content() -> None:
    """B8: style_fingerprint prompts come from the fixed elicitation set."""
    spec = build_spec_dict(_handle_with_prose_first_sentence())
    style_entry = next((e for e in spec["suite"] if e["kind"] == "style_fingerprint"), None)
    assert style_entry is not None, "autogen should emit a style_fingerprint entry"
    style_prompts = style_entry["prompts"]
    # Every prompt comes from the elicitation set.
    assert set(style_prompts) <= set(_STYLE_ELICITATION_PROMPTS)
    # No prompt smells like the leading prose sentence.
    assert not any("mitochondrion" in p.lower() for p in style_prompts)
    assert not any("powerhouse" in p.lower() for p in style_prompts)


def test_style_prompts_nonempty_even_without_prose() -> None:
    """The fixed list means the probe always has something to ask the model."""
    sections = (Section(id="i1", kind="instruction", content="What is X? X is Y.", probes=()),)
    handle = DlmHandle(
        dlm_id="x",
        base_model="b",
        adapter_path=Path("/tmp/a"),
        sections=sections,
        doc_text=None,
    )
    spec = build_spec_dict(handle)
    style_entry = next((e for e in spec["suite"] if e["kind"] == "style_fingerprint"), None)
    assert style_entry is not None
    assert len(style_entry["prompts"]) >= 4


def test_elicitation_prompts_are_open_ended() -> None:
    """A sanity check on the constant itself: each prompt invites prose,
    not a single-token completion."""
    for prompt in _STYLE_ELICITATION_PROMPTS:
        assert len(prompt) >= 30, f"prompt too short to elicit prose: {prompt!r}"
        assert prompt.endswith(".")


def _handle_with_many_instruction_probes(n: int) -> DlmHandle:
    """A handle rigged to produce at least ``n`` distinct instruction
    prompts (used to clear ``cluster_kl``'s 20-prompt floor)."""
    from dlm_sway.core.sections import SectionProbe

    probes = tuple(SectionProbe(prompt=f"Q{i}: what is topic {i}?", gold=f"A{i}") for i in range(n))
    sections = (
        Section(id="i1", kind="instruction", content="…", probes=probes),
        Section(
            id="p1",
            kind="prose",
            content="Prose sentence one. Prose sentence two. Prose sentence three.",
        ),
    )
    return DlmHandle(
        dlm_id="x",
        base_model="b",
        adapter_path=Path("/tmp/a"),
        sections=sections,
        doc_text=None,
    )


class TestSkippedProbesRollup:
    """F07 (Audit 03) — ``_render_annotated_yaml`` prepends a
    ``# skipped: <probe> (<reason>)`` block so users see which probes
    the autogen intentionally omitted, without diffing this module's
    docstring."""

    def test_prose_only_handle_omits_instruction_heavy_probes(self) -> None:
        """A .dlm with only PROSE sections skips adapter_revert +
        paraphrase_invariance + preference_flip + (with 1 section)
        section_internalization."""
        from dlm_sway.integrations.dlm.autogen import collect_skipped_probe_reasons

        handle = DlmHandle(
            dlm_id="x",
            base_model="b",
            adapter_path=Path("/tmp/a"),
            sections=(
                Section(
                    id="s1",
                    kind="prose",
                    content="One paragraph of prose. Second sentence.",
                ),
            ),
            doc_text="doc",
        )
        skipped = collect_skipped_probe_reasons(handle)
        skipped_kinds = {k for k, _ in skipped}
        assert "adapter_revert" in skipped_kinds
        assert "paraphrase_invariance" in skipped_kinds
        assert "preference_flip" in skipped_kinds
        assert "section_internalization" in skipped_kinds
        # delta_kl should NOT be skipped — prose provides a fallback
        # prompt pool.
        assert "delta_kl" not in skipped_kinds

    def test_instruction_only_handle_omits_prose_heavy_probes(self) -> None:
        """An instruction-only doc skips external_perplexity + leakage."""
        from dlm_sway.core.sections import SectionProbe
        from dlm_sway.integrations.dlm.autogen import collect_skipped_probe_reasons

        handle = DlmHandle(
            dlm_id="x",
            base_model="b",
            adapter_path=Path("/tmp/a"),
            sections=(
                Section(
                    id="i1",
                    kind="instruction",
                    content="Q/A",
                    probes=(SectionProbe(prompt="Q?", gold="A"),),
                ),
            ),
            doc_text=None,
        )
        skipped = collect_skipped_probe_reasons(handle)
        skipped_kinds = {k for k, _ in skipped}
        assert "external_perplexity" in skipped_kinds
        assert "leakage" in skipped_kinds

    def test_rendered_yaml_carries_skipped_block(self, tmp_path: Path) -> None:
        """End-to-end: on a minimal prose-only .dlm, the rendered YAML
        header has the ``# skipped:`` lines."""
        from dlm_sway.integrations.dlm.autogen import (
            _render_annotated_yaml,
            build_spec_dict,
            collect_skipped_probe_reasons,
        )

        handle = DlmHandle(
            dlm_id="x",
            base_model="b",
            adapter_path=Path("/tmp/a"),
            sections=(Section(id="s1", kind="prose", content="Short prose."),),
            doc_text="doc",
        )
        dlm_path = tmp_path / "demo.dlm"
        dlm_path.write_text("# empty")
        spec = build_spec_dict(handle, dlm_source="demo.dlm")
        skipped = collect_skipped_probe_reasons(handle)
        rendered = _render_annotated_yaml(spec, handle, dlm_path, skipped=skipped)
        assert "# skipped: adapter_revert" in rendered
        assert "# skipped: preference_flip" in rendered
        assert "(no " in rendered  # reasons start with "no ..."

    def test_rendered_yaml_omits_skipped_block_when_all_probes_fit(self) -> None:
        """A heavily-populated doc that triggers every probe emits no
        ``# skipped:`` lines."""
        from dlm_sway.core.sections import SectionPreference, SectionProbe
        from dlm_sway.integrations.dlm.autogen import (
            _render_annotated_yaml,
            build_spec_dict,
            collect_skipped_probe_reasons,
        )

        probes = tuple(SectionProbe(prompt=f"Q{i}?", gold=f"A{i}") for i in range(25))
        preferences = (SectionPreference(prompt="P1", chosen="good", rejected="bad"),)
        handle = DlmHandle(
            dlm_id="x",
            base_model="b",
            adapter_path=Path("/tmp/a"),
            sections=(
                Section(id="i1", kind="instruction", content="Q/A", probes=probes),
                Section(
                    id="p1",
                    kind="prose",
                    content="A first prose sentence. A second. A third.",
                ),
                Section(
                    id="pref1",
                    kind="preference",
                    content="pref",
                    preferences=preferences,
                ),
            ),
            doc_text="doc",
        )
        spec = build_spec_dict(handle)
        skipped = collect_skipped_probe_reasons(handle)
        rendered = _render_annotated_yaml(spec, handle, Path("/tmp/demo.dlm"), skipped=skipped)
        assert "# skipped:" not in rendered


class TestPortableDlmSource:
    """F09 (Audit 03) — ``_portable_dlm_source`` emits a cwd-relative
    path when the ``.dlm`` lives inside the cwd (survives CI checkout),
    absolute path when it lives elsewhere.
    """

    def test_cwd_relative_when_inside(self, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        from dlm_sway.integrations.dlm.autogen import _portable_dlm_source

        # Set cwd to tmp_path; drop a .dlm inside a subdir.
        subdir = tmp_path / "src"
        subdir.mkdir()
        dlm_file = subdir / "demo.dlm"
        dlm_file.write_text("# empty\n")
        monkeypatch.chdir(tmp_path)
        source = _portable_dlm_source(dlm_file)
        assert source == "src/demo.dlm"
        # Not an absolute path — the whole point of F09.
        assert not Path(source).is_absolute()

    def test_absolute_when_outside(self, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """A ``.dlm`` somewhere outside the cwd falls back to its
        absolute path — relative-ization would point at a nonexistent
        parent directory on a fresh checkout."""
        from dlm_sway.integrations.dlm.autogen import _portable_dlm_source

        # cwd inside tmp_path; .dlm lives in a sibling tree.
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        sibling = tmp_path / "other"
        sibling.mkdir()
        dlm_file = sibling / "demo.dlm"
        dlm_file.write_text("# empty\n")
        monkeypatch.chdir(cwd)
        source = _portable_dlm_source(dlm_file)
        assert Path(source).is_absolute()
        assert source == str(dlm_file.resolve())


class TestAutogenClusterKL:
    """F07 — autogen emits ``cluster_kl`` when the prompt pool has
    enough entries to clear S16's ``min_prompts=20`` floor, and omits
    it otherwise."""

    def test_emits_cluster_kl_when_prompt_pool_is_large(self) -> None:
        spec = build_spec_dict(_handle_with_many_instruction_probes(25))
        entry = next((e for e in spec["suite"] if e["kind"] == "cluster_kl"), None)
        assert entry is not None, "autogen should emit cluster_kl on large prompt pools"
        assert entry["num_clusters"] == 5
        assert entry["min_prompts"] == 20
        assert len(entry["prompts"]) >= 20
        # Cap at 64 so a doc with hundreds of probes doesn't explode
        # the cluster runtime.
        assert len(entry["prompts"]) <= 64

    def test_omits_cluster_kl_on_small_prompt_pool(self) -> None:
        """Under 20 prompts → omit the entry. The probe would SKIP
        anyway; skipping emission keeps the autogen'd YAML tidy."""
        spec = build_spec_dict(_handle_with_many_instruction_probes(5))
        entry = next((e for e in spec["suite"] if e["kind"] == "cluster_kl"), None)
        assert entry is None

    def test_prompts_deduplicated(self) -> None:
        """No duplicate entries (instruction prompts + prose leading
        sentences are merged but must not repeat verbatim)."""
        spec = build_spec_dict(_handle_with_many_instruction_probes(30))
        entry = next((e for e in spec["suite"] if e["kind"] == "cluster_kl"), None)
        assert entry is not None
        assert len(entry["prompts"]) == len(set(entry["prompts"]))
