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
