"""Tests for :mod:`dlm_sway.probes.tool_use_fidelity`."""

from __future__ import annotations

import json
from typing import Any

from dlm_sway.backends.dummy import DummyDifferentialBackend, DummyResponses
from dlm_sway.core.result import Verdict
from dlm_sway.probes.base import RunContext, build_probe
from dlm_sway.probes.tool_use_fidelity import (
    _extract_fenced_json,
    _field_disagreement,
    _first_balanced_braces,
    _matches_schema,
    _parse_tool_call,
)

# ---------------------------------------------------------------------------
# Schema fixtures
# ---------------------------------------------------------------------------


SEARCH_SPEC: dict[str, Any] = {
    "name": "search_web",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        "required": ["query"],
    },
}

CALCULATOR_SPEC: dict[str, Any] = {
    "name": "calculator",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
    },
}


def _backend(
    generations_base: dict[str, str], generations_ft: dict[str, str]
) -> DummyDifferentialBackend:
    """Build a dummy backend keyed by prompt for both views."""
    return DummyDifferentialBackend(
        base=DummyResponses(generations=generations_base),
        ft=DummyResponses(generations=generations_ft),
    )


def _call(name: str, **args: Any) -> str:
    return json.dumps({"name": name, "arguments": args})


# ---------------------------------------------------------------------------
# End-to-end probe behavior
# ---------------------------------------------------------------------------


class TestProbeBehavior:
    def test_pass_when_ft_preserves_validity_and_no_hallucination(self) -> None:
        """Both views produce schema-valid calls naming the gold tool."""
        prompt = "Search the web for cats"
        backend = _backend(
            generations_base={prompt: _call("search_web", query="cats")},
            generations_ft={prompt: _call("search_web", query="cats", max_results=5)},
        )
        probe, spec = build_probe(
            {
                "name": "tuf",
                "kind": "tool_use_fidelity",
                "cases": [
                    {
                        "prompt": prompt,
                        "tool_spec": SEARCH_SPEC,
                        "gold_tool_name": "search_web",
                    }
                ],
            }
        )
        result = probe.run(spec, RunContext(backend=backend))
        assert result.verdict == Verdict.PASS, result.message
        assert result.evidence["json_valid_rate_ft"] == 1.0
        assert result.evidence["json_valid_rate_base"] == 1.0
        assert result.evidence["validity_delta"] == 0.0
        assert result.evidence["hallucination_rate"] == 0.0

    def test_fail_when_ft_breaks_json(self) -> None:
        """ft regresses on validity → FAIL on the validity-delta floor."""
        prompt = "Search the web for cats"
        backend = _backend(
            generations_base={prompt: _call("search_web", query="cats")},
            generations_ft={prompt: "I cannot help with that request."},
        )
        probe, spec = build_probe(
            {
                "name": "tuf",
                "kind": "tool_use_fidelity",
                "cases": [
                    {
                        "prompt": prompt,
                        "tool_spec": SEARCH_SPEC,
                        "gold_tool_name": "search_web",
                    }
                ],
            }
        )
        result = probe.run(spec, RunContext(backend=backend))
        assert result.verdict == Verdict.FAIL
        assert result.evidence["json_valid_rate_ft"] == 0.0
        assert result.evidence["json_valid_rate_base"] == 1.0
        assert result.evidence["validity_delta"] == -1.0

    def test_fail_when_ft_hallucinates_tool_name(self) -> None:
        """Schema-valid call but wrong tool name → fails hallucination cap."""
        prompt = "Search the web for cats"
        backend = _backend(
            generations_base={prompt: _call("search_web", query="cats")},
            # ft picks a tool that isn't in `allowed_tools`.
            generations_ft={prompt: _call("delete_database", query="cats")},
        )
        probe, spec = build_probe(
            {
                "name": "tuf",
                "kind": "tool_use_fidelity",
                "cases": [
                    {
                        "prompt": prompt,
                        "tool_spec": SEARCH_SPEC,
                        "gold_tool_name": "search_web",
                    }
                ],
                "allowed_tools": ["search_web"],
            }
        )
        result = probe.run(spec, RunContext(backend=backend))
        assert result.verdict == Verdict.FAIL
        assert result.evidence["hallucination_rate"] == 1.0

    def test_skip_when_no_cases(self) -> None:
        backend = _backend({}, {})
        probe, spec = build_probe({"name": "tuf", "kind": "tool_use_fidelity"})
        result = probe.run(spec, RunContext(backend=backend))
        assert result.verdict == Verdict.SKIP

    def test_pass_with_arg_drift_under_validity_floor(self) -> None:
        """Argument value drift counts as evidence but doesn't fail v1."""
        prompt = "calc 2+2"
        backend = _backend(
            generations_base={prompt: _call("calculator", a=2, b=2)},
            generations_ft={prompt: _call("calculator", a=2, b=3)},  # b drifted
        )
        probe, spec = build_probe(
            {
                "name": "tuf",
                "kind": "tool_use_fidelity",
                "cases": [
                    {
                        "prompt": prompt,
                        "tool_spec": CALCULATOR_SPEC,
                        "gold_tool_name": "calculator",
                    }
                ],
            }
        )
        result = probe.run(spec, RunContext(backend=backend))
        assert result.verdict == Verdict.PASS
        # 1/2 leaf disagreements (b differs, a matches).
        assert result.evidence["mean_arg_disagreement"] == 0.5

    def test_allowed_tools_check_supersedes_gold(self) -> None:
        """When allowed_tools is set, gold is informational; surface gate wins."""
        prompt = "search"
        backend = _backend(
            generations_base={prompt: _call("search_web", query="x")},
            # ft picks a tool that's in the allowed surface but != gold;
            # the surface gate accepts it → no hallucination.
            generations_ft={prompt: _call("search_web_v2", query="x")},
        )
        probe, spec = build_probe(
            {
                "name": "tuf",
                "kind": "tool_use_fidelity",
                "cases": [
                    {
                        "prompt": prompt,
                        "tool_spec": {**SEARCH_SPEC, "name": "search_web_v2"},
                        "gold_tool_name": "search_web",
                    }
                ],
                "allowed_tools": ["search_web", "search_web_v2"],
            }
        )
        result = probe.run(spec, RunContext(backend=backend))
        assert result.evidence["hallucination_rate"] == 0.0

    def test_fenced_json_in_generation_parses(self) -> None:
        """A model that wraps the call in ```json ... ``` is still valid."""
        prompt = "search for cats"
        fenced = "Here you go:\n```json\n" + _call("search_web", query="cats") + "\n```"
        backend = _backend(
            generations_base={prompt: _call("search_web", query="cats")},
            generations_ft={prompt: fenced},
        )
        probe, spec = build_probe(
            {
                "name": "tuf",
                "kind": "tool_use_fidelity",
                "cases": [
                    {
                        "prompt": prompt,
                        "tool_spec": SEARCH_SPEC,
                        "gold_tool_name": "search_web",
                    }
                ],
            }
        )
        result = probe.run(spec, RunContext(backend=backend))
        assert result.evidence["json_valid_rate_ft"] == 1.0


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


class TestParseToolCall:
    def test_whole_text_is_json(self) -> None:
        out = _parse_tool_call('{"name": "x", "arguments": {}}')
        assert out == {"name": "x", "arguments": {}}

    def test_embedded_braces_extracted(self) -> None:
        out = _parse_tool_call('I will call: {"name": "x", "arguments": {"q": 1}} now.')
        assert out is not None
        assert out["name"] == "x"

    def test_fenced_json_block_extracted(self) -> None:
        out = _parse_tool_call('Sure.\n```json\n{"name": "x", "arguments": {}}\n```\nDone.')
        assert out == {"name": "x", "arguments": {}}

    def test_returns_none_for_pure_prose(self) -> None:
        assert _parse_tool_call("I'm sorry, I can't do that.") is None

    def test_returns_none_for_non_dict_root(self) -> None:
        assert _parse_tool_call("[1, 2, 3]") is None

    def test_returns_none_for_unbalanced_braces(self) -> None:
        assert _parse_tool_call("{not closed") is None

    def test_braces_inside_string_dont_throw_off_balance(self) -> None:
        # The inner `}` is inside a string literal — must not pop the depth.
        out = _parse_tool_call('text {"name": "x", "arguments": {"q": "}"}}')
        assert out is not None
        assert out["arguments"]["q"] == "}"


class TestExtractFencedJson:
    def test_extracts_with_language_tag(self) -> None:
        body = _extract_fenced_json('pre\n```json\n{"a": 1}\n```\npost')
        assert body == '{"a": 1}'

    def test_extracts_without_language_tag(self) -> None:
        body = _extract_fenced_json('```\n{"a": 1}\n```')
        assert body == '{"a": 1}'

    def test_returns_none_when_no_fence(self) -> None:
        assert _extract_fenced_json("no fences here") is None

    def test_returns_none_when_unclosed(self) -> None:
        assert _extract_fenced_json('```json\n{"a": 1}') is None


class TestFirstBalancedBraces:
    def test_simple_object(self) -> None:
        assert _first_balanced_braces('text {"a": 1} more') == '{"a": 1}'

    def test_nested(self) -> None:
        assert _first_balanced_braces('{"a": {"b": 1}} trailing') == '{"a": {"b": 1}}'

    def test_no_braces_returns_none(self) -> None:
        assert _first_balanced_braces("plain text") is None

    def test_unbalanced_returns_none(self) -> None:
        assert _first_balanced_braces("{unclosed") is None


# ---------------------------------------------------------------------------
# Schema validator
# ---------------------------------------------------------------------------


class TestMatchesSchema:
    def test_full_valid_call(self) -> None:
        call = {"name": "search_web", "arguments": {"query": "cats", "max_results": 5}}
        assert _matches_schema(call, SEARCH_SPEC) is True

    def test_missing_name_fails(self) -> None:
        call = {"arguments": {"query": "cats"}}
        assert _matches_schema(call, SEARCH_SPEC) is False

    def test_missing_arguments_fails(self) -> None:
        call = {"name": "search_web"}
        assert _matches_schema(call, SEARCH_SPEC) is False

    def test_missing_required_arg_fails(self) -> None:
        call = {"name": "search_web", "arguments": {"max_results": 5}}
        assert _matches_schema(call, SEARCH_SPEC) is False

    def test_wrong_type_fails(self) -> None:
        call = {"name": "search_web", "arguments": {"query": 123}}
        assert _matches_schema(call, SEARCH_SPEC) is False

    def test_extra_unknown_field_allowed(self) -> None:
        """Real models add benign extras; we tolerate them."""
        call = {
            "name": "search_web",
            "arguments": {"query": "cats", "_internal_id": "abc"},
        }
        assert _matches_schema(call, SEARCH_SPEC) is True

    def test_bool_rejected_as_integer(self) -> None:
        """``isinstance(True, int)`` is True — guard against silent acceptance."""
        spec = {
            "name": "x",
            "parameters": {
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
            },
        }
        assert _matches_schema({"name": "x", "arguments": {"n": True}}, spec) is False
        assert _matches_schema({"name": "x", "arguments": {"n": 5}}, spec) is True

    def test_number_accepts_int_and_float(self) -> None:
        assert (
            _matches_schema(
                {"name": "calculator", "arguments": {"a": 1, "b": 2.5}}, CALCULATOR_SPEC
            )
            is True
        )

    def test_array_type_check(self) -> None:
        spec = {
            "name": "x",
            "parameters": {
                "type": "object",
                "properties": {"items": {"type": "array"}},
                "required": ["items"],
            },
        }
        assert _matches_schema({"name": "x", "arguments": {"items": [1, 2]}}, spec) is True
        assert _matches_schema({"name": "x", "arguments": {"items": "nope"}}, spec) is False


# ---------------------------------------------------------------------------
# Field disagreement
# ---------------------------------------------------------------------------


class TestFieldDisagreement:
    def test_identical_calls_zero(self) -> None:
        a = {"arguments": {"x": 1, "y": "hi"}}
        b = {"arguments": {"x": 1, "y": "hi"}}
        assert _field_disagreement(a, b) == 0.0

    def test_one_field_drifted(self) -> None:
        a = {"arguments": {"x": 1, "y": "hi"}}
        b = {"arguments": {"x": 1, "y": "bye"}}
        assert _field_disagreement(a, b) == 0.5

    def test_missing_field_counts_as_disagreement(self) -> None:
        a = {"arguments": {"x": 1, "y": "hi"}}
        b = {"arguments": {"x": 1}}
        assert _field_disagreement(a, b) == 0.5

    def test_nested_dicts_walked(self) -> None:
        a = {"arguments": {"outer": {"inner": "a"}}}
        b = {"arguments": {"outer": {"inner": "b"}}}
        assert _field_disagreement(a, b) == 1.0

    def test_lists_treated_as_leaves(self) -> None:
        a = {"arguments": {"items": [1, 2, 3]}}
        b = {"arguments": {"items": [3, 2, 1]}}
        # Different ordering → list leaves differ → full disagreement.
        assert _field_disagreement(a, b) == 1.0

    def test_empty_args_zero(self) -> None:
        assert _field_disagreement({"arguments": {}}, {"arguments": {}}) == 0.0

    def test_non_dict_arguments_full_disagreement(self) -> None:
        assert _field_disagreement({"arguments": "oops"}, {"arguments": {}}) == 1.0

    def test_present_none_distinct_from_absent(self) -> None:
        """A field with value ``None`` ≠ a field that's absent."""
        a = {"arguments": {"x": None}}
        b = {"arguments": {}}
        assert _field_disagreement(a, b) == 1.0


# ---------------------------------------------------------------------------
# Calibration spec — sanity check the null-adapter handoff
# ---------------------------------------------------------------------------


class TestCalibrateSpec:
    def test_calibrate_spec_returns_two_sentinels(self) -> None:
        from dlm_sway.probes.tool_use_fidelity import ToolUseFidelityProbe

        cspec = ToolUseFidelityProbe.calibrate_spec(RunContext())
        assert cspec is not None
        assert len(cspec.cases) == 2
        assert cspec.kind == "tool_use_fidelity"
