"""Tool-use fidelity — does the adapter preserve the base's tool-call format?

LoRA fine-tunes increasingly target tool-use behavior: function-calling
JSON schemas, MCP tool plans, code-execution gating. An adapter that
accidentally degrades JSON-schema validity or starts hallucinating tool
names while acing every other probe is a silent production failure.

For each ``(prompt, tool_spec, gold_tool_name)`` case the probe greedy-
generates from both views and computes three independent signals:

- **JSON-schema validity delta** — ``ft_valid_rate − base_valid_rate``.
  Negative values mean the adapter degraded the base's tool-call
  formatting; positive values mean it improved (or the base was
  already tool-clueless).
- **Argument-field disagreement rate** — over cases where *both* views
  produced a schema-valid call, the leaf-field disagreement rate
  between ``ft_call.arguments`` and ``base_call.arguments``. Catches
  numeric / string drift inside an otherwise-well-formed call.
  (We deliberately ship leaf-field equality rather than per-token KL
  on argument values: per-token KL requires alignment between two
  decoded strings, which is the same hard problem we explicitly
  defer past v1.)
- **Tool-name hallucination rate** — over schema-valid ft calls, the
  fraction whose ``name`` is not in ``allowed_tools`` (when the user
  declared a tool surface) or differs from ``gold_tool_name`` (when
  no surface is declared).

Composite verdict logic:

- The pass criterion is compound — validity delta above the floor AND
  hallucination rate below the cap. Argument disagreement is
  informational on the v1 surface.
- ``json_valid_rate_ft`` is the metric that's z-scored against the
  null-adapter baseline. A null adapter should produce essentially no
  schema-valid calls, so ``z >= assert_z_gte`` is the principled
  "the adapter actually preserved tool-call structure" claim.

JSON-schema check is deliberately minimal: this v1 implementation
validates ``required`` membership and per-field type tags from a tiny
OpenAI-flavored subset (``string``, ``number``, ``integer``, ``boolean``,
``object``, ``array``). We don't pull in the ``jsonschema`` package —
the OpenAI-style spec is small enough to validate directly, and core
sway dependencies stay lean.
"""

from __future__ import annotations

import json
import statistics
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dlm_sway.core.result import ProbeResult, Verdict, safe_finalize
from dlm_sway.core.stats import bootstrap_ci
from dlm_sway.probes._zscore import (
    no_calibration_note,
    score_from_z,
    verdict_from_z,
    z_score,
    z_scores_by_rank,
)
from dlm_sway.probes.base import Probe, ProbeSpec, RunContext
from dlm_sway.probes.null_adapter import get_null_stats, get_null_stats_by_rank


class ToolUseCase(BaseModel):
    """One ``(prompt, tool_spec, gold_tool_name)`` case.

    ``tool_spec`` follows the OpenAI function-calling shape::

        {
            "name": "search_web",
            "description": "...",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"}
                },
                "required": ["query"]
            }
        }

    ``gold_tool_name`` is the tool the case expects ft to call. It's
    used by the hallucination check when no broader ``allowed_tools``
    list is declared on the spec.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str
    tool_spec: dict[str, Any]
    gold_tool_name: str
    max_new_tokens: int = 256


class ToolUseFidelitySpec(ProbeSpec):
    """Spec for ``kind: tool_use_fidelity``."""

    kind: Literal["tool_use_fidelity"] = "tool_use_fidelity"
    cases: list[ToolUseCase] = Field(default_factory=list, min_length=0)
    """Inline cases. Empty list → probe SKIPs (the .dlm autogen path
    leaves this empty unless a tool-use template seeded the doc)."""
    allowed_tools: list[str] | None = None
    """Optional tool-surface declaration. When set, hallucination is
    ``ft.name not in allowed_tools``. When ``None``, hallucination is
    ``ft.name != case.gold_tool_name`` per-case."""
    assert_validity_delta_gte: float = -0.05
    """Pass criterion on JSON-schema validity. Default tolerates a 5pp
    drop from base — anything worse is an adapter regression."""
    assert_hallucination_lte: float = 0.10
    """Pass criterion on tool-name hallucination. Default 10% — above
    that the adapter is actively inventing tools."""
    assert_z_gte: float = 3.0
    """Z-score pass criterion on ``json_valid_rate_ft`` against the
    null-adapter baseline. The principled signal — preferred over
    the raw thresholds when null calibration ran."""


class ToolUseFidelityProbe(Probe):
    """The "did the LoRA preserve tool-call format?" probe."""

    kind = "tool_use_fidelity"
    spec_cls = ToolUseFidelitySpec
    category = "attribution"

    @classmethod
    def calibrate_spec(cls, ctx: RunContext) -> ToolUseFidelitySpec | None:
        """Two trivial sentinel cases for null calibration.

        A null (random-noise) adapter should produce essentially no
        schema-valid output here, so ``json_valid_rate`` clusters
        tightly near zero — exactly what we want as the denominator
        when z-scoring the real adapter's validity rate.
        """
        del ctx
        return ToolUseFidelitySpec(
            name="_calibration",
            kind="tool_use_fidelity",
            cases=[
                ToolUseCase(
                    prompt="Search the web for the capital of France.",
                    tool_spec={
                        "name": "search_web",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                    gold_tool_name="search_web",
                ),
                ToolUseCase(
                    prompt="Add 2 and 2.",
                    tool_spec={
                        "name": "calculator",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "number"},
                                "b": {"type": "number"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                    gold_tool_name="calculator",
                ),
            ],
        )

    def run(self, spec: ProbeSpec, ctx: RunContext) -> ProbeResult:
        assert isinstance(spec, ToolUseFidelitySpec)
        if not spec.cases:
            return ProbeResult(
                name=spec.name,
                kind=spec.kind,
                verdict=Verdict.SKIP,
                score=None,
                message="no tool-use cases (inline 'cases' was empty)",
            )

        base_valid: list[bool] = []
        ft_valid: list[bool] = []
        ft_calls: list[dict[str, Any] | None] = []
        base_calls: list[dict[str, Any] | None] = []

        # Greedy decode each case under both views. We open one base
        # context for the whole sweep and one ft context — same shape
        # as preference_flip's per-triple loop, but at the case level
        # so a single backend toggle covers all N cases.
        with ctx.require_backend.as_base() as base_view:
            base_outputs = [
                base_view.generate(c.prompt, max_new_tokens=c.max_new_tokens) for c in spec.cases
            ]
        with ctx.require_backend.as_finetuned() as ft_view:
            ft_outputs = [
                ft_view.generate(c.prompt, max_new_tokens=c.max_new_tokens) for c in spec.cases
            ]

        for case, b_out, f_out in zip(spec.cases, base_outputs, ft_outputs, strict=True):
            b_call = _parse_tool_call(b_out)
            f_call = _parse_tool_call(f_out)
            base_calls.append(b_call)
            ft_calls.append(f_call)
            base_valid.append(b_call is not None and _matches_schema(b_call, case.tool_spec))
            ft_valid.append(f_call is not None and _matches_schema(f_call, case.tool_spec))

        n = len(spec.cases)
        base_valid_rate = sum(base_valid) / n
        ft_valid_rate = sum(ft_valid) / n
        validity_delta = ft_valid_rate - base_valid_rate

        # Argument-disagreement: only computed where BOTH views produced
        # a schema-valid call. Otherwise the comparison is between a
        # call and a non-call, which the probe scores via the validity
        # path instead.
        agreed_pairs = [
            (bc, fc)
            for bc, fc, bv, fv in zip(base_calls, ft_calls, base_valid, ft_valid, strict=True)
            if bv and fv and bc is not None and fc is not None
        ]
        if agreed_pairs:
            disagreement_per_pair = [_field_disagreement(bc, fc) for bc, fc in agreed_pairs]
            mean_arg_disagreement = statistics.fmean(disagreement_per_pair)
        else:
            mean_arg_disagreement = 0.0

        # Hallucination: over schema-valid ft calls, what fraction call
        # the wrong tool? Denominator excludes invalid calls so
        # validity-failure double-counts don't pollute the metric.
        ft_valid_calls = [
            (case, fc)
            for case, fc, fv in zip(spec.cases, ft_calls, ft_valid, strict=True)
            if fv and fc is not None
        ]
        if ft_valid_calls:
            hallucinated = [
                not _name_allowed(fc.get("name", ""), case.gold_tool_name, spec.allowed_tools)
                for case, fc in ft_valid_calls
            ]
            hallucination_rate = sum(hallucinated) / len(ft_valid_calls)
        else:
            hallucination_rate = 0.0

        # Z-score the ft validity rate against the null baseline. CIs
        # are over the per-case validity flags so they reflect sampling
        # noise on the proportion estimate.
        stats = get_null_stats(ctx, spec.kind)
        z = z_score(ft_valid_rate, stats)
        z_by_rank = z_scores_by_rank(ft_valid_rate, get_null_stats_by_rank(ctx, spec.kind), sign=+1)
        ci_95 = bootstrap_ci([1.0 if v else 0.0 for v in ft_valid], seed=ctx.seed)

        # Compound pass criterion: every gate must hold. The z-score
        # path subsumes the validity-delta gate when null calibration
        # ran (a 3σ-significant validity rate is by construction a
        # rate above noise), but we apply both so a probe with z=4σ
        # but a -20pp validity delta still flags as a regression.
        validity_pass = validity_delta >= spec.assert_validity_delta_gte
        hallucination_pass = hallucination_rate <= spec.assert_hallucination_lte
        verdict_z = verdict_from_z(z, spec.assert_z_gte)

        if verdict_z is not None:
            z_path_pass = verdict_z == Verdict.PASS
            verdict = (
                Verdict.PASS
                if (z_path_pass and validity_pass and hallucination_pass)
                else Verdict.FAIL
            )
            base_msg = (
                f"validity ft={ft_valid_rate:.0%} (base={base_valid_rate:.0%}, "
                f"Δ={validity_delta:+.0%}), hallucination={hallucination_rate:.0%}, "
                f"z={z:+.2f}σ vs null"
            )
        else:
            verdict = Verdict.PASS if (validity_pass and hallucination_pass) else Verdict.FAIL
            base_msg = (
                f"validity ft={ft_valid_rate:.0%} (base={base_valid_rate:.0%}, "
                f"Δ={validity_delta:+.0%}), hallucination={hallucination_rate:.0%} "
                f"{no_calibration_note(spec.kind)}"
            )

        # Composite score blends the three signals. Weights tuned so
        # that perfect validity preservation + zero hallucination
        # → 1.0; full hallucination or full validity collapse → 0.0.
        validity_factor = max(0.0, min(1.0, 1.0 + validity_delta))
        hallucination_factor = max(0.0, 1.0 - hallucination_rate)
        score_raw = validity_factor * hallucination_factor
        # Z-score boost only applies when calibration is available
        # — scaling toward score_from_z(z) so a strongly-significant
        # adapter scores higher than a marginally-significant one
        # even at the same raw rates.
        z_score_val = score_from_z(z) if z is not None else None
        score = 0.7 * score_raw + 0.3 * z_score_val if z_score_val is not None else score_raw

        return safe_finalize(
            name=spec.name,
            kind=spec.kind,
            verdict=verdict,
            score=score,
            raw=ft_valid_rate,
            z_score=z,
            base_value=base_valid_rate,
            ft_value=ft_valid_rate,
            evidence={
                "json_valid_rate_base": base_valid_rate,
                "json_valid_rate_ft": ft_valid_rate,
                "validity_delta": validity_delta,
                "mean_arg_disagreement": mean_arg_disagreement,
                "hallucination_rate": hallucination_rate,
                "num_cases": n,
                "num_arg_pairs_compared": len(agreed_pairs),
                "z_by_rank": z_by_rank,
                "raw_ci_95": list(ci_95) if ci_95 is not None else None,
                "weight": spec.weight,
            },
            message=base_msg,
            ci_95=ci_95,
        )


# ---------------------------------------------------------------------------
# Tool-call parsing + minimal schema validator
# ---------------------------------------------------------------------------


def _parse_tool_call(text: str) -> dict[str, Any] | None:
    """Extract ``{"name": ..., "arguments": {...}}`` from a generation.

    Tries three increasingly forgiving strategies:

    1. The whole text parses as JSON and is a dict.
    2. The text contains a ``{...}`` substring that parses as JSON.
    3. The text contains a fenced code block (``\\`\\`\\`json ... \\`\\`\\```)
       whose body parses as JSON.

    Returns ``None`` when no parse succeeds OR when the parsed object
    isn't a dict — both base and ft models often wander prose without
    producing structured calls; that's exactly the failure mode the
    probe scores.

    The returned dict isn't validated yet — callers that care about
    schema conformance run :func:`_matches_schema` separately so the
    "valid JSON, wrong shape" failure mode is observable downstream.
    """
    text = text.strip()
    # Strategy 1.
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        obj = None
    if isinstance(obj, dict):
        return obj

    # Strategy 3 — fenced ```json ... ``` block first because its
    # boundaries are unambiguous; embedded ``{...}`` heuristics in
    # strategy 2 sometimes match a JSON-looking fragment inside a
    # fenced block's body.
    fenced = _extract_fenced_json(text)
    if fenced is not None:
        try:
            obj = json.loads(fenced)
        except (ValueError, TypeError):
            obj = None
        if isinstance(obj, dict):
            return obj

    # Strategy 2 — first balanced ``{...}`` substring.
    candidate = _first_balanced_braces(text)
    if candidate is not None:
        try:
            obj = json.loads(candidate)
        except (ValueError, TypeError):
            obj = None
        if isinstance(obj, dict):
            return obj

    return None


def _extract_fenced_json(text: str) -> str | None:
    """Return the body of a ```json ... ``` block, if any."""
    marker = "```"
    start = text.find(marker)
    if start < 0:
        return None
    # Skip the opening fence (and an optional ``json`` language tag).
    body_start = start + len(marker)
    if text[body_start : body_start + 4].lower() == "json":
        body_start += 4
    # Strip a single trailing newline after the language tag.
    if body_start < len(text) and text[body_start] == "\n":
        body_start += 1
    end = text.find(marker, body_start)
    if end < 0:
        return None
    return text[body_start:end].strip()


def _first_balanced_braces(text: str) -> str | None:
    """Return the first balanced ``{...}`` substring, or ``None``.

    Tracks string-literal context so a brace inside a string doesn't
    throw off the depth counter. Quadratic worst case is fine — the
    inputs are model generations capped at a few hundred tokens.
    """
    depth = 0
    in_string = False
    escape = False
    start = -1
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]
            if depth < 0:
                # Unbalanced — reset and keep scanning.
                depth = 0
                start = -1
    return None


# Type tags we recognize. Matches the OpenAI function-call subset; we
# accept ``int`` for ``"integer"`` and ``int|float`` for ``"number"``.
_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


def _matches_schema(call: dict[str, Any], tool_spec: dict[str, Any]) -> bool:
    """Light OpenAI-flavored JSON-schema check.

    Validates:

    - The call has a ``name`` field (string) — the model is supposed to
      identify the tool it's calling.
    - The call has an ``arguments`` field (dict) — the OpenAI shape.
    - Every name in ``parameters.required`` appears in ``arguments``.
    - Every field declared in ``parameters.properties`` whose value is
      present has a type that matches the property's ``type`` tag.

    Doesn't validate: enums, ``minLength``/``maxLength``, nested
    schemas, ``oneOf``/``anyOf``. Those land in a follow-up sprint
    when users push tool specs that need them — the v1 surface is
    deliberately the OpenAI-tutorial shape.

    Booleans-as-integers caveat: Python's ``isinstance(True, int)``
    is True, so we explicitly reject bools when the schema asks for
    integer/number. Otherwise an adapter that emits ``true`` would
    silently pass an "integer" check.
    """
    if not isinstance(call.get("name"), str):
        return False
    args = call.get("arguments")
    if not isinstance(args, dict):
        return False

    parameters = tool_spec.get("parameters") or {}
    properties = parameters.get("properties") or {}
    required = parameters.get("required") or []

    for r in required:
        if r not in args:
            return False

    for field_name, field_value in args.items():
        prop = properties.get(field_name)
        if not prop:
            # Unknown field — OpenAI's actual function-call rules
            # forbid this, but we permit it because real models often
            # add benign extra fields. Validity stays True.
            continue
        type_tag = prop.get("type")
        if type_tag is None:
            continue
        allowed = _TYPE_CHECKS.get(type_tag)
        if allowed is None:
            continue
        if isinstance(field_value, bool) and bool not in allowed:
            return False
        if not isinstance(field_value, allowed):
            return False
    return True


def _name_allowed(name: str, gold: str, allowed_tools: list[str] | None) -> bool:
    """Return True iff the model's tool ``name`` is acceptable.

    Rules:

    - ``allowed_tools`` declared → ``name`` must be in the list.
    - Otherwise → ``name`` must equal ``gold``.

    The split exists because some users want a per-case strict gold
    (calibrator is checking "did the model pick the *right* tool")
    while others want a surface gate ("did the model stay inside the
    declared tool list").
    """
    if allowed_tools is not None:
        return name in allowed_tools
    return name == gold


def _field_disagreement(base_call: dict[str, Any], ft_call: dict[str, Any]) -> float:
    """Leaf-field disagreement rate between two parsed tool calls.

    Walks the union of leaf paths in ``base_call.arguments`` and
    ``ft_call.arguments``; counts a disagreement for each leaf where
    the values differ (or the path exists on only one side). Rate is
    ``disagreements / total_leaves``; returns ``0.0`` when both
    arguments dicts are empty (a vacuously-perfect agreement).

    Intentionally simple: no semantic equality (``"2.0"`` ≠ ``2.0``
    by design — drift in numeric *type* counts as drift). Probes
    that want softer comparison (numeric tolerance, embedding
    distance on strings) build on this in a follow-up sprint.
    """
    base_args = base_call.get("arguments") or {}
    ft_args = ft_call.get("arguments") or {}
    if not isinstance(base_args, dict) or not isinstance(ft_args, dict):
        # One side wasn't a real arguments dict; treat as full disagreement.
        return 1.0
    base_leaves = dict(_walk_leaves(base_args))
    ft_leaves = dict(_walk_leaves(ft_args))
    all_paths = set(base_leaves) | set(ft_leaves)
    if not all_paths:
        return 0.0
    disagreements = sum(
        1 for p in all_paths if base_leaves.get(p, _MISSING) != ft_leaves.get(p, _MISSING)
    )
    return disagreements / len(all_paths)


_MISSING = object()
"""Sentinel for "field not present" in :func:`_field_disagreement`. A
leaf with value ``None`` should NOT compare equal to a leaf that's
absent — using a private singleton makes the distinction explicit
where ``dict.get(k)`` would conflate the two."""


def _walk_leaves(obj: Any, path: tuple[str, ...] = ()) -> Any:
    """Yield ``(path, leaf_value)`` pairs for nested dicts.

    Lists are leaves themselves (not recursed into) — list-element
    drift would otherwise dominate the disagreement metric in cases
    where the model legitimately returns the same set in a different
    order, which we don't want to flag as fidelity loss.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_leaves(v, path + (str(k),))
    else:
        yield path, obj
