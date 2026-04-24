"""Integration test: dlm public surface compatibility (F06).

Asserts the contract sway's resolver depends on against a real
installed ``dlm`` — every entry in dlm's built-in base-model registry
must ``resolve()`` to an object carrying an ``hf_id`` attribute. Catches
the ``.hf_id`` → ``.repo_id`` rename (or any equivalent) before a user
hits it via ``sway autogen``.

Runs under ``slow + online + dlm-extra``. Skipped cleanly when the
``dlm`` extra isn't installed (so `pytest tests/integration` on a
no-extras runner still works).
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.online]


dlm_base_models = pytest.importorskip(
    "dlm.base_models",
    reason="dlm extra not installed (pip install 'dlm-sway[dlm]')",
)


def _iter_registry_keys() -> list[str]:
    """Return every registry key dlm exposes.

    dlm's API surface for enumerating the registry has moved around
    (``list_bases``, ``registry.keys()``, ``iter_registered``). Try
    the known shapes; if none work, yield the canonical keys hard-coded
    in the sway CLAUDE.md so the test still runs against a stable
    floor. Failure mode: zero keys → test is asserting nothing → fail
    loudly in ``test_registry_nonempty``.
    """
    for accessor in ("list_bases", "registered_bases", "list_registered"):
        fn = getattr(dlm_base_models, accessor, None)
        if callable(fn):
            return list(fn())
    registry = getattr(dlm_base_models, "REGISTRY", None) or getattr(
        dlm_base_models, "registry", None
    )
    if registry is not None and hasattr(registry, "keys"):
        return list(registry.keys())
    # Floor: keys the sway CLAUDE.md documents as shipped by dlm.
    return [
        "qwen2.5-0.5b",
        "qwen2.5-1.5b",
        "qwen2.5-3b",
        "qwen2.5-coder-1.5b",
        "llama-3.2-1b",
        "llama-3.2-3b",
        "smollm2-135m",
        "smollm2-360m",
        "smollm2-1.7b",
        "phi-3.5-mini",
    ]


def test_registry_nonempty() -> None:
    """At least one registry key exists — tests below are trivial
    otherwise."""
    keys = _iter_registry_keys()
    assert keys, "dlm base-model registry appears empty; accessor changed?"


@pytest.mark.parametrize("key", _iter_registry_keys())
def test_base_resolves_with_hf_id(key: str) -> None:
    """Every registry entry must resolve to an object with ``hf_id``.

    sway's resolver (``integrations/dlm/resolver.py``) calls
    ``dlm.base_models.resolve(key).hf_id``. A rename of that attribute
    would break every autogen run; this test catches the drift at
    sway's CI.
    """
    spec = dlm_base_models.resolve(key)
    assert hasattr(spec, "hf_id"), (
        f"dlm.base_models.resolve({key!r}) returned {type(spec).__name__} "
        f"without .hf_id — public-surface drift. Visible attrs: "
        f"{sorted(a for a in dir(spec) if not a.startswith('_'))[:8]!r}"
    )
    assert isinstance(spec.hf_id, str) and "/" in spec.hf_id, (
        f"dlm.base_models.resolve({key!r}).hf_id = {spec.hf_id!r} — "
        "not a plausible HuggingFace 'org/name' id."
    )


def test_sway_resolver_does_not_wrap_valid_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """A well-formed dlm spec passes through sway's resolver without
    raising — guards against an accidentally over-strict hasattr check.
    """
    from dlm_sway.integrations.dlm.resolver import _resolve_base_model_to_hf_id

    # Pick any key; use the smallest to keep the test fast.
    keys = _iter_registry_keys()
    # Prefer SmolLM 135M — it's the smallest known base and is the
    # integration fixture sway already uses.
    key = next((k for k in keys if "135m" in k), keys[0])

    hf_id = _resolve_base_model_to_hf_id(key)
    assert "/" in hf_id, f"{key!r} → {hf_id!r} (expected 'org/name')"
