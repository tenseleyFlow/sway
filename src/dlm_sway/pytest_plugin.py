"""pytest plugin — ``@pytest.mark.sway`` expands into per-probe items (S15 / F10).

Install with ``pip install 'dlm-sway[pytest]'`` and the plugin
auto-loads via the ``pytest11`` entry point. Writing::

    @pytest.mark.sway(spec="sway.yaml", threshold=0.6)
    def test_adapter_healthy(): ...

turns a single test function into **N + 1** pytest items:

- one item per probe in ``sway.yaml``, named
  ``test_adapter_healthy::<probe_name>``, outcome tied to the probe's
  verdict (``FAIL``/``ERROR`` → pytest Failed; ``SKIP`` → pytest
  Skipped; everything else passes),
- a single ``test_adapter_healthy::__gate__`` item that fails when
  the composite score falls below ``threshold`` (only added when the
  caller passes a positive ``threshold``).

The suite runs **once per decorated function**; subsequent synthetic
items read from a per-session cache so the N-way expansion doesn't
multiply backend wall time.

The body of the decorated function is intentionally ignored — the
decorator owns the test. A ``pass`` (or any non-raising body) is
conventional. This mirrors how ``@hypothesis.given(...)`` replaces
the function's behavior while pytest still discovers it through the
normal ``test_*`` name convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from dlm_sway.core.errors import SwayError
from dlm_sway.core.result import ProbeResult, SuiteResult, SwayScore, Verdict

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.nodes import Item


# ----------------------------------------------------------------------
# Session-scoped suite cache
# ----------------------------------------------------------------------


_CACHE_ATTR: str = "_sway_suite_cache"


class _SuiteCache:
    """Per-session cache of ``SuiteResult`` / ``SwayScore`` pairs.

    Keyed by ``(spec_path, sorted_weights_tuple)``. Two decorated
    functions pointing at the same spec with the same weights share
    one backend load + one suite run — the cache is the whole point
    of the "one item per probe" expansion being cheap.
    """

    def __init__(self) -> None:
        self._cache: dict[
            tuple[str, tuple[tuple[str, float], ...]], tuple[SuiteResult, SwayScore]
        ] = {}

    def get_or_run(
        self,
        spec_path: Path,
        *,
        weights: dict[str, float] | None,
    ) -> tuple[SuiteResult, SwayScore]:
        key_weights = tuple(sorted((weights or {}).items()))
        key = (str(spec_path.resolve()), key_weights)
        if key not in self._cache:
            # Deferred import: the CLI module pulls everything else;
            # lighter plugin entry if we only import when firing.
            from dlm_sway.cli.commands import _execute_spec

            self._cache[key] = _execute_spec(spec_path, weights_override=weights)
        return self._cache[key]


# ----------------------------------------------------------------------
# pytest hooks
# ----------------------------------------------------------------------


def pytest_configure(config: Config) -> None:
    """Register the ``sway`` marker and install the per-session cache."""
    config.addinivalue_line(
        "markers",
        "sway(spec, threshold=0.0, weights=None): "
        "expand a pytest function into one item per sway probe in the "
        "referenced spec. Required kwarg ``spec`` is a path-like to a "
        "sway.yaml. Optional ``threshold`` adds a ``__gate__`` item "
        "that fails when the composite score drops below that value. "
        "Optional ``weights`` overrides the composite-score category "
        "weights (same schema as ``sway run --weights``).",
    )
    # One cache per session. Using ``setattr`` so plugin uninstall
    # simply drops the attribute; no singleton-cleanup dance.
    setattr(config, _CACHE_ATTR, _SuiteCache())


def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    """Replace each ``@pytest.mark.sway``-decorated item with per-probe items.

    Runs after standard collection — at this point ``items`` is the
    list pytest is about to execute. We scan for items carrying the
    ``sway`` marker and substitute them in place.
    """
    cache: _SuiteCache = getattr(config, _CACHE_ATTR)
    new_items: list[Item] = []
    for item in items:
        mark = item.get_closest_marker("sway") if hasattr(item, "get_closest_marker") else None
        if mark is None:
            new_items.append(item)
            continue
        # Only operate on pytest Function items (vs. Class / Module).
        # Anything else — we leave alone with a warning surfaced via
        # the usual pytest WARN mechanism.
        if not _is_function_item(item) or item.parent is None:
            new_items.append(item)
            continue
        try:
            spec_path, threshold, weights = _parse_mark(mark)
        except _SwayMarkError as exc:
            # Surface the configuration error as a single failed item
            # so the user sees a green-field message in pytest's
            # output instead of a cryptic collect error.
            new_items.append(
                _ConfigErrorItem.from_parent(
                    parent=item.parent,
                    name=item.name,
                    message=str(exc),
                )
            )
            continue
        expanded = _expand_to_probe_items(
            parent_item=item,
            spec_path=spec_path,
            threshold=threshold,
            weights=weights,
            cache=cache,
        )
        new_items.extend(expanded)
    items[:] = new_items


# ----------------------------------------------------------------------
# Mark parsing + item expansion
# ----------------------------------------------------------------------


class _SwayMarkError(Exception):
    """Raised during mark parsing when the user's arguments are bad."""


def _parse_mark(
    mark: pytest.Mark,
) -> tuple[Path, float, dict[str, float] | None]:
    """Pull ``(spec_path, threshold, weights)`` out of a ``@pytest.mark.sway(...)``."""
    # ``mark.args`` + ``mark.kwargs`` together give the call shape.
    # We accept either ``@pytest.mark.sway("path.yaml")`` or
    # ``@pytest.mark.sway(spec="path.yaml")``.
    kwargs = dict(mark.kwargs)
    args = list(mark.args)

    spec = kwargs.pop("spec", None)
    if spec is None and args:
        spec = args.pop(0)
    if spec is None:
        raise _SwayMarkError("@pytest.mark.sway requires a `spec` kwarg or a positional spec path")
    spec_path = Path(spec)
    if not spec_path.is_absolute():
        # Resolve against the config rootpath — the project root pytest
        # discovers, the same one users edit from. Means the spec can
        # sit next to the test file without a full absolute path.
        spec_path = spec_path.resolve()

    threshold_raw = kwargs.pop("threshold", 0.0)
    try:
        threshold = float(threshold_raw)
    except (TypeError, ValueError) as exc:
        raise _SwayMarkError(
            f"@pytest.mark.sway `threshold` must be a float; got {threshold_raw!r}"
        ) from exc

    weights = kwargs.pop("weights", None)
    if weights is not None:
        if not isinstance(weights, dict):
            raise _SwayMarkError(
                f"@pytest.mark.sway `weights` must be a dict or None; got {type(weights).__name__}"
            )
        try:
            weights = {str(k): float(v) for k, v in weights.items()}
        except (TypeError, ValueError) as exc:
            raise _SwayMarkError(f"@pytest.mark.sway `weights` must map str→float ({exc})") from exc

    if args or kwargs:
        # Unknown args — pytest marks silently drop them otherwise.
        extra = list(args) + sorted(kwargs)
        raise _SwayMarkError(f"@pytest.mark.sway got unexpected arguments: {extra}")

    return spec_path, threshold, weights


def _expand_to_probe_items(
    *,
    parent_item: Item,
    spec_path: Path,
    threshold: float,
    weights: dict[str, float] | None,
    cache: _SuiteCache,
) -> list[Item]:
    """Build one ``_SwayProbeItem`` per probe + an optional gate item.

    We don't run the suite here — suite execution is deferred to the
    first item's ``runtest``. Collection stays fast; failures don't
    appear until `pytest` actually runs the test.
    """
    from dlm_sway.suite.loader import load_spec

    parent = parent_item.parent
    assert parent is not None  # narrowing: caller filtered None above

    try:
        spec = load_spec(spec_path)
    except SwayError as exc:
        return [
            _ConfigErrorItem.from_parent(
                parent=parent,
                name=parent_item.name,
                message=f"failed to load spec {spec_path}: {exc}",
            )
        ]

    base_name = parent_item.name
    out: list[Item] = []
    for probe_entry in spec.suite:
        probe_name = str(probe_entry.get("name", probe_entry.get("kind", "?")))
        out.append(
            _SwayProbeItem.from_parent(
                parent=parent,
                name=f"{base_name}::{probe_name}",
                spec_path=spec_path,
                weights=weights,
                probe_name=probe_name,
                cache=cache,
            )
        )
    if threshold > 0.0:
        out.append(
            _SwayGateItem.from_parent(
                parent=parent,
                name=f"{base_name}::__gate__",
                spec_path=spec_path,
                weights=weights,
                threshold=threshold,
                cache=cache,
            )
        )
    return out


# ----------------------------------------------------------------------
# Item classes
# ----------------------------------------------------------------------


def _is_function_item(item: Item) -> bool:
    """Duck-check for ``pytest.Function`` without importing its private
    module at top level (cheaper plugin init)."""
    return item.__class__.__name__ == "Function"


class _SwayProbeItem(pytest.Item):
    """One pytest item per sway probe.

    When pytest runs it, we ask the session cache for the suite's
    result (running it on first demand), find the matching probe by
    name, and translate its verdict to a pytest outcome.
    """

    def __init__(
        self,
        *,
        name: str,
        parent: Any,
        spec_path: Path,
        weights: dict[str, float] | None,
        probe_name: str,
        cache: _SuiteCache,
    ) -> None:
        super().__init__(name, parent)
        self._spec_path = spec_path
        self._weights = weights
        self._probe_name = probe_name
        self._cache = cache

    def runtest(self) -> None:  # noqa: D401
        suite_result, _score = self._cache.get_or_run(self._spec_path, weights=self._weights)
        probe = _find_probe(suite_result, self._probe_name)
        if probe is None:
            pytest.fail(
                f"probe {self._probe_name!r} not in suite result — "
                f"available: {sorted(p.name for p in suite_result.probes)}"
            )
        _apply_verdict(probe)

    def repr_failure(self, excinfo: Any, style: str | None = None) -> str:
        del style  # we always render "short"; pytest's style kwarg ignored
        return str(excinfo.getrepr(style="short"))

    def reportinfo(self) -> tuple[Any, int | None, str]:
        return self.fspath, 0, self.name


class _SwayGateItem(pytest.Item):
    """``__gate__`` item — fails when the composite score drops below ``threshold``."""

    def __init__(
        self,
        *,
        name: str,
        parent: Any,
        spec_path: Path,
        weights: dict[str, float] | None,
        threshold: float,
        cache: _SuiteCache,
    ) -> None:
        super().__init__(name, parent)
        self._spec_path = spec_path
        self._weights = weights
        self._threshold = threshold
        self._cache = cache

    def runtest(self) -> None:
        _suite, score = self._cache.get_or_run(self._spec_path, weights=self._weights)
        if score.overall < self._threshold:
            pytest.fail(
                f"composite score {score.overall:.2f} below threshold {self._threshold:.2f} "
                f"(band: {score.band or '—'})",
                pytrace=False,
            )

    def repr_failure(self, excinfo: Any, style: str | None = None) -> str:
        del style  # we always render "short"; pytest's style kwarg ignored
        return str(excinfo.getrepr(style="short"))

    def reportinfo(self) -> tuple[Any, int | None, str]:
        return self.fspath, 0, self.name


class _ConfigErrorItem(pytest.Item):
    """Synthetic item that simply fails with a configuration-error message.

    Used when ``@pytest.mark.sway`` itself is malformed — we want the
    user to see a clean pytest failure line, not a cryptic collection
    error, and we want ``pytest -k`` etc. to still find the test.
    """

    def __init__(self, *, name: str, parent: Any, message: str) -> None:
        super().__init__(name, parent)
        self._message = message

    def runtest(self) -> None:
        pytest.fail(self._message, pytrace=False)

    def repr_failure(self, excinfo: Any, style: str | None = None) -> str:
        del excinfo, style  # config-error path surfaces only the canned message
        return self._message

    def reportinfo(self) -> tuple[Any, int | None, str]:
        return self.fspath, 0, self.name


# ----------------------------------------------------------------------
# Verdict → pytest outcome translation
# ----------------------------------------------------------------------


def _find_probe(suite: SuiteResult, name: str) -> ProbeResult | None:
    for p in suite.probes:
        if p.name == name:
            return p
    return None


def _apply_verdict(probe: ProbeResult) -> None:
    """Translate a probe's :class:`Verdict` to a pytest outcome."""
    msg = probe.message or ""
    if probe.verdict == Verdict.PASS:
        return
    if probe.verdict == Verdict.WARN:
        # Surface the warning through pytest's own warning channel
        # so ``pytest -W`` flags play along. Test still passes.
        import warnings

        warnings.warn(f"sway WARN [{probe.kind}]: {msg}", stacklevel=2)
        return
    if probe.verdict == Verdict.SKIP:
        pytest.skip(msg or f"probe {probe.name!r} skipped")
    if probe.verdict == Verdict.FAIL:
        pytest.fail(f"FAIL [{probe.kind}]: {msg}", pytrace=False)
    if probe.verdict == Verdict.ERROR:
        pytest.fail(f"ERROR [{probe.kind}]: {msg}", pytrace=False)


__all__ = ["pytest_collection_modifyitems", "pytest_configure"]
