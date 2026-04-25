"""Exception hierarchy for sway.

Every error sway raises inherits from :class:`SwayError` so callers can
catch the whole family with a single ``except``. Subclasses carry enough
context (spec paths, probe names, missing extras) for the CLI to render
actionable messages without the caller having to introspect an exception
chain.
"""

from __future__ import annotations


class SwayError(Exception):
    """Root of the sway exception hierarchy."""


class SpecValidationError(SwayError):
    """A ``sway.yaml`` (or equivalent) failed pydantic validation.

    Parameters
    ----------
    message:
        Human-readable summary of what went wrong.
    source:
        Path or identifier of the spec being validated, if known.
    """

    def __init__(self, message: str, *, source: str | None = None) -> None:
        super().__init__(message)
        self.source = source

    def __str__(self) -> str:
        base = super().__str__()
        return f"{self.source}: {base}" if self.source else base


class BackendNotAvailableError(SwayError):
    """A requested backend's optional dependencies aren't installed.

    The CLI turns this into a pointed ``pip install dlm-sway[<extra>]``
    hint; programmatic callers can read :attr:`extra` directly.
    """

    def __init__(self, backend: str, *, extra: str, hint: str | None = None) -> None:
        message = (
            f"backend {backend!r} unavailable — install the extra: pip install 'dlm-sway[{extra}]'"
        )
        if hint:
            message = f"{message}\n{hint}"
        super().__init__(message)
        self.backend = backend
        self.extra = extra


class ProbeError(SwayError):
    """A probe failed to *execute* (as opposed to failing its assertion).

    Distinct from a ``verdict=FAIL`` result — assertion failures are
    normal and reported via :class:`ProbeResult`. This is for genuine
    bugs: missing sections, mismatched tokenizers, NaN logits.
    """

    def __init__(self, probe: str, message: str) -> None:
        super().__init__(f"probe {probe!r}: {message}")
        self.probe = probe


class MissingTrainingStateError(SwayError):
    """The pre-run probes (S25 ``gradient_ghost``) couldn't find a
    ``training_state.pt`` next to the adapter.

    Distinguishes "the file legitimately doesn't exist for this adapter"
    (probe SKIPs cleanly) from "the file exists but won't load"
    (probe ERRORs). Pre-run probes catch this and emit SKIP rather
    than letting the missing file kill the suite.
    """

    def __init__(self, adapter_path: object) -> None:
        super().__init__(
            f"no training_state.pt under {adapter_path} — adapter wasn't "
            f"produced by dlm or the file was pruned. Pre-run diagnostics "
            f"(gradient_ghost) will SKIP for this adapter."
        )
        self.adapter_path = adapter_path


class DlmCompatError(SwayError):
    """The installed ``dlm`` package's public surface doesn't match what
    sway's resolver expects.

    Raised when e.g. ``dlm.base_models.resolve`` returns an object
    without the ``hf_id`` attribute we depend on — historically dlm
    used ``hf_id``; if it renames to ``repo_id`` we want a loud,
    actionable error with both version strings in the message, not a
    silent pass-through that hands the backend a registry key it can't
    load.

    The installed-dlm version is introspected best-effort; it's
    informational, not a key for programmatic branching.
    """

    def __init__(self, message: str, *, installed_dlm_version: str | None = None) -> None:
        full = message
        if installed_dlm_version:
            full = f"{message} (installed dlm version: {installed_dlm_version})"
        full = (
            f"{full}\n"
            "Hint: pin a compatible dlm with: pip install 'dlm-sway[dlm]' "
            "(resolves the tested dlm version range from pyproject.toml)."
        )
        super().__init__(full)
        self.installed_dlm_version = installed_dlm_version
