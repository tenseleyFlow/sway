"""Exception hierarchy for dlm-sway.

Every error sway raises inherits from :class:`SwayError` so callers can
catch the whole family with a single ``except``. Subclasses carry enough
context (spec paths, probe names, missing extras) for the CLI to render
actionable messages without the caller having to introspect an exception
chain.
"""

from __future__ import annotations


class SwayError(Exception):
    """Root of the dlm-sway exception hierarchy."""


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
