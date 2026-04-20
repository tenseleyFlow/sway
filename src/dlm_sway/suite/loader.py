"""Load + validate a ``sway.yaml`` into a :class:`SwaySpec`.

Separated from :mod:`spec` so the data models stay trivially
importable (no YAML dependency at import time for callers that
construct specs programmatically).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from dlm_sway.core.errors import SpecValidationError
from dlm_sway.suite.spec import SwaySpec


def load_spec(path: Path | str) -> SwaySpec:
    """Parse ``path`` and return a validated :class:`SwaySpec`."""
    resolved = Path(path).expanduser().resolve()
    try:
        raw_text = resolved.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SpecValidationError(f"spec file not found: {resolved}", source=str(path)) from exc

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise SpecValidationError(f"invalid YAML: {exc}", source=str(path)) from exc

    if not isinstance(data, dict):
        raise SpecValidationError("top-level document must be a mapping", source=str(path))
    return from_dict(data, source=str(path))


def from_dict(data: dict[str, Any], *, source: str | None = None) -> SwaySpec:
    """Validate a dict (already parsed from YAML or JSON) as a SwaySpec."""
    try:
        spec = SwaySpec.model_validate(data)
    except ValidationError as exc:
        raise SpecValidationError(str(exc), source=source) from exc
    try:
        spec.check_version()
    except ValueError as exc:
        raise SpecValidationError(str(exc), source=source) from exc
    return spec
