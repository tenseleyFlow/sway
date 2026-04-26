"""swaypack tarball → ready-to-run directory tree (Sprint 26 / X3).

Inverse of :mod:`dlm_sway.cli._pack`. Given a ``*.swaypack.tar.gz``,
extract its contents into a target dir, validate the manifest, and
return enough info for ``sway unpack`` to print actionable next-step
instructions ("``cd <dir> && SWAY_NULL_CACHE_DIR=<dir>/null-stats sway run sway.yaml``").

Path-traversal hardening: ``tarfile.extractall`` defaults to ``filter='data'``
on Python 3.12+, which already rejects absolute paths and ``../``
escape attempts. We pin that filter explicitly so the safety holds
on 3.11 (which we support per ``pyproject.toml``) and isn't a
rolling-mean-of-warnings trap on future versions.
"""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

from dlm_sway.core.errors import SwayError


class UnpackError(SwayError):
    """Raised when ``sway unpack`` can't extract a usable swaypack."""


@dataclass(frozen=True, slots=True)
class UnpackReport:
    """Result of a successful unpack call.

    Attributes
    ----------
    out_dir:
        Where the contents were extracted (``<target>/swaypack/``).
    spec_path:
        Convenience pointer to ``out_dir / "sway.yaml"``.
    null_stats_dir:
        Path to the unpacked null-stats cache, or ``None`` when the
        pack didn't bundle one. Callers set ``SWAY_NULL_CACHE_DIR``
        to this value before running ``sway run`` to honor the
        bundled stats.
    manifest:
        Parsed ``manifest.json`` — useful for diagnostics.
    """

    out_dir: Path
    spec_path: Path
    null_stats_dir: Path | None
    manifest: dict[str, object]


def unpack_swaypack(pack_path: Path, *, target_dir: Path) -> UnpackReport:
    """Extract a swaypack into ``target_dir``.

    Parameters
    ----------
    pack_path:
        Path to a ``.swaypack.tar.gz`` produced by :func:`pack_spec`.
    target_dir:
        Parent directory to extract into. The pack's contents land
        at ``target_dir / "swaypack/"`` (the tarball's top-level
        directory). Must not already contain a ``swaypack/`` dir.

    Returns
    -------
    UnpackReport
        Pointers callers need to wire into a ``sway run`` invocation.

    Raises
    ------
    UnpackError
        Pack file missing / unreadable, target dir already has a
        ``swaypack/`` subdir, manifest missing or version-incompatible.
    """
    pack_path = Path(pack_path).expanduser().resolve()
    target_dir = Path(target_dir).expanduser().resolve()

    if not pack_path.exists():
        raise UnpackError(f"swaypack not found: {pack_path}")
    if not pack_path.is_file():
        raise UnpackError(f"swaypack must be a file, got: {pack_path}")

    out_root = target_dir / "swaypack"
    if out_root.exists():
        raise UnpackError(
            f"refusing to overwrite existing {out_root} — delete it or "
            f"pass --out to a fresh directory"
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(str(pack_path), mode="r:gz") as tar:
            # ``filter='data'`` rejects absolute paths, ``../`` escapes,
            # and special-device entries. Required for safe extraction
            # of untrusted-source archives — packs may travel via
            # email / shared drives.
            tar.extractall(path=str(target_dir), filter="data")
    except (tarfile.ReadError, tarfile.ExtractError, OSError) as exc:
        raise UnpackError(f"failed to extract {pack_path}: {type(exc).__name__}: {exc}") from exc

    if not out_root.exists():
        raise UnpackError(
            f"swaypack extracted but missing the expected 'swaypack/' "
            f"directory; either {pack_path} isn't a sway pack or it "
            f"was built by an incompatible writer"
        )

    manifest_path = out_root / "manifest.json"
    if not manifest_path.exists():
        raise UnpackError(f"swaypack missing manifest.json (looked at {manifest_path})")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise UnpackError(f"manifest.json is not valid JSON: {exc}") from exc

    swaypack_version = manifest.get("swaypack_version")
    if swaypack_version != 1:
        raise UnpackError(
            f"unsupported swaypack_version={swaypack_version}. This sway "
            f"build understands swaypack_version=1 (S26)."
        )

    spec_path = out_root / manifest.get("spec_filename", "sway.yaml")
    if not spec_path.exists():
        raise UnpackError(
            f"manifest claims spec at {spec_path.name} but it's missing from the pack"
        )

    null_stats_dir: Path | None = out_root / "null-stats"
    if not null_stats_dir.exists() or not null_stats_dir.is_dir():
        null_stats_dir = None

    return UnpackReport(
        out_dir=out_root,
        spec_path=spec_path,
        null_stats_dir=null_stats_dir,
        manifest=manifest,
    )
