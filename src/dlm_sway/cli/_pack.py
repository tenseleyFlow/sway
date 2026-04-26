"""Spec + artifacts → portable swaypack tarball (Sprint 26 / X3).

A ``swaypack`` is a single ``.tar.gz`` containing everything needed to
reproduce a sway run **without** hitting the user's home cache or the
network: the spec YAML, a copy of the source ``.dlm`` document (when
the spec resolves one), the null-stats cache entries the spec's probes
will look up, and an optional last-known-good golden JSON report.

Layout inside the tarball::

    swaypack/
      manifest.json          # version + included artifacts + pack-time pinned versions
      sway.yaml              # the spec the user ran ``sway pack`` on (verbatim)
      source.dlm             # copied from spec.dlm_source if present (X3 — optional)
      null-stats/
        <key>.json           # one per cache key the spec's probes might query
      golden.json            # last-known-good ``sway run`` report (optional)

Compression: stdlib ``tar.gz``. The sprint planning leaned toward
``zstd`` for compactness but the dep cost (no zstandard in core) isn't
worth it for the typical 1–5 MB pack. Future: swap to zstd behind a
``--compression`` flag.
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dlm_sway import __version__
from dlm_sway.core.errors import SwayError
from dlm_sway.suite.loader import load_spec

if TYPE_CHECKING:
    from dlm_sway.suite.spec import SwaySpec


logger = logging.getLogger(__name__)


SWAYPACK_VERSION = 1
"""Bump this when the on-disk pack layout changes incompatibly."""

DEFAULT_MAX_PACK_SIZE_BYTES = 50 * 1024 * 1024
"""50 MB cap. ``sway pack`` warns when the result exceeds this; use
``--max-size`` to override."""


class PackError(SwayError):
    """Raised when ``sway pack`` can't build a usable tarball."""


@dataclass(frozen=True, slots=True)
class PackReport:
    """Result of a successful pack call.

    Attributes
    ----------
    out_path:
        Where the tarball was written.
    size_bytes:
        Final tarball size.
    spec_path:
        Source spec path (resolved to absolute).
    section_bytes:
        Bytes of source ``.dlm`` content packed (0 when no
        ``dlm_source`` resolved).
    null_stats_count:
        Number of null-stats JSON entries packed.
    golden_included:
        Whether a golden report was bundled.
    """

    out_path: Path
    size_bytes: int
    spec_path: Path
    section_bytes: int
    null_stats_count: int
    golden_included: bool


def pack_spec(
    spec_path: Path,
    *,
    out_path: Path,
    include_golden: Path | None = None,
    include_null_cache: bool = True,
    max_size_bytes: int = DEFAULT_MAX_PACK_SIZE_BYTES,
) -> PackReport:
    """Build a swaypack tarball at ``out_path``.

    Parameters
    ----------
    spec_path:
        Path to the ``sway.yaml`` to pack.
    out_path:
        Destination tarball (typically ``<name>.swaypack.tar.gz``).
        Refuses to overwrite an existing file — caller must ``rm``
        first to avoid silent clobber of a previous pack.
    include_golden:
        Optional path to a JSON ``sway run`` report to embed as
        ``swaypack/golden.json`` for reproducibility comparison.
    include_null_cache:
        When True (default), copy any null-stats JSON files the
        spec's probes might query from
        ``$XDG_CACHE_HOME/dlm-sway/null-stats`` into the pack.
    max_size_bytes:
        Soft cap. The function builds the tarball regardless but
        raises ``PackError`` *before* writing if it'll exceed this
        cap. ``--max-size`` overrides on the CLI.

    Returns
    -------
    PackReport
        What was packed + final size.

    Raises
    ------
    PackError
        Spec invalid, output already exists, or final size exceeds
        ``max_size_bytes``.
    """
    spec_path = Path(spec_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()

    if out_path.exists():
        raise PackError(f"refusing to overwrite existing pack at {out_path} — delete it first")

    spec = load_spec(spec_path)

    # Build the tarball in-memory first so we can size-cap before
    # writing (the alternative — write then check + delete — risks
    # a half-written file on disk if the cap fails).
    buf = io.BytesIO()
    section_bytes = 0
    null_stats_count = 0
    null_stats_keys: list[str] = []
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # 1) Spec verbatim.
        _add_file_bytes(
            tar,
            "swaypack/sway.yaml",
            spec_path.read_bytes(),
            mtime=time.time(),
        )

        # 2) Source .dlm if the spec carries one.
        if spec.dlm_source:
            section_bytes = _add_dlm_source(tar, spec, spec_path)

        # 3) Null-stats cache (optional).
        if include_null_cache:
            null_stats_count, null_stats_keys = _add_null_cache(tar, spec)

        # 4) Golden report (optional).
        golden_included = False
        if include_golden is not None:
            golden_included = _add_golden(tar, include_golden)

        # 5) Manifest last so it sees the truth about what we packed.
        manifest = {
            "swaypack_version": SWAYPACK_VERSION,
            "sway_version": __version__,
            "packed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "spec_filename": "sway.yaml",
            "section_bytes": section_bytes,
            "null_stats_count": null_stats_count,
            "null_stats_keys": null_stats_keys,
            "golden_included": golden_included,
            "dlm_source_packed": spec.dlm_source is not None and section_bytes > 0,
        }
        _add_file_bytes(
            tar,
            "swaypack/manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n",
            mtime=time.time(),
        )

    size = buf.tell()
    if size > max_size_bytes:
        mb = size / (1024 * 1024)
        cap_mb = max_size_bytes / (1024 * 1024)
        raise PackError(
            f"pack would be {mb:.1f} MB which exceeds the cap of {cap_mb:.1f} MB. "
            f"Pass --max-size <bytes> to override, or drop --include-null-cache."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(buf.getvalue())

    return PackReport(
        out_path=out_path,
        size_bytes=size,
        spec_path=spec_path,
        section_bytes=section_bytes,
        null_stats_count=null_stats_count,
        golden_included=golden_included,
    )


def _add_file_bytes(tar: tarfile.TarFile, arcname: str, data: bytes, *, mtime: float) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    info.mtime = int(mtime)
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def _add_dlm_source(tar: tarfile.TarFile, spec: SwaySpec, spec_path: Path) -> int:
    """Copy the spec's ``dlm_source`` into the pack.

    Resolution: if dlm_source is absolute, use it. If relative, resolve
    against the spec file's directory (matches the runner's autogen
    convention).
    """
    if spec.dlm_source is None:
        return 0
    src = Path(spec.dlm_source).expanduser()
    if not src.is_absolute():
        src = (spec_path.parent / src).resolve()
    if not src.exists():
        logger.warning(
            "dlm_source=%s doesn't exist on disk; pack will be missing source.dlm",
            spec.dlm_source,
        )
        return 0
    data = src.read_bytes()
    _add_file_bytes(tar, "swaypack/source.dlm", data, mtime=src.stat().st_mtime)
    return len(data)


def _add_null_cache(tar: tarfile.TarFile, spec: SwaySpec) -> tuple[int, list[str]]:
    """Copy null-stats JSON entries that the spec's probes might query.

    Conservative scope: copy *every* JSON in
    ``$XDG_CACHE_HOME/dlm-sway/null-stats/`` (or the legacy home cache).
    Per-probe key matching is over-engineering — most users have
    O(10) cached entries, so the pack's max-size cap catches blowups.
    """
    del spec  # We pack the whole cache; per-spec filtering deferred.

    # Same root resolution as _null_cache._cache_root, but without
    # honoring SWAY_NULL_CACHE_DIR (we pack from the user's HOME
    # cache, not from another pack — that would be a no-op cycle).
    import os

    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        cache_root = Path(xdg).expanduser() / "dlm-sway" / "null-stats"
    else:
        cache_root = Path.home() / ".dlm-sway" / "null-stats"

    if not cache_root.exists() or not cache_root.is_dir():
        return 0, []

    count = 0
    keys: list[str] = []
    for entry in sorted(cache_root.iterdir()):
        if not entry.is_file() or entry.suffix != ".json":
            continue
        data = entry.read_bytes()
        _add_file_bytes(
            tar,
            f"swaypack/null-stats/{entry.name}",
            data,
            mtime=entry.stat().st_mtime,
        )
        count += 1
        keys.append(entry.stem)
    return count, keys


def _add_golden(tar: tarfile.TarFile, golden_path: Path) -> bool:
    """Bundle a known-good ``sway run`` report for verification."""
    golden_path = Path(golden_path).expanduser()
    if not golden_path.exists():
        raise PackError(f"--include-golden path does not exist: {golden_path}")
    if not golden_path.is_file():
        raise PackError(f"--include-golden must be a file, got: {golden_path}")
    data = golden_path.read_bytes()
    _add_file_bytes(
        tar,
        "swaypack/golden.json",
        data,
        mtime=golden_path.stat().st_mtime,
    )
    return True
