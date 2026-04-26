"""Unit tests for ``sway pack`` / ``sway unpack`` (Sprint 26 / X3).

Round-trips a synthetic spec through pack → unpack → re-load and
verifies every artifact comes back byte-identical (modulo tarball
metadata). Also pins the structural error paths: missing dlm_source,
size cap, refusing to overwrite, malformed manifest, version
mismatch, path-traversal rejection.

End-to-end "sway run on the unpacked spec matches the pre-pack
verdict" lives in ``tests/integration/test_pack_run_roundtrip.py``.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from dlm_sway.cli._pack import (
    DEFAULT_MAX_PACK_SIZE_BYTES,
    SWAYPACK_VERSION,
    PackError,
    pack_spec,
)
from dlm_sway.cli._unpack import UnpackError, unpack_swaypack


def _write_minimal_spec(spec_path: Path, *, dlm_source: str | None = None) -> None:
    """Write a tiny syntactically-valid sway.yaml. The dlm_source
    field is optional — set it to point at a sibling ``.dlm`` to
    test source-bundling."""
    body = """\
version: 1
models:
  base: {kind: dummy, base: dummy-base}
  ft:   {kind: dummy, base: dummy-base}
defaults:
  seed: 0
suite:
  - {name: dk, kind: delta_kl, prompts: [hello]}
"""
    if dlm_source:
        body += f"dlm_source: {dlm_source}\n"
    spec_path.write_text(body, encoding="utf-8")


class TestPackBasics:
    def test_round_trip_minimal_spec(self, tmp_path: Path) -> None:
        spec_path = tmp_path / "sway.yaml"
        _write_minimal_spec(spec_path)
        out = tmp_path / "test.swaypack.tar.gz"

        report = pack_spec(spec_path, out_path=out, include_null_cache=False)

        assert out.exists()
        assert report.size_bytes > 0
        assert report.spec_path == spec_path.resolve()
        assert report.section_bytes == 0
        assert report.null_stats_count == 0
        assert report.golden_included is False

    def test_unpack_round_trip(self, tmp_path: Path) -> None:
        spec_path = tmp_path / "sway.yaml"
        _write_minimal_spec(spec_path)
        out = tmp_path / "test.swaypack.tar.gz"
        pack_spec(spec_path, out_path=out, include_null_cache=False)

        target = tmp_path / "unpacked"
        report = unpack_swaypack(out, target_dir=target)

        assert report.out_dir == (target / "swaypack").resolve()
        assert report.spec_path.exists()
        # Spec round-trips byte-identical.
        assert report.spec_path.read_bytes() == spec_path.read_bytes()
        assert report.null_stats_dir is None  # No cache packed.

    def test_manifest_records_pack_metadata(self, tmp_path: Path) -> None:
        """Manifest must include swaypack_version + sway_version
        + packed_at + counts so an unpacker can reason about the pack."""
        spec_path = tmp_path / "sway.yaml"
        _write_minimal_spec(spec_path)
        out = tmp_path / "test.swaypack.tar.gz"
        pack_spec(spec_path, out_path=out, include_null_cache=False)

        with tarfile.open(str(out), mode="r:gz") as tar:
            manifest_member = tar.getmember("swaypack/manifest.json")
            f = tar.extractfile(manifest_member)
            assert f is not None
            manifest = json.loads(f.read().decode("utf-8"))

        assert manifest["swaypack_version"] == SWAYPACK_VERSION
        assert "sway_version" in manifest
        assert "packed_at" in manifest
        assert manifest["section_bytes"] == 0
        assert manifest["null_stats_count"] == 0
        assert manifest["golden_included"] is False
        assert manifest["dlm_source_packed"] is False

    def test_dlm_source_bundled_when_present(self, tmp_path: Path) -> None:
        """A spec with ``dlm_source`` next to it bundles source.dlm."""
        dlm_path = tmp_path / "doc.dlm"
        dlm_path.write_text("---\ndlm_id: 01TEST\n---\nbody\n", encoding="utf-8")

        spec_path = tmp_path / "sway.yaml"
        _write_minimal_spec(spec_path, dlm_source="doc.dlm")
        out = tmp_path / "test.swaypack.tar.gz"

        report = pack_spec(spec_path, out_path=out, include_null_cache=False)
        assert report.section_bytes > 0

        target = tmp_path / "unpacked"
        unpack_report = unpack_swaypack(out, target_dir=target)
        bundled = unpack_report.out_dir / "source.dlm"
        assert bundled.exists()
        assert bundled.read_bytes() == dlm_path.read_bytes()

    def test_dlm_source_missing_logs_warning_but_succeeds(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Spec with dlm_source pointing at a nonexistent file: pack
        succeeds with section_bytes=0 + a warning."""
        spec_path = tmp_path / "sway.yaml"
        _write_minimal_spec(spec_path, dlm_source="missing.dlm")
        out = tmp_path / "test.swaypack.tar.gz"

        with caplog.at_level("WARNING"):
            report = pack_spec(spec_path, out_path=out, include_null_cache=False)
        assert report.section_bytes == 0
        assert any("missing.dlm" in rec.message for rec in caplog.records)


class TestPackErrors:
    def test_refuses_to_overwrite_existing_pack(self, tmp_path: Path) -> None:
        spec_path = tmp_path / "sway.yaml"
        _write_minimal_spec(spec_path)
        out = tmp_path / "test.swaypack.tar.gz"
        out.write_bytes(b"placeholder")
        with pytest.raises(PackError, match="refusing to overwrite"):
            pack_spec(spec_path, out_path=out, include_null_cache=False)

    def test_size_cap_rejects_oversize_pack(self, tmp_path: Path) -> None:
        """A 0-byte cap should reject any non-empty pack."""
        spec_path = tmp_path / "sway.yaml"
        _write_minimal_spec(spec_path)
        out = tmp_path / "test.swaypack.tar.gz"
        with pytest.raises(PackError, match="exceeds the cap"):
            pack_spec(
                spec_path,
                out_path=out,
                include_null_cache=False,
                max_size_bytes=10,
            )
        # File must NOT have been written when the cap was tripped.
        assert not out.exists()

    def test_default_max_size_is_50mb(self) -> None:
        """The default cap is the published 50 MB number — a regression
        guard so a stray edit doesn't quietly raise the bar."""
        assert DEFAULT_MAX_PACK_SIZE_BYTES == 50 * 1024 * 1024

    def test_include_golden_missing_raises(self, tmp_path: Path) -> None:
        spec_path = tmp_path / "sway.yaml"
        _write_minimal_spec(spec_path)
        out = tmp_path / "test.swaypack.tar.gz"
        with pytest.raises(PackError, match="--include-golden"):
            pack_spec(
                spec_path,
                out_path=out,
                include_golden=tmp_path / "nope.json",
                include_null_cache=False,
            )

    def test_include_golden_bundles_file(self, tmp_path: Path) -> None:
        spec_path = tmp_path / "sway.yaml"
        _write_minimal_spec(spec_path)
        golden = tmp_path / "report.json"
        golden.write_text(json.dumps({"verdict": "pass"}), encoding="utf-8")
        out = tmp_path / "test.swaypack.tar.gz"

        report = pack_spec(
            spec_path,
            out_path=out,
            include_golden=golden,
            include_null_cache=False,
        )
        assert report.golden_included is True

        target = tmp_path / "u"
        unpack_swaypack(out, target_dir=target)
        bundled = target / "swaypack" / "golden.json"
        assert bundled.exists()
        assert json.loads(bundled.read_text()) == {"verdict": "pass"}


class TestUnpackErrors:
    def test_missing_pack_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(UnpackError, match="not found"):
            unpack_swaypack(tmp_path / "no.tar.gz", target_dir=tmp_path)

    def test_pack_path_is_dir_raises(self, tmp_path: Path) -> None:
        d = tmp_path / "adir"
        d.mkdir()
        with pytest.raises(UnpackError, match="must be a file"):
            unpack_swaypack(d, target_dir=tmp_path)

    def test_refuses_to_overwrite_existing_swaypack_dir(self, tmp_path: Path) -> None:
        spec_path = tmp_path / "sway.yaml"
        _write_minimal_spec(spec_path)
        out = tmp_path / "test.swaypack.tar.gz"
        pack_spec(spec_path, out_path=out, include_null_cache=False)

        target = tmp_path / "u"
        unpack_swaypack(out, target_dir=target)  # First unpack succeeds.
        with pytest.raises(UnpackError, match="refusing to overwrite"):
            unpack_swaypack(out, target_dir=target)  # Second refuses.

    def test_corrupt_pack_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.tar.gz"
        bad.write_bytes(b"not a tarball")
        with pytest.raises(UnpackError, match="failed to extract"):
            unpack_swaypack(bad, target_dir=tmp_path)

    def test_pack_without_swaypack_root_raises(self, tmp_path: Path) -> None:
        """A tar.gz that doesn't follow the ``swaypack/`` layout is rejected."""
        bad = tmp_path / "wrong.tar.gz"
        with tarfile.open(str(bad), mode="w:gz") as tar:
            info = tarfile.TarInfo(name="other/sway.yaml")
            payload = b"version: 1\n"
            info.size = len(payload)
            import io as _io

            tar.addfile(info, _io.BytesIO(payload))
        with pytest.raises(UnpackError, match="missing the expected 'swaypack/'"):
            unpack_swaypack(bad, target_dir=tmp_path)

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        """A pack with the right top-level dir but no manifest fails."""
        bad = tmp_path / "no-manifest.tar.gz"
        with tarfile.open(str(bad), mode="w:gz") as tar:
            info = tarfile.TarInfo(name="swaypack/sway.yaml")
            payload = b"version: 1\n"
            info.size = len(payload)
            import io as _io

            tar.addfile(info, _io.BytesIO(payload))
        with pytest.raises(UnpackError, match="missing manifest.json"):
            unpack_swaypack(bad, target_dir=tmp_path)

    def test_version_mismatch_raises(self, tmp_path: Path) -> None:
        """A future swaypack_version is rejected with a clear message."""
        bad = tmp_path / "future.tar.gz"
        manifest = json.dumps({"swaypack_version": 99, "spec_filename": "sway.yaml"})
        with tarfile.open(str(bad), mode="w:gz") as tar:
            for name, payload in (
                ("swaypack/sway.yaml", b"version: 1\n"),
                ("swaypack/manifest.json", manifest.encode("utf-8")),
            ):
                info = tarfile.TarInfo(name=name)
                info.size = len(payload)
                import io as _io

                tar.addfile(info, _io.BytesIO(payload))
        with pytest.raises(UnpackError, match="unsupported swaypack_version=99"):
            unpack_swaypack(bad, target_dir=tmp_path)
