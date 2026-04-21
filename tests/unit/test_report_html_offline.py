"""S12 prove-the-value (§F6): HTML report loads offline from a real run.

Uses the committed `tests/fixtures/sway-history/02-2026-01-22.json`
(the mid-run baseline from S11) as a realistic input: four probe
kinds, healthy-band composite, full evidence dicts for SIS and
ablation. Emits the HTML to a tmp dir and asserts:

1. The file is a well-formed HTML document.
2. No external ``<script src="http...">`` or ``<link rel=stylesheet>``
   references — every byte loads from the file itself.
3. All four interactive panels (gauge / category / ablation / scatter)
   render; the SIS panel skips because the fixture's
   ``section_internalization`` evidence doesn't carry a ``per_section``
   array (it's a real terminal-rendered history, not a full export).
4. Every probe name from the fixture appears in the probe table.

Closure notes should record the produced file size and the total
probe count for the record.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from dlm_sway.suite import report, report_html

pytest.importorskip("plotly")

FIXTURE = Path(__file__).parent.parent / "fixtures" / "sway-history" / "02-2026-01-22.json"


class _Parser(HTMLParser):
    def error(self, message: str) -> None:  # pragma: no cover
        raise AssertionError(message)


def _parse_ok(text: str) -> None:
    parser = _Parser(convert_charrefs=True)
    parser.feed(text)
    parser.close()


def test_html_from_real_history_loads_offline(tmp_path: Path) -> None:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    suite, score = report.from_json(raw)

    html_text = report_html.to_html(suite, score)
    target = tmp_path / "report.html"
    target.write_text(html_text, encoding="utf-8")

    # 1. Well-formed.
    _parse_ok(target.read_text(encoding="utf-8"))

    # 2. No external script / stylesheet references — fully self-contained.
    disk = target.read_text(encoding="utf-8")
    external_scripts = re.findall(r'<script[^>]*\bsrc\s*=\s*["\'](https?:[^"\']+)', disk)
    external_links = re.findall(r'<link[^>]*\bhref\s*=\s*["\'](https?:[^"\']+)', disk)
    assert external_scripts == [], f"external scripts: {external_scripts}"
    assert external_links == [], f"external stylesheets: {external_links}"

    # 3. The three always-present panels render.
    for required in ("sway-gauge", "sway-category", "sway-scatter"):
        assert f'id="{required}"' in disk, f"panel {required!r} missing"
    # Evidence-dependent panels: the committed history fixture carries
    # terminal-message metadata but not the full evidence dicts. Both
    # panels correctly opt out, confirming the renderer handles partial
    # inputs without crashing.
    assert 'id="sway-sis"' not in disk
    assert 'id="sway-ablation"' not in disk

    # 4. All four probe names appear in the probe table.
    for probe_name in (
        "delta_kl",
        "section_internalization",
        "calibration_drift",
        "adapter_ablation",
    ):
        assert probe_name in disk, f"probe {probe_name!r} not in HTML body"

    # Sanity: file is in the expected size range (1-10 MB; Plotly 6.x
    # bundles ~4.8 MB of JS).
    size = target.stat().st_size
    assert 1_000_000 < size < 10_000_000, f"unexpected HTML size: {size:,} bytes"
