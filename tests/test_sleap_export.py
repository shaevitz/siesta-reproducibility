from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

pytest.importorskip("sleap_io")
from PIL import Image

from siesta_repro.records import write_jsonl
from siesta_repro.sleap_export import export_sleap
from test_records import record


def test_sleap_export_roundtrip(tmp_path: Path) -> None:
    image = tmp_path / "a.png"
    Image.new("RGB", (32, 32), "white").save(image)
    value = record("a", "g1")
    value["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
    manifest = tmp_path / "records.jsonl"
    write_jsonl(manifest, [value])
    result = export_sleap([value], tmp_path / "training.slp", manifest_path=manifest, coordinate_scale=0.5)
    assert result["roundtrip_verified"] is True
    assert result["coordinate_scale"] == 0.5
