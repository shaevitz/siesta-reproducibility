from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from siesta_repro.records import validate_records


def record(sample_id: str, group: str) -> dict:
    return {
        "sample_id": sample_id,
        "dataset": "synthetic",
        "source_group": group,
        "image_path": f"{sample_id}.png",
        "image_sha256": "0" * 64,
        "width": 32,
        "height": 32,
        "points": {"head": {"xy": [10, 11], "state": "observed", "confidence": "high"}},
        "provenance": {"coordinate_source": "VLM", "human_or_sealed_labels_used": False, "classical_coordinate_method_used": False},
    }


def test_valid_records_are_summarized() -> None:
    summary = validate_records([record("a", "g1"), record("b", "g2")])
    assert summary["records"] == 2
    assert summary["node_counts"]["head"] == 2


def test_human_labels_are_rejected() -> None:
    value = record("a", "g1")
    value["provenance"]["human_or_sealed_labels_used"] = True
    with pytest.raises(PermissionError):
        validate_records([value])


def test_image_hash_can_be_verified(tmp_path: Path) -> None:
    image = tmp_path / "a.png"
    image.write_bytes(b"synthetic image bytes")
    value = record("a", "g1")
    value["image_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
    validate_records([value], manifest_path=tmp_path / "records.jsonl", verify_images=True)
