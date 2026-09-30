"""Strict readers for VLM-origin pose records."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

NODES = (
    "head", "thorax", "abdomen", "wingL", "wingR",
    "forelegL4", "forelegR4", "midlegL4", "midlegR4",
    "hindlegL4", "hindlegR4", "eyeL", "eyeR",
)

EDGES = (
    ("head", "thorax"), ("thorax", "abdomen"),
    ("thorax", "wingL"), ("thorax", "wingR"),
    ("thorax", "forelegL4"), ("thorax", "forelegR4"),
    ("thorax", "midlegL4"), ("thorax", "midlegR4"),
    ("thorax", "hindlegL4"), ("thorax", "hindlegR4"),
    ("head", "eyeL"), ("head", "eyeR"),
)

STATES = frozenset({"observed", "inferred", "prior_only"})
CONFIDENCE = frozenset({"high", "medium", "low"})
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{source}:{line_number}: each JSONL value must be an object")
        records.append(value)
    return records


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(canonical_json(record) + "\n")


def resolve_image(record: Mapping[str, Any], *, manifest_path: str | Path, images_root: str | Path | None = None) -> Path:
    image = Path(str(record["image_path"]))
    if image.is_absolute():
        return image
    base = Path(images_root) if images_root is not None else Path(manifest_path).resolve().parent
    return (base / image).resolve()


def _validate_point(sample_id: str, node: str, point: Any, *, width: int, height: int) -> None:
    if node not in NODES:
        raise ValueError(f"{sample_id}: unknown node {node!r}")
    if not isinstance(point, Mapping):
        raise ValueError(f"{sample_id}/{node}: point must be an object")
    xy = point.get("xy")
    if not isinstance(xy, list) or len(xy) != 2 or any(type(v) not in (int, float) for v in xy):
        raise ValueError(f"{sample_id}/{node}: xy must contain two numbers")
    x, y = map(float, xy)
    if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"{sample_id}/{node}: xy is nonfinite or outside the declared canvas")
    if point.get("state") not in STATES:
        raise ValueError(f"{sample_id}/{node}: state must be one of {sorted(STATES)}")
    if point.get("confidence") not in CONFIDENCE:
        raise ValueError(f"{sample_id}/{node}: confidence must be one of {sorted(CONFIDENCE)}")


def validate_record(record: Mapping[str, Any], *, development: bool = True) -> None:
    required = {"sample_id", "dataset", "source_group", "image_path", "image_sha256", "width", "height", "points", "provenance"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"record is missing required keys: {sorted(missing)}")
    sample_id = str(record["sample_id"])
    if not sample_id or not str(record["source_group"]):
        raise ValueError("sample_id and source_group must be nonempty")
    width, height = record["width"], record["height"]
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError(f"{sample_id}: width and height must be positive integers")
    if not SHA256_PATTERN.fullmatch(str(record["image_sha256"])):
        raise ValueError(f"{sample_id}: image_sha256 must be a lowercase SHA-256 digest")
    points = record["points"]
    if not isinstance(points, Mapping):
        raise ValueError(f"{sample_id}: points must be an object")
    for node, point in points.items():
        _validate_point(sample_id, node, point, width=width, height=height)
    provenance = record["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{sample_id}: provenance must be an object")
    if not isinstance(provenance.get("coordinate_source"), str) or not provenance["coordinate_source"]:
        raise ValueError(f"{sample_id}: provenance must name a coordinate_source")
    if development:
        if provenance.get("coordinate_source") != "VLM":
            raise PermissionError(f"{sample_id}: every training coordinate must come from a VLM")
        if provenance.get("human_or_sealed_labels_used") is not False:
            raise PermissionError(f"{sample_id}: human/sealed labels are forbidden in development records")
        if provenance.get("classical_coordinate_method_used") is not False:
            raise PermissionError(f"{sample_id}: classical coordinate methods are forbidden")


def validate_records(
    records: Iterable[Mapping[str, Any]],
    *,
    manifest_path: str | Path | None = None,
    images_root: str | Path | None = None,
    verify_images: bool = False,
    development: bool = True,
) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("record set is empty")
    seen: set[str] = set()
    node_counts = {node: 0 for node in NODES}
    for record in rows:
        validate_record(record, development=development)
        sample_id = str(record["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        for node in record["points"]:
            node_counts[node] += 1
        if verify_images:
            if manifest_path is None:
                raise ValueError("manifest_path is required when verifying images")
            image = resolve_image(record, manifest_path=manifest_path, images_root=images_root)
            if not image.is_file():
                raise FileNotFoundError(image)
            observed = sha256_file(image)
            if observed != record["image_sha256"]:
                raise ValueError(f"{sample_id}: image SHA-256 mismatch")
    summary = {
        "records": len(rows),
        "source_groups": len({str(row["source_group"]) for row in rows}),
        "datasets": sorted({str(row["dataset"]) for row in rows}),
        "node_counts": node_counts,
        "coordinate_sources": sorted({str(row["provenance"]["coordinate_source"]) for row in rows}),
    }
    if development:
        summary.update(human_or_sealed_labels_used=False, classical_coordinate_method_used=False)
    return summary


def load_records(path: str | Path, *, images_root: str | Path | None = None, verify_images: bool = False, development: bool = True) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    validate_records(rows, manifest_path=path, images_root=images_root, verify_images=verify_images, development=development)
    return rows
