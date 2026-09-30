"""Coordinate-preserving export of VLM records to SLEAP packages."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .records import EDGES, NODES, resolve_image, sha256_file, validate_records


def _sleap_io() -> Any:
    try:
        return importlib.import_module("sleap_io")
    except ImportError as exc:
        raise ImportError("Install the SLEAP extra with: pip install -e '.[sleap]'") from exc


def export_sleap(
    records: Iterable[Mapping[str, Any]],
    output: str | Path,
    *,
    manifest_path: str | Path,
    images_root: str | Path | None = None,
    coordinate_scale: float = 1.0,
) -> dict[str, Any]:
    rows = [dict(record) for record in records]
    validate_records(rows, manifest_path=manifest_path, images_root=images_root, verify_images=True)
    if not np.isfinite(coordinate_scale) or coordinate_scale <= 0:
        raise ValueError("coordinate_scale must be finite and positive")
    sio = _sleap_io()
    skeleton = sio.Skeleton(nodes=list(NODES), edges=list(EDGES), name="siesta_fly13")
    frames = []
    for row in rows:
        image = resolve_image(row, manifest_path=manifest_path, images_root=images_root)
        video = sio.Video(filename=[str(image)], open_backend=True)
        xy = np.full((len(NODES), 2), np.nan, dtype=float)
        for index, node in enumerate(NODES):
            if node in row["points"]:
                xy[index] = np.asarray(row["points"][node]["xy"], dtype=float) * coordinate_scale
        instance = sio.Instance.from_numpy(xy, skeleton=skeleton)
        frames.append(sio.LabeledFrame(video=video, frame_idx=0, instances=[instance]))
    labels = sio.Labels(
        labeled_frames=frames,
        skeletons=[skeleton],
        provenance={
            "label_source": "VLM only",
            "human_or_sealed_labels_used": False,
            "classical_coordinate_method_used": False,
            "coordinate_scale": coordinate_scale,
            "coordinate_transport_only": True,
        },
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sio.save_slp(labels, destination)
    roundtrip = sio.load_slp(destination)
    if len(roundtrip) != len(rows) or tuple(roundtrip.skeletons[0].node_names) != NODES:
        raise ValueError("SLEAP round trip changed frame count or skeleton")
    if any(frame.predicted_instances or len(frame.user_instances) != 1 for frame in roundtrip):
        raise ValueError("SLEAP round trip changed VLM records from user-label training instances")
    return {
        "output": str(destination.resolve()),
        "sha256": sha256_file(destination),
        "records": len(rows),
        "nodes": list(NODES),
        "coordinate_scale": coordinate_scale,
        "roundtrip_verified": True,
    }
