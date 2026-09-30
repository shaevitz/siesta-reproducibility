"""Crop-level point distances after association has been frozen upstream."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .records import NODES, validate_records


def compare_records(
    predictions: Iterable[Mapping[str, Any]],
    references: Iterable[Mapping[str, Any]],
    *,
    thresholds: Sequence[float] = (2.0, 4.0, 8.0, 16.0),
) -> dict[str, Any]:
    predicted = [dict(row) for row in predictions]
    reference = [dict(row) for row in references]
    validate_records(predicted, development=False)
    validate_records(reference, development=False)
    if not thresholds or any(not math.isfinite(value) or value <= 0 for value in thresholds):
        raise ValueError("thresholds must be finite and positive")
    by_id = {str(row["sample_id"]): row for row in predicted}
    if set(by_id) != {str(row["sample_id"]) for row in reference}:
        raise ValueError("prediction and reference sample_id sets must match exactly")
    distances: dict[str, list[float]] = defaultdict(list)
    denominators = {node: 0 for node in NODES}
    missing = {node: 0 for node in NODES}
    hits = {node: {str(float(t)): 0 for t in thresholds} for node in NODES}
    for target in reference:
        source = by_id[str(target["sample_id"])]
        for node, target_point in target["points"].items():
            denominators[node] += 1
            source_point = source["points"].get(node)
            if source_point is None:
                missing[node] += 1
                continue
            distance = float(np.linalg.norm(np.asarray(source_point["xy"], dtype=float) - np.asarray(target_point["xy"], dtype=float)))
            distances[node].append(distance)
            for threshold in thresholds:
                if distance <= threshold:
                    hits[node][str(float(threshold))] += 1
    per_node: dict[str, Any] = {}
    for node in NODES:
        values = distances[node]
        denominator = denominators[node]
        per_node[node] = {
            "reference_points": denominator,
            "finite_predictions": len(values),
            "missing_predictions": missing[node],
            "median_distance": float(np.median(values)) if values else None,
            "pck": {key: (count / denominator if denominator else None) for key, count in hits[node].items()},
        }
    macro = {}
    for threshold in thresholds:
        key = str(float(threshold))
        values = [per_node[node]["pck"][key] for node in NODES if per_node[node]["pck"][key] is not None]
        macro[key] = float(np.mean(values)) if values else None
    all_distances = [distance for values in distances.values() for distance in values]
    return {
        "association": "sample_id equality; association must be frozen upstream",
        "missing_predictions_count_as_pck_misses": True,
        "samples": len(reference),
        "finite_distances": len(all_distances),
        "median_finite_distance": float(np.median(all_distances)) if all_distances else None,
        "macro_node_pck": macro,
        "per_node": per_node,
    }
