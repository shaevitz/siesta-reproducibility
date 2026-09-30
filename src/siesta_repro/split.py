"""Deterministic source-group-disjoint data splitting."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from typing import Any, Iterable, Mapping

from .records import canonical_json, validate_records


def _rank(seed: int, group: str) -> str:
    return hashlib.sha256(canonical_json([seed, group]).encode("utf-8")).hexdigest()


def group_disjoint_split(
    records: Iterable[Mapping[str, Any]],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(record) for record in records]
    validate_records(rows)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between zero and one")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_group"])].append(row)
    if len(grouped) < 2:
        raise ValueError("at least two source groups are required")
    target = len(rows) * validation_fraction
    ordered = sorted(grouped, key=lambda group: (_rank(seed, group), group))
    validation_groups: set[str] = set()
    validation_size = 0
    for group in ordered:
        size = len(grouped[group])
        before = abs(validation_size - target)
        after = abs(validation_size + size - target)
        if after <= before or not validation_groups:
            validation_groups.add(group)
            validation_size += size
    if len(validation_groups) == len(grouped):
        validation_groups.remove(max(validation_groups, key=lambda group: (_rank(seed, group), group)))
    train, validation = [], []
    for row in rows:
        destination = validation if str(row["source_group"]) in validation_groups else train
        destination.append({**row, "split": "validation" if destination is validation else "train"})
    train.sort(key=lambda row: str(row["sample_id"]))
    validation.sort(key=lambda row: str(row["sample_id"]))
    train_groups = {str(row["source_group"]) for row in train}
    observed_validation_groups = {str(row["source_group"]) for row in validation}
    if train_groups & observed_validation_groups:
        raise AssertionError("source-group leakage")
    receipt = {
        "seed": seed,
        "requested_validation_fraction": validation_fraction,
        "records": {"train": len(train), "validation": len(validation)},
        "source_groups": {"train": len(train_groups), "validation": len(observed_validation_groups)},
        "group_disjoint": True,
        "ranking": "sha256(canonical_json([seed, source_group]))",
    }
    return train, validation, receipt
