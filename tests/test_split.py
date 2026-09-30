from __future__ import annotations

from siesta_repro.split import group_disjoint_split
from test_records import record


def test_split_is_deterministic_and_group_disjoint() -> None:
    rows = [record(f"s{i}", f"g{i // 2}") for i in range(12)]
    first = group_disjoint_split(rows, validation_fraction=1 / 3, seed=17)
    second = group_disjoint_split(rows, validation_fraction=1 / 3, seed=17)
    assert first == second
    train_groups = {row["source_group"] for row in first[0]}
    validation_groups = {row["source_group"] for row in first[1]}
    assert train_groups.isdisjoint(validation_groups)
    assert first[2]["group_disjoint"] is True
