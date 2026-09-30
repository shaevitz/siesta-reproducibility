from __future__ import annotations

import pytest

from siesta_repro.metrics import compare_records
from test_records import record


def test_missing_predictions_count_as_pck_misses() -> None:
    reference = [record("a", "g1"), record("b", "g2")]
    prediction = [record("a", "g1"), record("b", "g2")]
    prediction[0]["points"]["head"]["xy"] = [13, 15]
    prediction[1]["points"] = {}
    result = compare_records(prediction, reference, thresholds=[4, 5])
    assert result["per_node"]["head"]["missing_predictions"] == 1
    assert result["per_node"]["head"]["pck"]["4.0"] == 0
    assert result["per_node"]["head"]["pck"]["5.0"] == pytest.approx(0.5)
    assert result["median_finite_distance"] == pytest.approx(5)
