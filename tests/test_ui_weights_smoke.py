from __future__ import annotations

from arcticroute.ui import planner_minimal


def test_default_weights_keys_complete():
    weights = planner_minimal.default_weights()
    expected_keys = {
        "w_sic",
        "w_swh",
        "w_sit",
        "w_drift",
        "w_ais",
        "w_corridor",
        "w_congestion",
        "w_shallow",
        "w_polaris",
    }
    assert set(weights.keys()) >= expected_keys


