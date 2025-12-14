from __future__ import annotations

from ArcticRoute.core.route.pareto import nondominated, pick_representatives


def test_nondominated_and_representatives_simple():
    points = [
        {"risk_integral": 10.0, "distance_km": 100.0, "congest_integral": 5.0},
        {"risk_integral": 8.0,  "distance_km": 120.0, "congest_integral": 4.0},
        {"risk_integral": 12.0, "distance_km": 90.0,  "congest_integral": 6.0},
        {"risk_integral": 9.0,  "distance_km": 95.0,  "congest_integral": 7.0},
    ]
    nd = nondominated(points)
    # Expect at least two on front (tradeoffs)
    assert len(nd) >= 2
    reps = pick_representatives([points[i] for i in nd])
    assert set(reps.keys()) == {"safe", "efficient", "balanced"}
    assert reps["safe"] >= 0 and reps["efficient"] >= 0 and reps["balanced"] >= 0









