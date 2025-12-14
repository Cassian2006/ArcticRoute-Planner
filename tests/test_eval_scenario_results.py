"""Unit tests for eval_scenario_results.py"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import numpy as np

from scripts.eval_scenario_results import evaluate


@pytest.fixture
def sample_scenario_df() -> pd.DataFrame:
    """Create a sample scenario results DataFrame for testing."""
    data = [
        # Scenario 1: barents_to_chukchi
        {
            "scenario_id": "barents_to_chukchi",
            "mode": "efficient",
            "reachable": True,
            "distance_km": 4000.0,
            "total_cost": 50.0,
            "edl_risk_cost": 10.0,
            "edl_uncertainty_cost": 2.0,
        },
        {
            "scenario_id": "barents_to_chukchi",
            "mode": "edl_safe",
            "reachable": True,
            "distance_km": 4100.0,
            "total_cost": 52.0,
            "edl_risk_cost": 5.0,
            "edl_uncertainty_cost": 1.5,
        },
        {
            "scenario_id": "barents_to_chukchi",
            "mode": "edl_robust",
            "reachable": True,
            "distance_km": 4200.0,
            "total_cost": 54.0,
            "edl_risk_cost": 2.0,
            "edl_uncertainty_cost": 1.0,
        },
        # Scenario 2: kara_short
        {
            "scenario_id": "kara_short",
            "mode": "efficient",
            "reachable": True,
            "distance_km": 1000.0,
            "total_cost": 20.0,
            "edl_risk_cost": 0.0,
            "edl_uncertainty_cost": 0.0,
        },
        {
            "scenario_id": "kara_short",
            "mode": "edl_safe",
            "reachable": True,
            "distance_km": 1050.0,
            "total_cost": 21.0,
            "edl_risk_cost": 0.0,
            "edl_uncertainty_cost": 0.0,
        },
        {
            "scenario_id": "kara_short",
            "mode": "edl_robust",
            "reachable": True,
            "distance_km": 1100.0,
            "total_cost": 22.0,
            "edl_risk_cost": 0.0,
            "edl_uncertainty_cost": 0.0,
        },
    ]
    return pd.DataFrame(data)


def test_evaluate_delta_calculations(sample_scenario_df):
    """Test that delta and percentage calculations are correct."""
    eval_df = evaluate(sample_scenario_df)

    # Check barents_to_chukchi edl_safe
    barents_safe = eval_df[
        (eval_df["scenario_id"] == "barents_to_chukchi")
        & (eval_df["mode"] == "edl_safe")
    ]
    assert len(barents_safe) == 1
    row = barents_safe.iloc[0]

    # Expected: dist_eff=4000, dist_safe=4100
    # delta_dist = 4100 - 4000 = 100
    assert row["delta_dist_km"] == pytest.approx(100.0)
    # rel_dist_pct = 100 * 100 / 4000 = 2.5
    assert row["rel_dist_pct"] == pytest.approx(2.5)

    # Expected: cost_eff=50, cost_safe=52
    # delta_cost = 52 - 50 = 2
    assert row["delta_cost"] == pytest.approx(2.0)
    # rel_cost_pct = 100 * 2 / 50 = 4.0
    assert row["rel_cost_pct"] == pytest.approx(4.0)

    # Expected: risk_eff=10, risk_safe=5
    # delta_risk = 5 - 10 = -5
    assert row["delta_edl_risk"] == pytest.approx(-5.0)
    # risk_reduction_pct = 100 * (10 - 5) / 10 = 50
    assert row["risk_reduction_pct"] == pytest.approx(50.0)


def test_evaluate_robust_mode(sample_scenario_df):
    """Test evaluation for edl_robust mode."""
    eval_df = evaluate(sample_scenario_df)

    # Check barents_to_chukchi edl_robust
    barents_robust = eval_df[
        (eval_df["scenario_id"] == "barents_to_chukchi")
        & (eval_df["mode"] == "edl_robust")
    ]
    assert len(barents_robust) == 1
    row = barents_robust.iloc[0]

    # Expected: dist_eff=4000, dist_robust=4200
    # delta_dist = 4200 - 4000 = 200
    assert row["delta_dist_km"] == pytest.approx(200.0)
    # rel_dist_pct = 100 * 200 / 4000 = 5.0
    assert row["rel_dist_pct"] == pytest.approx(5.0)

    # Expected: risk_eff=10, risk_robust=2
    # risk_reduction_pct = 100 * (10 - 2) / 10 = 80
    assert row["risk_reduction_pct"] == pytest.approx(80.0)


def test_evaluate_zero_baseline_risk(sample_scenario_df):
    """Test that risk_reduction_pct is NaN when baseline risk is 0."""
    eval_df = evaluate(sample_scenario_df)

    # Check kara_short (baseline risk = 0)
    kara_safe = eval_df[
        (eval_df["scenario_id"] == "kara_short") & (eval_df["mode"] == "edl_safe")
    ]
    assert len(kara_safe) == 1
    row = kara_safe.iloc[0]

    # risk_reduction_pct should be NaN when baseline risk is 0
    assert pd.isna(row["risk_reduction_pct"])

    # But distance/cost deltas should still be computed
    assert row["delta_dist_km"] == pytest.approx(50.0)
    assert row["rel_dist_pct"] == pytest.approx(5.0)


def test_evaluate_missing_efficient_mode():
    """Test that scenario is skipped when efficient mode is missing."""
    data = [
        {
            "scenario_id": "incomplete_scenario",
            "mode": "edl_safe",
            "reachable": True,
            "distance_km": 1000.0,
            "total_cost": 20.0,
            "edl_risk_cost": 5.0,
            "edl_uncertainty_cost": 1.0,
        },
    ]
    df = pd.DataFrame(data)
    eval_df = evaluate(df)

    # Should be empty since efficient mode is missing
    assert len(eval_df) == 0


def test_evaluate_unreachable_routes():
    """Test that unreachable routes are filtered out."""
    data = [
        {
            "scenario_id": "test_scenario",
            "mode": "efficient",
            "reachable": False,
            "distance_km": 0.0,
            "total_cost": 0.0,
            "edl_risk_cost": 0.0,
            "edl_uncertainty_cost": 0.0,
        },
        {
            "scenario_id": "test_scenario",
            "mode": "edl_safe",
            "reachable": False,
            "distance_km": 0.0,
            "total_cost": 0.0,
            "edl_risk_cost": 0.0,
            "edl_uncertainty_cost": 0.0,
        },
    ]
    df = pd.DataFrame(data)
    eval_df = evaluate(df)

    # Should be empty since no reachable routes
    assert len(eval_df) == 0


def test_evaluate_missing_edl_cost_columns():
    """Test that missing EDL cost columns are handled gracefully."""
    data = [
        {
            "scenario_id": "test_scenario",
            "mode": "efficient",
            "reachable": True,
            "distance_km": 1000.0,
            "total_cost": 20.0,
        },
        {
            "scenario_id": "test_scenario",
            "mode": "edl_safe",
            "reachable": True,
            "distance_km": 1050.0,
            "total_cost": 21.0,
        },
    ]
    df = pd.DataFrame(data)
    eval_df = evaluate(df)

    # Should work with missing columns (defaults to 0)
    assert len(eval_df) == 1
    row = eval_df.iloc[0]
    assert row["delta_edl_risk"] == pytest.approx(0.0)


def test_evaluate_output_columns(sample_scenario_df):
    """Test that output DataFrame has all expected columns."""
    eval_df = evaluate(sample_scenario_df)

    expected_cols = [
        "scenario_id",
        "mode",
        "delta_dist_km",
        "rel_dist_pct",
        "delta_cost",
        "rel_cost_pct",
        "delta_edl_risk",
        "risk_reduction_pct",
        "delta_edl_unc",
    ]

    for col in expected_cols:
        assert col in eval_df.columns, f"Missing column: {col}"


def test_evaluate_multiple_scenarios(sample_scenario_df):
    """Test evaluation with multiple scenarios."""
    eval_df = evaluate(sample_scenario_df)

    # Should have 4 rows: 2 scenarios * 2 modes (edl_safe, edl_robust)
    assert len(eval_df) == 4

    # Check scenario_id values
    scenarios = eval_df["scenario_id"].unique()
    assert set(scenarios) == {"barents_to_chukchi", "kara_short"}

    # Check mode values
    modes = eval_df["mode"].unique()
    assert set(modes) == {"edl_safe", "edl_robust"}


def test_evaluate_csv_roundtrip(sample_scenario_df):
    """Test that evaluation results can be written to and read from CSV."""
    eval_df = evaluate(sample_scenario_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_eval.csv"
        eval_df.to_csv(csv_path, index=False)

        # Read back
        loaded_df = pd.read_csv(csv_path)

        # Check shape and columns
        assert loaded_df.shape == eval_df.shape
        assert list(loaded_df.columns) == list(eval_df.columns)

        # Check some values
        assert loaded_df["scenario_id"].tolist() == eval_df["scenario_id"].tolist()
        assert loaded_df["mode"].tolist() == eval_df["mode"].tolist()



