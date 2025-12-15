import pytest
from arcticroute.core.constraints.polaris import (
    thickness_to_ice_type, compute_rio_for_cell, classify_operation_level, recommended_speed_limit_knots
)

def test_thickness_bins():
    assert thickness_to_ice_type(0.14, 0.9) == "grey_ice"
    assert thickness_to_ice_type(0.20, 0.9) == "grey_white_ice"
    assert thickness_to_ice_type(0.40, 0.9) == "thin_fy_1st"
    assert thickness_to_ice_type(0.60, 0.9) == "thin_fy_2nd"
    assert thickness_to_ice_type(0.90, 0.9) == "medium_fy"
    assert thickness_to_ice_type(1.50, 0.9) == "thick_fy"
    assert thickness_to_ice_type(2.20, 0.9) == "second_year"
    assert thickness_to_ice_type(3.00, 0.9) == "multi_year"

def test_rio_formula_simple():
    # SIC=0.6 => 6/10 ice, 4/10 open water
    meta = compute_rio_for_cell(sic=0.6, thickness_m=0.4, ice_class="PC4")
    # PC4: open water RIV=3; thin_fy_1st RIV=1 (table 1.3)
    assert meta.rio == pytest.approx(4*3 + 6*1)

def test_operation_thresholds_and_speed():
    assert classify_operation_level(0, "PC6") == "normal"
    assert classify_operation_level(-1, "PC6") == "elevated"
    assert classify_operation_level(-11, "PC6") == "special"
    assert recommended_speed_limit_knots("elevated","PC2") == 8.0
    assert recommended_speed_limit_knots("elevated","PC4") == 5.0


