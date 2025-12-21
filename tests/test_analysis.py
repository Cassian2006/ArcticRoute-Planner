import numpy as np
from arcticroute.core.analysis import compute_route_cost_breakdown, RouteCostBreakdown
from arcticroute.core.cost import build_demo_cost
from arcticroute.core.grid import make_demo_grid

def test_compute_route_cost_breakdown_empty_route():
    """Test that an empty route results in zero cost and empty lists."""
    grid, land_mask = make_demo_grid()
    cost_field = build_demo_cost(grid, land_mask)
    breakdown = compute_route_cost_breakdown(grid, cost_field, [])
    assert isinstance(breakdown, RouteCostBreakdown)
    assert breakdown.total_cost == 0.0
    assert breakdown.s_km == []
    # In the minimal implementation, these might be non-empty dicts with 0 values
    if breakdown.component_totals:
        assert all(v == 0 for v in breakdown.component_totals.values())
    if breakdown.component_along_path:
         assert all(v == [] for v in breakdown.component_along_path.values())

def test_compute_route_cost_breakdown_simple_route():
    """Test breakdown for a simple route has expected structure and values."""
    grid, land_mask = make_demo_grid()
    cost_field = build_demo_cost(grid, land_mask)
    route = [(66.0, 5.0), (67.0, 7.0)]  # A short route with 2 points
    
    breakdown = compute_route_cost_breakdown(grid, cost_field, route)
    
    assert isinstance(breakdown, RouteCostBreakdown)
    assert breakdown.total_cost > 0
    assert len(breakdown.s_km) == len(route)
    assert breakdown.s_km[0] == 0.0
    assert breakdown.s_km[-1] > 0.0
    
    # Check for expected components from build_demo_cost
    assert "base_distance" in breakdown.component_totals
    assert "ice_risk" in breakdown.component_totals
    
    # Check that along-path data has the correct length
    for component_path in breakdown.component_along_path.values():
        assert len(component_path) == len(route)







