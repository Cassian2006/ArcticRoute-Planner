import pandas as pd

from arcticroute.core.grid import make_demo_grid
from arcticroute.core.env_real import RealEnvLayers
from arcticroute.core.cost import build_cost_from_real_env
from arcticroute.core.astar import plan_route_latlon
from arcticroute.core.analysis import compute_route_cost_breakdown
from arcticroute.core.pareto import ParetoSolution, pareto_front, extract_objectives_from_breakdown, solutions_to_dataframe

def test_pareto_demo_smoke():
    grid, land_mask = make_demo_grid(ny=20, nx=20)

    ny, nx = grid.shape()
    sic = (0.3 * (1.0 - land_mask)).astype(float)
    wave = (1.0 * (1.0 - land_mask)).astype(float)
    thick = (0.2 * (1.0 - land_mask)).astype(float)
    env = RealEnvLayers(grid=grid, sic=sic, wave_swh=wave, ice_thickness_m=thick, land_mask=land_mask)

    candidates = [
        ("c1", dict(ice_penalty=2.0, wave_penalty=1.0, use_edl=False, w_edl=0.0, use_edl_uncertainty=False, edl_uncertainty_weight=0.0)),
        ("c2", dict(ice_penalty=6.0, wave_penalty=2.0, use_edl=True, w_edl=2.0, use_edl_uncertainty=False, edl_uncertainty_weight=0.0)),
        ("c3", dict(ice_penalty=4.0, wave_penalty=3.0, use_edl=True, w_edl=2.0, use_edl_uncertainty=True, edl_uncertainty_weight=2.0)),
    ]

    sols = []
    for key, cfg in candidates:
        cost_field = build_cost_from_real_env(grid, land_mask, env, **cfg)
        route = plan_route_latlon(cost_field, start_lat=66.0, start_lon=5.0, end_lat=78.0, end_lon=150.0, neighbor8=True)
        assert route
        bd = compute_route_cost_breakdown(grid, cost_field, route)
        sols.append(ParetoSolution(key, extract_objectives_from_breakdown(bd), list(route), dict(bd.component_totals), dict(cfg)))

    front = pareto_front(sols, fields=["distance_km", "total_cost"])
    assert len(front) >= 1

    df = solutions_to_dataframe(front)
    assert isinstance(df, pd.DataFrame)
    assert "distance_km" in df.columns
    assert "total_cost" in df.columns


