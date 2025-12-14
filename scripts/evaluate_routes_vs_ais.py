"""
Compare planned routes against AIS corridor density for standard scenarios.

Usage:
    python -m scripts.evaluate_routes_vs_ais
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from arcticroute.config import EDL_MODES, SCENARIOS
from arcticroute.core.ais_analysis import evaluate_route_vs_ais_density
from arcticroute.core.ais_ingest import build_ais_density_da_for_demo_grid, AIS_RAW_DIR
from arcticroute.core.analysis import haversine_km
from arcticroute.core.astar import plan_route_latlon
from arcticroute.core.cost import build_cost_from_real_env
from arcticroute.core.env_real import load_real_env_for_grid
from arcticroute.core.grid import load_real_grid_from_nc
from arcticroute.core.landmask import load_real_landmask_from_nc


def _maybe_downsample(grid, land_mask: np.ndarray, ais_density: np.ndarray, max_cells: int = 200 * 200):
    ny, nx = grid.shape()
    total = ny * nx
    if total <= max_cells:
        return grid, land_mask, ais_density

    import math

    factor = int(math.ceil(math.sqrt(total / max_cells)))
    factor = max(factor, 1)
    slicer = (slice(None, None, factor), slice(None, None, factor))

    lat2d = grid.lat2d[slicer]
    lon2d = grid.lon2d[slicer]
    new_grid = type(grid)(lat2d=lat2d, lon2d=lon2d)
    new_land = land_mask[slicer]
    new_ais = ais_density[slicer]

    print(
        f"[INFO] downsampled grid from ({ny}, {nx}) to {new_grid.shape()} with factor {factor}"
    )
    return new_grid, new_land, new_ais


def _load_ais_density(grid_lat2d: np.ndarray, grid_lon2d: np.ndarray) -> np.ndarray | None:
    """从 AIS 原始目录加载密度；若不可用则返回 None。"""
    if not AIS_RAW_DIR.is_dir():
        print(f"[AIS] 原始 AIS 目录不存在: {AIS_RAW_DIR}")
        return None
    
    try:
        # 从原始 AIS 目录构建密度
        ais_da = build_ais_density_da_for_demo_grid(
            AIS_RAW_DIR,
            grid_lat2d[:, 0],  # 1D 纬度
            grid_lon2d[0, :],  # 1D 经度
        )
        return ais_da.values
    except Exception as e:
        print(f"[AIS] 从原始目录加载失败: {e}")
        return None


def _compute_distance_km(route: List[Tuple[float, float]]) -> float:
    if len(route) < 2:
        return 0.0
    dist = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(route[:-1], route[1:]):
        dist += haversine_km(lat1, lon1, lat2, lon2)
    return dist


def evaluate_scenario(
    scenario_name: str,
    grid,
    land_mask: np.ndarray,
    ais_density: np.ndarray,
) -> None:
    scenario = next((s for s in SCENARIOS if s.name == scenario_name), None)
    if scenario is None:
        print(f"[ERROR] Unknown scenario: {scenario_name}")
        return

    env = load_real_env_for_grid(grid, ym=scenario.ym)
    if env is None or (getattr(env, "grid", None) is not None and env.grid.shape() != grid.shape()):
        print("[WARN] real environment unavailable or mismatched; using demo-style cost.")
        env = None

    modes = ["efficient", "edl_safe", "edl_robust"]

    results: List[Dict[str, object]] = []
    for mode in modes:
        mode_cfg = EDL_MODES.get(mode)
        if mode_cfg is None:
            print(f"[WARN] skip unknown mode {mode}")
            continue

        cost_field = build_cost_from_real_env(
            grid=grid,
            land_mask=land_mask,
            env=env,
            ice_penalty=mode_cfg.get("ice_penalty", 4.0),
            wave_penalty=mode_cfg.get("wave_penalty", 0.0),
            vessel_profile=None,
            w_edl=mode_cfg.get("w_edl", 0.0),
            use_edl=mode_cfg.get("use_edl", False),
            use_edl_uncertainty=mode_cfg.get("use_edl_uncertainty", False),
            edl_uncertainty_weight=mode_cfg.get("edl_uncertainty_weight", 0.0),
            ais_density=ais_density,
            ais_weight=1.0,
            ym=scenario.ym if env is not None else None,
        )

        route = plan_route_latlon(
            cost_field,
            scenario.start_lat,
            scenario.start_lon,
            scenario.end_lat,
            scenario.end_lon,
            neighbor8=True,
        )

        if not route:
            results.append(
                {
                    "mode": mode,
                    "reachable": False,
                    "distance_km": 0.0,
                    "stats": None,
                }
            )
            continue

        distance_km = _compute_distance_km(route)
        stats = evaluate_route_vs_ais_density(
            route_latlon=route,
            grid_lats=grid.lat2d,
            grid_lons=grid.lon2d,
            ais_density=ais_density,
        )
        results.append(
            {
                "mode": mode,
                "reachable": True,
                "distance_km": distance_km,
                "stats": stats,
            }
        )

    print(f"[{scenario.name}]")
    header = (
        f"{'Mode':<12}{'Reachable':<12}{'Dist(km)':>10}"
        f"{'mean_AIS':>12}{'frac_high_corridor':>22}{'frac_low_usage':>18}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        if not row["reachable"]:
            print(f"{row['mode']:<12}{'No':<12}{'--':>10}{'--':>12}{'--':>22}{'--':>18}")
            continue
        stats = row["stats"]
        assert stats is not None
        print(
            f"{row['mode']:<12}"
            f"{'Yes':<12}"
            f"{row['distance_km']:>10.1f}"
            f"{stats.mean_density:>12.4f}"
            f"{stats.frac_high_corridor:>22.2f}"
            f"{stats.frac_low_usage:>18.2f}"
        )
    print()


def main() -> None:
    grid = load_real_grid_from_nc()
    if grid is None:
        print("[WARN] real grid unavailable; cannot evaluate AIS corridor adherence.")
        sys.exit(0)

    land_mask = load_real_landmask_from_nc(grid)
    if land_mask is None:
        print("[WARN] real landmask unavailable; using all-ocean mask for evaluation.")
        land_mask = np.zeros(grid.shape(), dtype=bool)

    ais_density = _load_ais_density(grid.lat2d, grid.lon2d)
    if ais_density is None:
        print("[WARN] AIS density not available; aborting evaluation.")
        sys.exit(0)

    grid, land_mask, ais_density = _maybe_downsample(grid, land_mask, ais_density)

    for scenario in SCENARIOS:
        evaluate_scenario(scenario.name, grid, land_mask, ais_density)


if __name__ == "__main__":
    main()
