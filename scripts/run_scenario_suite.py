"""Batch runner for predefined scenarios.

Usage:
    python -m scripts.run_scenario_suite
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List

from arcticroute.core.analysis import compute_route_cost_breakdown
from arcticroute.core.eco.vessel_profiles import get_default_profiles
from arcticroute.core.grid import load_real_grid_from_nc, make_demo_grid
from arcticroute.core.landmask import load_real_landmask_from_nc
from arcticroute.core.scenarios import ScenarioConfig, load_all_scenarios
from arcticroute.ui import planner_minimal


OUTPUT_COLUMNS = [
    "scenario_id",
    "mode",
    "grid_mode",
    "reachable",
    "distance_km",
    "total_cost",
    "base_distance_cost",
    "ice_cost",
    "wave_cost",
    "ais_cost",
    "ais_corridor_cost",
    "ais_congestion_cost",
    "edl_risk_cost",
    "edl_uncertainty_cost",
    "vessel",
    "w_ice",
    "w_wave",
    "w_ais",
    "w_ais_corridor",
    "w_ais_congestion",
    "use_edl",
    "use_edl_uncertainty",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all scenarios defined in configs/scenarios.yaml")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports") / "scenario_suite_results.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def load_grid_and_landmask(grid_mode: str):
    """Load grid + landmask based on requested mode, with fallback to demo."""
    if grid_mode == "real":
        grid = load_real_grid_from_nc()
        if grid is not None:
            landmask = load_real_landmask_from_nc(grid)
            if landmask is not None:
                return grid, landmask, "real"
    grid, landmask = make_demo_grid()
    return grid, landmask, "demo"


def _component_total(breakdown, name: str) -> float:
    if breakdown is None:
        return 0.0
    return float(breakdown.component_totals.get(name, 0.0))


def run_single_scenario(scen: ScenarioConfig) -> list[dict]:
    """Run one scenario and return list of row dicts (one per route profile)."""
    grid, land_mask, grid_used = load_grid_and_landmask(scen.grid_mode)
    cost_mode = "real_sic_if_available" if grid_used == "real" else "demo_icebelt"

    vessel_profiles = get_default_profiles()
    vessel = vessel_profiles.get(scen.vessel)

    w_edl = scen.w_ice if scen.use_edl else 0.0
    edl_uncertainty_weight = w_edl if scen.use_edl_uncertainty else 0.0

    routes_info, cost_fields, _, _, _ = planner_minimal.plan_three_routes(
        grid,
        land_mask,
        scen.start_lat,
        scen.start_lon,
        scen.end_lat,
        scen.end_lon,
        allow_diag=True,
        vessel=vessel,
        cost_mode=cost_mode,
        wave_penalty=scen.w_wave,
        use_edl=scen.use_edl,
        w_edl=w_edl,
        weight_risk=1.0,
        weight_uncertainty=1.0,
        weight_fuel=1.0,
        edl_uncertainty_weight=edl_uncertainty_weight,
        w_ais=scen.w_ais,
        w_ais_corridor=getattr(scen, "w_ais_corridor", scen.w_ais),
        w_ais_congestion=getattr(scen, "w_ais_congestion", 0.0),
    )

    rows: list[dict] = []
    for idx, profile in enumerate(planner_minimal.ROUTE_PROFILES):
        key = profile["key"]
        route_info = routes_info[idx] if idx < len(routes_info) else None
        cost_field = cost_fields.get(key)
        breakdown = None
        if route_info and route_info.reachable and cost_field is not None:
            breakdown = compute_route_cost_breakdown(grid, cost_field, route_info.coords)

        reachable = bool(route_info and route_info.reachable)
        distance_km = float(route_info.approx_length_km or 0.0) if route_info else 0.0
        total_cost = float(breakdown.total_cost) if breakdown is not None else 0.0
        corridor_cost = _component_total(breakdown, "ais_corridor")
        congestion_cost = _component_total(breakdown, "ais_congestion")
        legacy_ais_cost = _component_total(breakdown, "ais_density")
        ais_cost_total = corridor_cost + congestion_cost
        if ais_cost_total == 0.0:
            ais_cost_total = legacy_ais_cost

        rows.append(
            {
                "scenario_id": scen.id,
                "mode": key,
                "grid_mode": grid_used,
                "reachable": reachable,
                "distance_km": distance_km,
                "total_cost": total_cost,
                "base_distance_cost": _component_total(breakdown, "base_distance"),
                "ice_cost": _component_total(breakdown, "ice_risk"),
                "wave_cost": _component_total(breakdown, "wave_risk"),
                "ais_cost": ais_cost_total,
                "ais_corridor_cost": corridor_cost,
                "ais_congestion_cost": congestion_cost,
                "edl_risk_cost": _component_total(breakdown, "edl_risk"),
                "edl_uncertainty_cost": _component_total(breakdown, "edl_uncertainty_penalty"),
                "vessel": scen.vessel,
                "w_ice": float(scen.w_ice),
                "w_wave": float(scen.w_wave),
                "w_ais": float(scen.w_ais),
                "w_ais_corridor": float(getattr(scen, "w_ais_corridor", scen.w_ais)),
                "w_ais_congestion": float(getattr(scen, "w_ais_congestion", 0.0)),
                "use_edl": bool(scen.use_edl),
                "use_edl_uncertainty": bool(scen.use_edl_uncertainty),
            }
        )

    return rows


def print_summary(rows: Iterable[dict]) -> None:
    lines = []
    header = f"{'Scenario':<26} {'Mode':<11} {'Grid':<6} {'Reachable':<10} {'Dist(km)':>9} {'Total':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        lines.append(
            f"{row['scenario_id']:<26} "
            f"{row['mode']:<11} "
            f"{row['grid_mode']:<6} "
            f"{'Yes' if row['reachable'] else 'No':<10} "
            f"{row['distance_km']:>9.1f} "
            f"{row['total_cost']:>10.2f}"
        )
    print("\n".join(lines))


def main() -> None:
    args = parse_args()

    scenarios = load_all_scenarios()
    all_rows: List[dict] = []
    for scen in scenarios.values():
        rows = run_single_scenario(scen)
        all_rows.extend(rows)

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print_summary(all_rows)
    print(f"\n[INFO] Wrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
