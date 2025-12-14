"""
AIS 权重灵敏度分析脚本。
对若干起终点，在真实网格上分别测试 w_ais ∈ {0,4,8} 对 balanced/EDL-Safe 方案的成本与 AIS 分量影响。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from arcticroute.core.analysis import compute_route_cost_breakdown
from arcticroute.core.env_real import load_real_env_for_grid
from arcticroute.ui.planner_minimal import (
    plan_three_routes,
    ROUTE_PROFILES,
    compute_path_length_km,
)


W_AIS_VALUES = [0.0, 4.0, 8.0]

ROUTES: List[Tuple[str, float, float, float, float]] = [
    ("barents_to_chukchi", 69.0, 33.0, 70.5, 170.0),
    ("kara_short", 73.0, 60.0, 75.0, 90.0),
    ("west_to_east_demo", 66.0, 10.0, 78.0, 150.0),
]


@dataclass
class AISRunResult:
    route_name: str
    start: Tuple[float, float]
    end: Tuple[float, float]
    w_ais: float
    reachable: bool
    distance_km: float
    total_cost: float
    components: Dict[str, float]
    path_changed_vs_w0: bool = False
    path_coords: List[Tuple[float, float]] | None = None

    def to_row(self) -> Dict[str, str]:
        return {
            "route_name": self.route_name,
            "start_lat": f"{self.start[0]:.4f}",
            "start_lon": f"{self.start[1]:.4f}",
            "end_lat": f"{self.end[0]:.4f}",
            "end_lon": f"{self.end[1]:.4f}",
            "w_ais": f"{self.w_ais:.1f}",
            "reachable": "yes" if self.reachable else "no",
            "distance_km": f"{self.distance_km:.2f}" if self.reachable else "N/A",
            "total_cost": f"{self.total_cost:.4f}" if self.reachable else "N/A",
            "ais_cost": f"{self.components.get('ais_density', 0.0):.4f}" if self.reachable else "N/A",
            "ice_cost": f"{self.components.get('ice_risk', 0.0):.4f}" if self.reachable else "N/A",
            "wave_cost": f"{self.components.get('wave_risk', 0.0):.4f}" if self.reachable else "N/A",
            "base_distance": f"{self.components.get('base_distance', 0.0):.4f}" if self.reachable else "N/A",
            "edl_cost": f"{self.components.get('edl_risk', 0.0):.4f}" if self.reachable else "N/A",
            "path_changed": "yes" if self.path_changed_vs_w0 else "no",
        }


def run_single_case(
    grid,
    land_mask,
    route_name: str,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    w_ais: float,
) -> AISRunResult:
    routes, cost_fields, _, _, _ = plan_three_routes(
        grid,
        land_mask,
        start_lat,
        start_lon,
        end_lat,
        end_lon,
        allow_diag=True,
        vessel=None,
        cost_mode="real_sic_if_available",
        wave_penalty=0.0,
        use_edl=False,
        w_edl=0.0,
        w_ais=w_ais,
    )

    profile_key = "edl_safe"
    profile_idx = [p["key"] for p in ROUTE_PROFILES].index(profile_key)
    route_info = routes[profile_idx]

    if not route_info.reachable:
        return AISRunResult(
            route_name,
            (start_lat, start_lon),
            (end_lat, end_lon),
            w_ais,
            reachable=False,
            distance_km=0.0,
            total_cost=0.0,
            components={},
        )

    cf = cost_fields[profile_key]
    breakdown = compute_route_cost_breakdown(grid, cf, route_info.coords)

    return AISRunResult(
        route_name,
        (start_lat, start_lon),
        (end_lat, end_lon),
        w_ais,
        reachable=True,
        distance_km=compute_path_length_km(route_info.coords),
        total_cost=breakdown.total_cost,
        components=breakdown.component_totals,
        path_coords=route_info.coords,
    )


def print_summary(results: List[AISRunResult]) -> None:
    by_route: Dict[str, List[AISRunResult]] = {}
    for r in results:
        by_route.setdefault(r.route_name, []).append(r)

    for route_name, rows in by_route.items():
        print(f"\n[{route_name}]")
        print(f"{'w_ais':<6} {'Reachable':<10} {'Distance_km':<13} {'TotalCost':<12} {'AIS_Cost':<10}")
        print("-" * 60)
        for r in sorted(rows, key=lambda x: x.w_ais):
            reach = "Yes" if r.reachable else "No"
            dist = f"{r.distance_km:.1f}" if r.reachable else "N/A"
            tc = f"{r.total_cost:.3f}" if r.reachable else "N/A"
            ais = f"{r.components.get('ais_density', 0.0):.4f}" if r.reachable else "N/A"
            print(f"{r.w_ais:<6.1f} {reach:<10} {dist:<13} {tc:<12} {ais:<10}")
        print("-" * 60)


def write_csv(results: List[AISRunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "route_name",
        "start_lat",
        "start_lon",
        "end_lat",
        "end_lon",
        "w_ais",
        "reachable",
        "distance_km",
        "total_cost",
        "ais_cost",
        "ice_cost",
        "wave_cost",
        "base_distance",
        "edl_cost",
        "path_changed",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_row())
    print(f"[OK] CSV written: {path}")


def main() -> None:
    env = load_real_env_for_grid()
    if env is None:
        print("[ERROR] real environment not available; aborting.")
        return
    grid = env.grid
    land_mask = env.land_mask

    all_results: List[AISRunResult] = []

    for route_name, s_lat, s_lon, e_lat, e_lon in ROUTES:
        for w in W_AIS_VALUES:
            result = run_single_case(grid, land_mask, route_name, s_lat, s_lon, e_lat, e_lon, w)
            all_results.append(result)

        # 路径变化对比：仅在 w=0 可达时比对坐标
        w0 = next((r for r in all_results if r.route_name == route_name and r.w_ais == 0.0), None)
        if w0 and w0.reachable and w0.path_coords:
            for r in all_results:
                if r.route_name == route_name and r.w_ais != 0.0 and r.reachable and r.path_coords:
                    r.path_changed_vs_w0 = r.path_coords != w0.path_coords

    print_summary(all_results)
    write_csv(all_results, Path("reports/ais_sensitivity_results.csv"))


if __name__ == "__main__":
    main()
