from __future__ import annotations

"""
对一组固定场景，在【真实网格 + 真实环境】下做体检：
- 规划 efficient / edl_safe / edl_robust 三条路线
- 检查是否可达、是否踩陆
- 打印成本分解
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from arcticroute.core.analysis import compute_route_cost_breakdown
from arcticroute.core.astar import plan_route_latlon
from arcticroute.core.cost import build_cost_from_real_env
from arcticroute.core.env_real import load_real_env_for_grid
from arcticroute.core.landmask import evaluate_route_against_landmask


SCENARIOS: List[Tuple[str, float, float, float, float]] = [
    ("barents_to_chukchi", 70.0, 30.0, 70.5, 170.0),
    ("kara_short", 72.0, 60.0, 73.0, 90.0),
    ("laptev_long", 73.0, 90.0, 75.0, 150.0),
]

MODES: List[Tuple[str, Dict[str, Any]]] = [
    ("efficient", {"ice_penalty": 4.0, "wave_penalty": 0.0, "use_edl": False, "w_edl": 0.0}),
    ("edl_safe", {"ice_penalty": 4.0, "wave_penalty": 0.0, "use_edl": True, "w_edl": 0.09}),
    ("edl_robust", {"ice_penalty": 4.0, "wave_penalty": 0.0, "use_edl": True, "w_edl": 0.3, "use_edl_uncertainty": True, "edl_uncertainty_weight": 1.0}),
]


@dataclass
class CaseResult:
    name: str
    mode: str
    reachable: bool
    on_land_steps: int
    reason: str | None
    distance_km: float | None
    cost_total: float | None
    ice_risk: float | None
    wave_risk: float | None
    edl_risk: float | None
    edl_uncertainty: float | None


def main() -> None:
    env = load_real_env_for_grid(grid=None)
    if env is None or env.grid is None or env.sic is None or env.land_mask is None:
        print("[CHECK] 真实环境不可用，体检中止（请检查数据路径或环境变量）。")
        return

    grid = env.grid
    land_mask = env.land_mask
    results: List[CaseResult] = []

    for scenario_name, st_lat, st_lon, ed_lat, ed_lon in SCENARIOS:
        for mode_name, cfg in MODES:
            print(f"\n[CHECK] Scenario={scenario_name}, mode={mode_name}")
            try:
                cost_field = build_cost_from_real_env(
                    grid,
                    land_mask,
                    env,
                    ice_penalty=cfg.get("ice_penalty", 4.0),
                    wave_penalty=cfg.get("wave_penalty", 0.0),
                    w_edl=cfg.get("w_edl", 0.0),
                    use_edl=cfg.get("use_edl", False),
                    use_edl_uncertainty=cfg.get("use_edl_uncertainty", False),
                    edl_uncertainty_weight=cfg.get("edl_uncertainty_weight", 0.0),
                )
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL] 构建成本失败: {e}")
                results.append(
                    CaseResult(scenario_name, mode_name, False, 0, f"成本构建失败: {e}", None, None, None, None, None, None)
                )
                continue

            path = plan_route_latlon(
                cost_field,
                st_lat,
                st_lon,
                ed_lat,
                ed_lon,
                neighbor8=True,
            )

            if not path:
                print("[FAIL] 不可达")
                results.append(
                    CaseResult(scenario_name, mode_name, False, 0, "不可达", None, None, None, None, None, None)
                )
                continue

            stats = evaluate_route_against_landmask(grid, land_mask, path)
            breakdown = compute_route_cost_breakdown(grid, cost_field, path)

            reachable = stats.on_land_steps == 0
            reason = None if reachable else f"踩陆 {stats.on_land_steps} steps"
            if reachable:
                print(f"[OK] reachable, steps={len(path)}, distance≈{breakdown.component_totals.get('base_distance', 0):.2f}")
            else:
                print(f"[FAIL] {reason}")

            results.append(
                CaseResult(
                    scenario_name,
                    mode_name,
                    reachable,
                    stats.on_land_steps,
                    reason,
                    distance_km=float(breakdown.component_totals.get("base_distance", 0.0)),
                    cost_total=float(breakdown.total_cost),
                    ice_risk=float(breakdown.component_totals.get("ice_risk", 0.0)),
                    wave_risk=float(breakdown.component_totals.get("wave_risk", 0.0)),
                    edl_risk=float(breakdown.component_totals.get("edl_risk", 0.0)),
                    edl_uncertainty=float(breakdown.component_totals.get("edl_uncertainty_penalty", 0.0)),
                )
            )

    # Summary
    total = len(results)
    ok = sum(1 for r in results if r.reachable and (r.on_land_steps == 0))
    failed = total - ok

    print("\n========================= SUMMARY =========================")
    print(f"total_cases = {total}")
    print(f"ok_cases    = {ok}")
    print(f"failed_cases= {failed}")
    print("-----------------------------------------------------------")
    if failed:
        print("failed details:")
        for r in results:
            if not r.reachable or r.on_land_steps > 0:
                print(f"- {r.name} / {r.mode}: reason={r.reason}")
    print("===========================================================")


if __name__ == "__main__":
    main()
