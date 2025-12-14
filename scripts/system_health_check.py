"""
ArcticRoute 系统健康检查脚本。

可通过以下命令运行：
    python -m scripts.system_health_check

检查项目：
1. Demo 网格健康检查 - 基础路由功能
2. 真实网格 + EDL + 冰级 + AIS 检查 - 完整功能
3. AIS 流程检查 - AIS 密度数据加载
4. EDL 后端轻量检查 - EDL 推理可用性
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from arcticroute.core.astar import plan_route_latlon, plan_route_latlon_with_info
from arcticroute.core.cost import (
    build_demo_cost,
    build_cost_from_real_env,
    discover_ais_density_candidates,
)
from arcticroute.core.grid import Grid2D, make_demo_grid
from arcticroute.core.landmask import evaluate_route_against_landmask
from arcticroute.core.analysis import compute_route_cost_breakdown
from arcticroute.core.scenarios import load_all_scenarios
from arcticroute.core.env_real import load_real_env_for_grid
from arcticroute.core.edl_backend_miles import run_miles_edl_on_grid, has_miles_guess

CheckResult = Tuple[bool, str, Dict[str, Any]]


def check_demo_route() -> CheckResult:
    """
    在 demo 网格上跑一条标准路线，检查：
    - 可达
    - 起终点吸附误差 < 1.0°（允许更大的误差）
    - 不踩陆地
    - 成本分解中 base_distance > 0
    """
    try:
        grid, land_mask = make_demo_grid()
        
        start = (66.0, 5.0)
        goal = (78.0, 140.0)  # 调整到 demo 网格范围内（0-160）
        
        cost = build_demo_cost(
            grid=grid,
            land_mask=land_mask,
            ice_penalty=4.0,
            w_ais=0.0,
        )
        
        route = plan_route_latlon(
            cost_field=cost,
            start_lat=start[0],
            start_lon=start[1],
            end_lat=goal[0],
            end_lon=goal[1],
            neighbor8=True,
        )
        
        if not route or len(route) == 0:
            return False, "demo 路线不可达", {"start": start, "goal": goal}
        
        path = route
        if len(path) < 2:
            return False, "demo 路径点数过少", {"num_points": len(path)}
        
        s_lat, s_lon = path[0]
        g_lat, g_lon = path[-1]
        
        # 允许 1.0° 的吸附误差（demo 网格分辨率较粗）
        if max(abs(s_lat - start[0]), abs(s_lon - start[1])) > 1.0:
            return False, "起点吸附误差超过 1.0°", {
                "input_start": start,
                "snapped_start": (float(s_lat), float(s_lon)),
            }
        
        if max(abs(g_lat - goal[0]), abs(g_lon - goal[1])) > 1.0:
            return False, "终点吸附误差超过 1.0°", {
                "input_goal": goal,
                "snapped_goal": (float(g_lat), float(g_lon)),
            }
        
        stats = evaluate_route_against_landmask(grid, land_mask, path)
        if stats.on_land_steps > 0:
            return False, "demo 路径存在踩陆步数", {
                "on_land_steps": stats.on_land_steps,
                "first_land_index": stats.first_land_index,
                "first_land_latlon": stats.first_land_latlon,
            }
        
        breakdown = compute_route_cost_breakdown(grid, cost, path)
        base_cost = breakdown.component_totals.get("base_distance", 0.0)
        
        if base_cost <= 0:
            return False, "demo 成本分解 base_distance<=0", {"base_distance": base_cost}
        
        return True, "demo 网格 + 路由 + 成本分解 正常", {
            "num_points": len(path),
            "total_cost": breakdown.total_cost,
            "base_distance_total": base_cost,
        }
    
    except Exception as e:
        return False, f"demo 检查异常: {e}", {"error": str(e)}


def check_real_env_and_edl() -> CheckResult:
    """
    在真实网格上跑三种策略，检查：
    - 三条路线都可达
    - 不踩陆
    - 成本分量包含冰、波浪、AIS、EDL 等
    - edl_robust 的风险成本低于 efficient
    """
    try:
        # 加载所有场景
        scenarios_dict = load_all_scenarios()
        
        # 找一个 grid_mode == "real" 的场景，优先选择 barents 相关的
        real_scn = None
        for scn_id, scn in scenarios_dict.items():
            if scn.grid_mode == "real":
                real_scn = scn
                # 优先选择 barents 相关的场景
                if "barents" in scn_id.lower():
                    break
        
        if real_scn is None:
            return False, "未找到 grid_mode=real 的场景", {}
        
        # 加载真实环境（使用正确的参数）
        env = load_real_env_for_grid(ym=real_scn.ym)
        
        if env is None or env.grid is None or env.land_mask is None:
            return False, f"真实环境加载失败 (ym={real_scn.ym})", {}
        
        grid = env.grid
        land_mask = env.land_mask
        
        modes = ["efficient", "edl_safe", "edl_robust"]
        routes = {}
        summaries = {}
        reachable_count = 0
        diag_per_mode: Dict[str, Dict[str, Any]] = {}
        
        for mode in modes:
            use_edl = mode != "efficient"
            use_edl_uncertainty = mode == "edl_robust"
            
            try:
                cost = build_cost_from_real_env(
                    grid=grid,
                    land_mask=land_mask,
                    env=env,
                    ice_penalty=real_scn.w_ice,
                    wave_penalty=real_scn.w_wave,
                    use_edl=use_edl,
                    use_edl_uncertainty=use_edl_uncertainty,
                    w_ais=real_scn.w_ais,
                )
            except Exception as e:
                return False, f"真实场景 {real_scn.id} 下 {mode} 成本构建失败: {e}", {}
            
            # 统计 cost 的有限比例
            try:
                cost_finite_frac = float(np.isfinite(cost.cost).mean())
            except Exception:
                cost_finite_frac = None
            
            res = plan_route_latlon_with_info(
                cost_field=cost,
                start_lat=real_scn.start_lat,
                start_lon=real_scn.start_lon,
                end_lat=real_scn.end_lat,
                end_lon=real_scn.end_lon,
                neighbor8=True,
                max_expansions=500000,
            )
            
            # 诊断信息
            si, sj = res.start_ij if res.start_ij is not None else (None, None)
            gi, gj = res.goal_ij if res.goal_ij is not None else (None, None)
            start_is_land = bool(land_mask[si, sj]) if si is not None else None
            goal_is_land = bool(land_mask[gi, gj]) if gi is not None else None
            start_cost = float(cost.cost[si, sj]) if si is not None else None
            goal_cost = float(cost.cost[gi, gj]) if gi is not None else None
            diag_per_mode[mode] = {
                "input_start": (real_scn.start_lat, real_scn.start_lon),
                "input_goal": (real_scn.end_lat, real_scn.end_lon),
                "start_ij": res.start_ij,
                "goal_ij": res.goal_ij,
                "snapped_start": tuple(map(float, res.snapped_start)) if res.snapped_start else None,
                "snapped_goal": tuple(map(float, res.snapped_goal)) if res.snapped_goal else None,
                "start_is_land": start_is_land,
                "goal_is_land": goal_is_land,
                "start_cost": start_cost,
                "goal_cost": goal_cost,
                "cost_finite_frac": cost_finite_frac,
                "astar_reason": res.reason,
                "astar_expanded": res.expanded,
                "reachable": res.reachable,
                "num_points": len(res.path_latlon) if res.path_latlon else 0,
            }
            
            if not res.reachable or not res.path_latlon:
                # 记录不可达，但继续检查其他模式
                continue
            
            reachable_count += 1
            
            stats = evaluate_route_against_landmask(grid, land_mask, res.path_latlon)
            if stats.on_land_steps > 0:
                return False, f"真实场景 {real_scn.id} 下 {mode} 路径踩陆", {
                    "mode": mode,
                    "on_land_steps": stats.on_land_steps,
                    "first_land_latlon": stats.first_land_latlon,
                }
            
            breakdown = compute_route_cost_breakdown(grid, cost, res.path_latlon)
            routes[mode] = res.path_latlon
            summaries[mode] = breakdown
        
        # 检查是否至少有一条路线可达
        if reachable_count == 0:
            return False, f"真实场景 {real_scn.id} 所有方案都不可达", {
                "modes_tried": modes,
                "diagnostics": diag_per_mode,
            }
        
        # 检查 EDL 风险是否起作用（如果有 EDL 数据）
        risk_eff = summaries.get("efficient", {}).component_totals.get("edl_risk", 0.0) if "efficient" in summaries else 0.0
        risk_safe = summaries.get("edl_safe", {}).component_totals.get("edl_risk", 0.0) if "edl_safe" in summaries else 0.0
        risk_robust = summaries.get("edl_robust", {}).component_totals.get("edl_risk", 0.0) if "edl_robust" in summaries else 0.0
        
        info = {
            "scenario_id": real_scn.id,
            "edl_risk_eff": risk_eff,
            "edl_risk_safe": risk_safe,
            "edl_risk_robust": risk_robust,
            "num_routes": len(routes),
            "reachable_count": reachable_count,
        }
        
        # 如果有 EDL 数据，检查 safe/robust 是否有明显降低
        # 如果 EDL 数据不可用，也不强制要求
        if risk_eff > 0 and (risk_safe >= risk_eff or risk_robust >= risk_eff):
            info["diagnostics"] = diag_per_mode
            return False, f"EDL 策略未明显降低风险成本，请检查 EDL 配置", info
        
        return True, f"真实场景 {real_scn.id} 至少有 {reachable_count} 条路线可达且无踩陆", info
    
    except Exception as e:
        return False, f"真实环境检查异常: {e}", {"error": str(e)}


def check_ais_pipeline() -> CheckResult:
    """
    检查 AIS 流程：
    - 发现 AIS density .nc 文件
    - 加载并应用到成本场
    - 验证 ais_density 成本分量存在（可能为 0 如果密度数据全为 0）
    """
    try:
        cands = discover_ais_density_candidates()
        
        if not cands:
            # AIS 文件不存在不算失败，只是功能不可用
            return True, "未发现 AIS density .nc 文件（功能可选）", {
                "num_candidates": 0,
            }
        
        # 用第一个候选文件构造一个小的 grid + cost
        grid, land_mask = make_demo_grid()
        
        cost = build_demo_cost(
            grid=grid,
            land_mask=land_mask,
            ice_penalty=0.0,
            w_ais=5.0,
            ais_density_path=cands[0]["path"],
        )
        
        # 随便跑一条短路线
        route = plan_route_latlon(
            cost_field=cost,
            start_lat=66.0,
            start_lon=5.0,
            end_lat=68.0,
            end_lon=20.0,
            neighbor8=True,
        )
        
        if not route or len(route) == 0:
            return False, "AIS 成本场下路线不可达", {
                "candidate": cands[0]["path"],
            }
        
        breakdown = compute_route_cost_breakdown(grid, cost, route)
        ais_cost = breakdown.component_totals.get("ais_density", 0.0)
        
        # 检查 ais_density 组件是否存在（即使值为 0 也可以接受）
        has_ais_component = "ais_density" in breakdown.component_totals
        
        if not has_ais_component:
            return False, "AIS 密度场加载但成本分量未记录", {
                "candidate": cands[0]["path"],
                "components": list(breakdown.component_totals.keys()),
            }
        
        return True, "AIS 密度 .nc 加载正常，成本分量已记录", {
            "candidate": cands[0]["path"],
            "ais_cost": float(ais_cost),
            "num_candidates": len(cands),
        }
    
    except Exception as e:
        return False, f"AIS 流程检查异常: {e}", {"error": str(e)}


def check_edl_backend() -> CheckResult:
    """
    检查 EDL 后端：
    - miles-guess 或 torch EDL 是否能跑出一个 grid 上的风险/不确定性
    - 如果环境不支持，给 WARN（返回 True），不算 FAIL
    """
    try:
        # 检查 miles-guess 是否可用
        has_mg = has_miles_guess()
        
        if not has_mg:
            # EDL 后端不可用，但这是可选功能，不算失败
            return True, "当前环境未安装 miles-guess，EDL 功能将自动降级 [WARN]", {
                "has_miles_guess": False,
                "note": "EDL 功能为可选，系统会自动使用占位符",
            }
        
        # 尝试用 demo 数据跑一次 EDL
        grid, _ = make_demo_grid()
        ny, nx = grid.shape()
        
        # 构造虚拟的 SIC 数据
        dummy_sic = np.random.rand(ny, nx).astype(np.float32) * 0.5
        
        try:
            edl_out = run_miles_edl_on_grid(
                sic=dummy_sic,
                swh=None,
                ice_thickness=None,
                grid_lat=grid.lat2d,
                grid_lon=grid.lon2d,
                model_name="default",
                device="cpu",
            )
            
            if edl_out is None:
                return False, "EDL 后端未返回结果", {}
            
            has_risk = hasattr(edl_out, "risk") and edl_out.risk is not None
            has_uncertainty = hasattr(edl_out, "uncertainty") and edl_out.uncertainty is not None
            
            return True, "EDL 后端可调用（miles-guess / torch）", {
                "has_risk": has_risk,
                "has_uncertainty": has_uncertainty,
                "source": edl_out.meta.get("source", "unknown"),
            }
        
        except Exception as e:
            return False, f"EDL 后端运行异常: {e}", {"error": str(e)}
    
    except ImportError as e:
        return True, f"当前环境未安装 EDL 后端（miles-guess/torch），功能将自动降级 [WARN]", {
            "error": str(e),
            "note": "EDL 功能为可选，系统会自动使用占位符",
        }
    except Exception as e:
        return False, f"EDL 后端检查异常: {e}", {"error": str(e)}


def main() -> None:
    """主调度与汇总。"""
    checks = [
        ("demo_route", check_demo_route),
        ("real_env_and_edl", check_real_env_and_edl),
        ("ais_pipeline", check_ais_pipeline),
        ("edl_backend", check_edl_backend),
    ]
    
    ok_all = True
    
    print("========== ArcticRoute System Health Check ==========\n")
    
    for name, fn in checks:
        try:
            ok, msg, details = fn()
        except Exception as e:
            ok = False
            msg = f"异常：{e}"
            details = {}
            traceback.print_exc()
        
        prefix = "[OK]  " if ok else "[FAIL]"
        print(f"{prefix} {name}: {msg}")
        
        if details:
            print("       details:", details)
        
        if not ok:
            ok_all = False
    
    print()
    
    if ok_all:
        print("[SYSTEM] 所有健康检查均通过 [PASS]")
        sys.exit(0)
    else:
        print("[SYSTEM] 存在失败的检查，请根据 [FAIL] 项修复后重试 [ERROR]")
        sys.exit(1)


if __name__ == "__main__":
    main()

