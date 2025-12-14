#!/usr/bin/env python
"""
EDL 模式演示脚本。

在虚拟环境数据上演示三种 EDL 模式的行为。
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from arcticroute.core.grid import make_demo_grid
from arcticroute.core.cost import build_cost_from_real_env
from arcticroute.core.env_real import RealEnvLayers
from arcticroute.core.astar import plan_route_latlon
from arcticroute.core.analysis import compute_route_cost_breakdown

from scripts.run_edl_sensitivity_study import MODES


def demo_edl_modes():
    """演示三种 EDL 模式的行为。"""
    
    print("\n" + "=" * 80)
    print("EDL 模式演示")
    print("=" * 80)
    
    # 创建 demo 网格
    grid, land_mask = make_demo_grid()
    ny, nx = grid.shape()
    
    print(f"\n网格大小: {ny} x {nx}")
    
    # 创建虚拟环境数据
    # 使用一个有梯度的 SIC 场，模拟真实的冰况分布
    lat2d = grid.lat2d
    lon2d = grid.lon2d
    
    # SIC: 纬度越高，冰况越浓
    sic = np.clip((lat2d - 65.0) / 20.0, 0.0, 1.0)
    
    # Wave: 简单常数
    wave_swh = np.full((ny, nx), 2.0, dtype=float)
    
    # Ice thickness: 与 SIC 相关
    ice_thickness_m = sic * 1.5
    
    env = RealEnvLayers(
        sic=sic,
        wave_swh=wave_swh,
        ice_thickness_m=ice_thickness_m,
    )
    
    print(f"\n虚拟环境数据:")
    print(f"  SIC 范围: [{sic.min():.3f}, {sic.max():.3f}]")
    print(f"  Wave SWH: {wave_swh[0, 0]:.1f} m")
    print(f"  Ice thickness 范围: [{ice_thickness_m.min():.3f}, {ice_thickness_m.max():.3f}] m")
    
    # 定义起终点
    start_lat, start_lon = 66.0, 5.0
    end_lat, end_lon = 78.0, 150.0
    
    print(f"\n起点: ({start_lat:.1f}°N, {start_lon:.1f}°E)")
    print(f"终点: ({end_lat:.1f}°N, {end_lon:.1f}°E)")
    
    # 对三种模式进行规划
    print("\n" + "-" * 80)
    print("三种模式的规划结果:")
    print("-" * 80)
    
    results = {}
    
    for mode_name in ["efficient", "edl_safe", "edl_robust"]:
        cfg = MODES[mode_name]
        
        print(f"\n【{mode_name}】{cfg['description']}")
        print(f"  配置: w_edl={cfg['w_edl']}, use_edl={cfg['use_edl']}, use_edl_uncertainty={cfg['use_edl_uncertainty']}")
        
        # 构建成本场
        cost_field = build_cost_from_real_env(
            grid=grid,
            land_mask=land_mask,
            env=env,
            ice_penalty=cfg["ice_penalty"],
            wave_penalty=0.0,
            vessel_profile=None,
            w_edl=cfg["w_edl"],
            use_edl=cfg["use_edl"],
            use_edl_uncertainty=cfg["use_edl_uncertainty"],
            edl_uncertainty_weight=cfg["edl_uncertainty_weight"],
        )
        
        # 规划路线
        route = plan_route_latlon(
            cost_field,
            start_lat,
            start_lon,
            end_lat,
            end_lon,
            neighbor8=True,
        )
        
        if route:
            # 计算成本分解
            breakdown = compute_route_cost_breakdown(grid, cost_field, route)
            
            # 提取成本分量
            edl_risk = breakdown.component_totals.get("edl_risk", 0.0)
            edl_uncertainty = breakdown.component_totals.get("edl_uncertainty_penalty", 0.0)
            ice_risk = breakdown.component_totals.get("ice_risk", 0.0)
            base_distance = breakdown.component_totals.get("base_distance", 0.0)
            
            results[mode_name] = {
                "reachable": True,
                "total_cost": breakdown.total_cost,
                "edl_risk": edl_risk,
                "edl_uncertainty": edl_uncertainty,
                "ice_risk": ice_risk,
                "base_distance": base_distance,
                "distance_km": len(route) * 0.1,  # 粗略估计
            }
            
            print(f"  ✓ 可达")
            print(f"    总成本: {breakdown.total_cost:.4f}")
            print(f"    - 基础距离: {base_distance:.4f}")
            print(f"    - 冰风险: {ice_risk:.4f}")
            print(f"    - EDL 风险: {edl_risk:.4f}")
            print(f"    - EDL 不确定性: {edl_uncertainty:.4f}")
            
            # 计算占比
            if breakdown.total_cost > 0:
                edl_total = edl_risk + edl_uncertainty
                edl_fraction = edl_total / breakdown.total_cost
                print(f"    - EDL 总占比: {edl_fraction*100:.1f}%")
        else:
            print(f"  ✗ 不可达")
            results[mode_name] = {"reachable": False}
    
    # 对比分析
    print("\n" + "-" * 80)
    print("对比分析:")
    print("-" * 80)
    
    reachable_modes = [m for m, r in results.items() if r.get("reachable", False)]
    
    if len(reachable_modes) >= 2:
        # 比较 EDL 成本
        print("\nEDL 风险成本对比:")
        edl_costs = {m: results[m].get("edl_risk", 0.0) for m in reachable_modes}
        
        for mode in ["efficient", "edl_safe", "edl_robust"]:
            if mode in edl_costs:
                cost = edl_costs[mode]
                print(f"  {mode:12s}: {cost:.4f}")
        
        # 检查相对关系
        if "efficient" in edl_costs and "edl_safe" in edl_costs:
            efficient_edl = edl_costs["efficient"]
            safe_edl = edl_costs["edl_safe"]
            
            if efficient_edl > 0 and safe_edl > 0:
                ratio = efficient_edl / safe_edl
                print(f"\n  efficient / edl_safe = {ratio:.2f}")
                
                if ratio < 0.5:
                    print(f"  ✓ 符合预期：efficient 的 EDL 成本约为 safe 的 {ratio*100:.0f}%")
                else:
                    print(f"  ⚠ 注意：比例 {ratio:.2f} 可能不符合预期")
        
        # 检查不确定性成本
        print("\nEDL 不确定性成本对比:")
        for mode in ["efficient", "edl_safe", "edl_robust"]:
            if mode in results and results[mode].get("reachable", False):
                unc = results[mode].get("edl_uncertainty", 0.0)
                print(f"  {mode:12s}: {unc:.4f}")
        
        if "edl_robust" in results and results["edl_robust"].get("reachable", False):
            robust_unc = results["edl_robust"].get("edl_uncertainty", 0.0)
            if robust_unc > 0:
                print(f"\n  ✓ edl_robust 有不确定性成本（{robust_unc:.4f}）")
            else:
                print(f"\n  ⚠ edl_robust 没有不确定性成本（可能是虚拟数据中不确定性较低）")
    
    print("\n" + "=" * 80)
    print("演示完成")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    demo_edl_modes()









