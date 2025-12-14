#!/usr/bin/env python3
"""
Phase 1.5 Step A：CLI 验证脚本 - 验证 AIS 密度对路径和成本的实际影响

目标：
1. 使用真实网格（如果可用，否则用 demo 网格）
2. 对同一起终点，跑 3 组规划：w_ais = 0.0, 1.0, 3.0
3. 打印成本分解和路径信息
4. 观察 AIS 权重变化对成本和路径的影响

使用方式：
    python -m scripts.debug_ais_effect
"""

import sys
from pathlib import Path
import numpy as np
import logging

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from arcticroute.core.grid import make_demo_grid, load_real_grid_from_nc
from arcticroute.core.landmask import load_real_landmask_from_nc
from arcticroute.core.cost import build_cost_from_real_env, build_demo_cost
from arcticroute.core.env_real import load_real_env_for_grid
from arcticroute.core.astar import plan_route_latlon
from arcticroute.core.analysis import compute_route_cost_breakdown
from arcticroute.core.ais_ingest import build_ais_density_da_for_demo_grid, AIS_RAW_DIR

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算两点间的大圆距离（单位：km）"""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (
        np.sin(dphi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def compute_path_length_km(path: list[tuple[float, float]]) -> float:
    """计算路径的总长度（单位：km）"""
    if len(path) < 2:
        return 0.0
    total_dist = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(path[:-1], path[1:]):
        total_dist += haversine_km(lat1, lon1, lat2, lon2)
    return total_dist


def run_ais_effect_test():
    """运行 AIS 效果验证测试"""
    
    print("\n" + "="*80)
    print("Phase 1.5 Step A: AIS 密度效果验证")
    print("="*80 + "\n")
    
    # ========================================================================
    # 1. 加载网格和陆地掩码
    # ========================================================================
    print("[1/5] 加载网格和陆地掩码...")
    
    grid_source = "demo"
    real_grid = load_real_grid_from_nc()
    if real_grid is not None:
        grid = real_grid
        land_mask = load_real_landmask_from_nc(grid)
        if land_mask is None:
            logger.warning("真实 landmask 不可用，使用 demo landmask")
            _, land_mask = make_demo_grid(ny=grid.shape()[0], nx=grid.shape()[1])
            grid_source = "real_grid_demo_landmask"
        else:
            grid_source = "real"
    else:
        logger.warning("真实网格不可用，使用 demo 网格")
        grid, land_mask = make_demo_grid()
        grid_source = "demo"
    
    print(f"  [OK] 网格来源: {grid_source}")
    print(f"  [OK] 网格大小: {grid.shape()}")
    print(f"  [OK] 陆地掩码大小: {land_mask.shape}")
    
    # ========================================================================
    # 2. 定义测试起终点
    # ========================================================================
    print("\n[2/5] 定义测试起终点...")
    
    # 根据网格范围调整起终点
    # demo 网格范围: lat [60, 85], lon [-180, 180]
    if grid_source == "demo":
        start_lat, start_lon = 75.0, -30.0   # Barents Sea
        end_lat, end_lon = 72.0, 100.0       # Siberia
    else:
        # 真实网格范围更大
        start_lat, start_lon = 75.0, 30.0    # Barents Sea
        end_lat, end_lon = 70.0, 170.0       # Chukchi Sea
    
    print(f"  [OK] 起点: ({start_lat:.1f}N, {start_lon:.1f}E)")
    print(f"  [OK] 终点: ({end_lat:.1f}N, {end_lon:.1f}E)")
    
    # ========================================================================
    # 3. 加载 AIS 密度数据（从原始目录）
    # ========================================================================
    print("\n[3/5] 加载 AIS 密度数据...")
    
    ais_density = None
    
    if AIS_RAW_DIR.is_dir():
        try:
            # 从原始 AIS 目录构建密度
            ais_da = build_ais_density_da_for_demo_grid(
                AIS_RAW_DIR,
                grid.lat2d[:, 0],  # 1D 纬度
                grid.lon2d[0, :],  # 1D 经度
            )
            ais_density = ais_da.values
            print(f"  [OK] AIS 数据已加载（网格大小: {ais_density.shape}）")
            print(f"    - 密度范围: {ais_da.min().item():.4f} ~ {ais_da.max().item():.4f}")
        except Exception as e:
            logger.warning(f"加载 AIS 数据失败: {e}，将使用零密度")
            ais_density = np.zeros_like(grid.lat2d)
    else:
        logger.warning(f"AIS 原始数据目录不存在: {AIS_RAW_DIR}，将使用零密度")
        ais_density = np.zeros_like(grid.lat2d)
    
    # ========================================================================
    # 4. 尝试加载真实环境数据（仅在使用真实网格时）
    # ========================================================================
    print("\n[4/5] 加载真实环境数据...")
    
    real_env = None
    if grid_source != "demo":
        try:
            real_env = load_real_env_for_grid(grid)
            if real_env is not None and (real_env.sic is not None or real_env.wave_swh is not None):
                print(f"  [OK] 真实环境数据已加载")
                if real_env.sic is not None:
                    print(f"    - 海冰浓度范围: {real_env.sic.min():.3f} ~ {real_env.sic.max():.3f}")
                if real_env.wave_swh is not None:
                    print(f"    - 波浪高度范围: {real_env.wave_swh.min():.3f} ~ {real_env.wave_swh.max():.3f}")
            else:
                logger.warning("真实环境数据不可用，将使用 demo 成本")
                real_env = None
        except Exception as e:
            logger.warning(f"加载真实环境数据失败: {e}，将使用 demo 成本")
            real_env = None
    else:
        print(f"  [INFO] 使用 demo 网格，跳过真实环境数据加载")
    
    # ========================================================================
    # 5. 对三个 AIS 权重值进行规划和成本计算
    # ========================================================================
    print("\n[5/5] 规划路线并计算成本...\n")
    
    ais_weights = [0.0, 1.0, 3.0]
    results = []
    
    for w_ais in ais_weights:
        print(f"\n{'='*80}")
        print(f"规划方案: w_ais = {w_ais:.1f}")
        print(f"{'='*80}")
        
        # 构建成本场
        try:
            if real_env is not None and (real_env.sic is not None or real_env.wave_swh is not None):
                cost_field = build_cost_from_real_env(
                    grid,
                    land_mask,
                    real_env,
                    ice_penalty=4.0,
                    wave_penalty=0.0,
                    ais_density=ais_density,
                    ais_weight=w_ais,
                )
            else:
                # 使用 demo 成本（支持 AIS）
                cost_field = build_demo_cost(grid, land_mask, ice_penalty=4.0)
                # 手动添加 AIS 成本到 demo 成本
                if w_ais > 0 and ais_density is not None:
                    ais_cost = w_ais * np.clip(ais_density, 0.0, 1.0)
                    cost_field.cost = cost_field.cost + ais_cost
                    if cost_field.components is None:
                        cost_field.components = {}
                    cost_field.components["ais_density"] = ais_cost
        except Exception as e:
            logger.error(f"构建成本场失败: {e}")
            continue
        
        # 规划路线
        try:
            route_coords = plan_route_latlon(
                cost_field,
                start_lat,
                start_lon,
                end_lat,
                end_lon,
                neighbor8=True,
            )
            route_steps = len(route_coords) if route_coords else None
        except Exception as e:
            logger.error(f"规划路线失败: {e}")
            route_coords = None
            route_steps = None
        
        # 计算路线信息
        reachable = route_coords is not None and len(route_coords) > 0
        
        if reachable:
            path_length_km = compute_path_length_km(route_coords)
            
            # 计算成本分解
            breakdown = compute_route_cost_breakdown(cost_field.grid, cost_field, route_coords)
            total_cost = breakdown.total_cost
            
            print(f"  [OK] 路线可达")
            print(f"    - 路径点数: {len(route_coords)}")
            print(f"    - 路径长度: {path_length_km:.1f} km")
            print(f"    - 总成本: {total_cost:.2f}")
            print(f"    - 起点: ({route_coords[0][0]:.2f}N, {route_coords[0][1]:.2f}E)")
            print(f"    - 终点: ({route_coords[-1][0]:.2f}N, {route_coords[-1][1]:.2f}E)")
            
            # 打印成本分解
            print(f"\n  成本分解:")
            if breakdown.component_totals:
                for comp_name in sorted(breakdown.component_totals.keys()):
                    comp_value = breakdown.component_totals[comp_name]
                    comp_frac = breakdown.component_fractions.get(comp_name, 0.0)
                    print(f"    - {comp_name:20s}: {comp_value:10.2f} ({comp_frac:6.2%})")
            else:
                print(f"    (无分解信息)")
            
            # 特别关注 AIS 成本
            if "ais_density" in breakdown.component_totals:
                ais_cost = breakdown.component_totals["ais_density"]
                ais_frac = breakdown.component_fractions.get("ais_density", 0.0)
                print(f"\n  [AIS] AIS 拥挤风险成本: {ais_cost:.2f} ({ais_frac:.2%})")
            else:
                print(f"\n  [AIS] AIS 拥挤风险成本: 0.00 (0.00%)")
            
            results.append({
                "w_ais": w_ais,
                "reachable": True,
                "path_length_km": path_length_km,
                "total_cost": total_cost,
                "num_steps": len(route_coords),
                "ais_cost": breakdown.component_totals.get("ais_density", 0.0),
                "breakdown": breakdown.component_totals.copy(),
            })
        else:
            print(f"  [FAIL] 路线不可达")
            results.append({
                "w_ais": w_ais,
                "reachable": False,
                "path_length_km": None,
                "total_cost": None,
                "num_steps": None,
                "ais_cost": None,
                "breakdown": {},
            })
    
    # ========================================================================
    # 6. 总结和观察
    # ========================================================================
    print(f"\n\n{'='*80}")
    print("总结与观察")
    print(f"{'='*80}\n")
    
    # 检查所有方案是否可达
    reachable_results = [r for r in results if r["reachable"]]
    if len(reachable_results) < len(results):
        print("[WARN] 警告: 部分方案不可达，无法进行对比")
    
    if len(reachable_results) >= 2:
        # 对比成本变化
        print("成本变化对比:")
        for i in range(len(reachable_results) - 1):
            r1 = reachable_results[i]
            r2 = reachable_results[i + 1]
            cost_delta = r2["total_cost"] - r1["total_cost"]
            cost_delta_pct = (cost_delta / r1["total_cost"] * 100) if r1["total_cost"] > 0 else 0
            ais_delta = r2["ais_cost"] - r1["ais_cost"]
            
            print(f"\n  w_ais: {r1['w_ais']:.1f} → {r2['w_ais']:.1f}")
            print(f"    - 总成本变化: {cost_delta:+.2f} ({cost_delta_pct:+.1f}%)")
            print(f"    - AIS 成本变化: {ais_delta:+.2f}")
            print(f"    - 路径长度变化: {r2['path_length_km'] - r1['path_length_km']:+.1f} km")
        
        # 检查 AIS 成本单调性
        ais_costs = [r["ais_cost"] for r in reachable_results]
        is_monotonic = all(ais_costs[i] <= ais_costs[i+1] for i in range(len(ais_costs)-1))
        
        print(f"\n[CHECK] AIS 成本单调性检查: {'通过' if is_monotonic else '失败'}")
        print(f"  AIS 成本序列: {[f'{c:.2f}' for c in ais_costs]}")
        
        # 检查总成本是否增加
        total_costs = [r["total_cost"] for r in reachable_results]
        total_cost_increasing = all(total_costs[i] <= total_costs[i+1] for i in range(len(total_costs)-1))
        
        print(f"\n[CHECK] 总成本单调性检查: {'通过' if total_cost_increasing else '失败'}")
        print(f"  总成本序列: {[f'{c:.2f}' for c in total_costs]}")
    
    print(f"\n{'='*80}")
    print("验证完成！")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    run_ais_effect_test()

