"""
轻量级 EDL 真实数据检查脚本。

目标：验证"接入 data_real 下的真实 nc 数据 + miles-guess EDL 成本"是否真正生效。

执行方式：
    python -m scripts.check_real_edl_task

输出：
    - [ENV] 环境数据加载信息
    - [COST] 成本场构建信息
    - CHECK_REAL_EDL_OK 或 CHECK_REAL_EDL_FAIL: reason=...
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# 导入必要的核心模块
from arcticroute.core.env_real import load_real_env_for_grid, load_real_grid_from_data_real
from arcticroute.core.cost import build_cost_from_real_env
from arcticroute.core.landmask import load_real_landmask_from_nc
from arcticroute.core.analysis import compute_route_cost_breakdown


# ============================================================================
# 配置常量（易改）
# ============================================================================

# 真实数据年月
YM = "202412"

# 真实数据文件路径（相对于项目根目录）
DATA_REAL_DIR = Path(__file__).resolve().parents[1] / "data_real" / YM

# 文件名常量
SIC_FILE = DATA_REAL_DIR / f"sic_{YM}.nc"
WAVE_FILE = DATA_REAL_DIR / f"wave_{YM}.nc"
ICE_THICKNESS_FILE = DATA_REAL_DIR / f"ice_thickness_{YM}.nc"
LANDMASK_FILE = DATA_REAL_DIR / "land_mask_gebco.nc"

# 成本构建参数
ICE_PENALTY = 4.0
WAVE_PENALTY = 1.0
W_EDL = 2.0
EDL_UNCERTAINTY_WEIGHT = 2.0

# 简单路径参数（虚拟对角线路径）
SIMPLE_PATH_POINTS = 20


# ============================================================================
# 辅助函数
# ============================================================================

def create_simple_path(grid, num_points: int = 20):
    """
    创建一条简单的虚拟路径（沿对角线）。
    
    Args:
        grid: Grid2D 对象
        num_points: 路径点数
    
    Returns:
        [(lat, lon), ...] 路径列表
    """
    ny, nx = grid.shape()
    
    # 取网格的四个角
    lat_min = grid.lat2d.min()
    lat_max = grid.lat2d.max()
    lon_min = grid.lon2d.min()
    lon_max = grid.lon2d.max()
    
    # 沿对角线生成路径
    path = []
    for i in range(num_points):
        t = i / max(1, num_points - 1)
        lat = lat_min + t * (lat_max - lat_min)
        lon = lon_min + t * (lon_max - lon_min)
        path.append((lat, lon))
    
    return path


# ============================================================================
# 主检查逻辑
# ============================================================================

def main() -> None:
    """主函数：执行 EDL 真实数据检查。"""
    
    print("=" * 70)
    print("EDL 真实数据检查脚本")
    print("=" * 70)
    print()
    
    # ========================================================================
    # Step 1: 加载真实网格和环境数据
    # ========================================================================
    print("[STEP 1] 加载真实网格和环境数据...")
    print()
    
    # 从真实数据文件加载网格
    grid = load_real_grid_from_data_real(YM)
    if grid is None:
        print("CHECK_REAL_EDL_FAIL: reason=failed_to_load_real_grid")
        return
    
    print(f"[GRID] shape={grid.shape()}, lat_range=[{grid.lat2d.min():.2f}, {grid.lat2d.max():.2f}], "
          f"lon_range=[{grid.lon2d.min():.2f}, {grid.lon2d.max():.2f}]")
    print()
    
    # 加载真实环境数据（sic + wave + ice_thickness）
    real_env = load_real_env_for_grid(grid, ym=YM)
    if real_env is None:
        print("CHECK_REAL_EDL_FAIL: reason=failed_to_load_real_env")
        return
    
    # 检查 sic 和 wave 数据有效性
    if real_env.sic is None:
        print("CHECK_REAL_EDL_FAIL: reason=sic_is_none")
        return
    
    if real_env.wave_swh is None:
        print("CHECK_REAL_EDL_FAIL: reason=wave_swh_is_none")
        return
    
    # 统计 sic 和 wave 数据
    sic_min = float(np.nanmin(real_env.sic))
    sic_max = float(np.nanmax(real_env.sic))
    sic_mean = float(np.nanmean(real_env.sic))
    sic_has_nan = bool(np.any(np.isnan(real_env.sic)))
    
    wave_min = float(np.nanmin(real_env.wave_swh))
    wave_max = float(np.nanmax(real_env.wave_swh))
    wave_mean = float(np.nanmean(real_env.wave_swh))
    wave_has_nan = bool(np.any(np.isnan(real_env.wave_swh)))
    
    print(f"[ENV] sic: min={sic_min:.4f}, max={sic_max:.4f}, mean={sic_mean:.4f}, has_nan={sic_has_nan}")
    print(f"[ENV] wave: min={wave_min:.4f}, max={wave_max:.4f}, mean={wave_mean:.4f}, has_nan={wave_has_nan}")
    
    # 检查数据是否全为 0 或全相等
    if sic_min >= sic_max:
        print("CHECK_REAL_EDL_FAIL: reason=sic_all_equal_or_zero")
        return
    
    if wave_min >= wave_max:
        print("CHECK_REAL_EDL_FAIL: reason=wave_all_equal_or_zero")
        return
    
    print()
    
    # ========================================================================
    # Step 2: 加载陆地掩码
    # ========================================================================
    print("[STEP 2] 加载陆地掩码...")
    print()
    
    land_mask = load_real_landmask_from_nc(grid, nc_path=LANDMASK_FILE)
    if land_mask is None:
        print("[WARN] landmask not available, using default (all ocean)")
        land_mask = np.zeros(grid.shape(), dtype=bool)
    else:
        ocean_count = np.sum(~land_mask)
        print(f"[LANDMASK] ocean_cells={ocean_count}, land_cells={np.sum(land_mask)}")
    
    print()
    
    # ========================================================================
    # Step 3: 构建真实环境成本场（启用 EDL）
    # ========================================================================
    print("[STEP 3] 构建真实环境成本场（启用 EDL）...")
    print()
    
    cost_field = build_cost_from_real_env(
        grid=grid,
        land_mask=land_mask,
        env=real_env,
        ice_penalty=ICE_PENALTY,
        wave_penalty=WAVE_PENALTY,
        vessel_profile=None,  # 不使用冰级约束
        w_edl=W_EDL,
        use_edl=True,
        use_edl_uncertainty=True,
        edl_uncertainty_weight=EDL_UNCERTAINTY_WEIGHT,
    )
    
    # 统计成本场各组件
    components = cost_field.components
    
    # 计算各组件的总和（排除 inf）
    component_sums = {}
    for comp_name, comp_array in components.items():
        valid_mask = np.isfinite(comp_array)
        if np.any(valid_mask):
            comp_sum = float(np.sum(comp_array[valid_mask]))
        else:
            comp_sum = 0.0
        component_sums[comp_name] = comp_sum
    
    # 提取关键组件
    ice_sum = component_sums.get("ice_risk", 0.0)
    wave_sum = component_sums.get("wave_risk", 0.0)
    edl_sum = component_sums.get("edl_risk", 0.0)
    edl_unc_sum = component_sums.get("edl_uncertainty_penalty", 0.0)
    
    print(f"[COST] ice_risk={ice_sum:.3f}, wave_risk={wave_sum:.3f}, "
          f"edl_risk={edl_sum:.3f}, edl_uncertainty={edl_unc_sum:.3f}")
    
    # 打印所有组件（用于调试）
    print(f"[COST] all_components: {list(components.keys())}")
    
    print()
    
    # ========================================================================
    # Step 4: 选取简单路径做成本评估
    # ========================================================================
    print("[STEP 4] 选取简单路径做成本评估...")
    print()
    
    simple_path = create_simple_path(grid, num_points=SIMPLE_PATH_POINTS)
    print(f"[PATH] created simple diagonal path with {len(simple_path)} points")
    print(f"[PATH] start: {simple_path[0]}, end: {simple_path[-1]}")
    
    # 计算路径成本分解
    route_breakdown = compute_route_cost_breakdown(grid, cost_field, simple_path)
    
    path_total_cost = route_breakdown.total_cost
    path_components = route_breakdown.component_totals
    
    path_ice_cost = path_components.get("ice_risk", 0.0)
    path_wave_cost = path_components.get("wave_risk", 0.0)
    path_edl_cost = path_components.get("edl_risk", 0.0)
    path_edl_unc_cost = path_components.get("edl_uncertainty_penalty", 0.0)
    
    print(f"[PATH_COST] total={path_total_cost:.3f}")
    print(f"[PATH_COST] ice={path_ice_cost:.3f}, wave={path_wave_cost:.3f}, "
          f"edl={path_edl_cost:.3f}, edl_unc={path_edl_unc_cost:.3f}")
    
    print()
    
    # ========================================================================
    # Step 5: 判定规则
    # ========================================================================
    print("[STEP 5] 执行判定规则...")
    print()
    
    failures = []
    
    # 检查 1: sic 数据有变化
    if sic_min >= sic_max:
        failures.append("sic_no_variation")
    
    # 检查 2: wave 数据有变化
    if wave_min >= wave_max:
        failures.append("wave_no_variation")
    
    # 检查 3: 冰风险成本 > 0
    if path_ice_cost <= 0:
        failures.append("ice_cost_zero")
    
    # 检查 4: 波浪风险成本 > 0
    if path_wave_cost <= 0:
        failures.append("wave_cost_zero")
    
    # 检查 5: 至少有一个 EDL 相关成本 > 0
    if path_edl_cost <= 0 and path_edl_unc_cost <= 0:
        failures.append("edl_cost_all_zero")
    
    # 检查 6: 成本场中的 EDL 组件存在
    if "edl_risk" not in components and "edl_uncertainty_penalty" not in components:
        failures.append("edl_components_missing")
    
    print()
    if not failures:
        print("CHECK_REAL_EDL_OK")
        return
    else:
        reason = ", ".join(failures)
        print(f"CHECK_REAL_EDL_FAIL: reason={reason}")
        return


if __name__ == "__main__":
    main()

