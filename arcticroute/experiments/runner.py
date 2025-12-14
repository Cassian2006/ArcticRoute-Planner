"""
核心运行器：统一的"运行一次规划并返回 DataFrame/字典"的封装。

功能：
- 单次规划运行（run_single_case）
- 批量规划运行（run_case_grid）
- 结果导出为 DataFrame/JSON/CSV
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Literal, Optional, List

import numpy as np
import pandas as pd

from arcticroute.config import get_scenario_by_name, get_edl_mode_config
from arcticroute.core.grid import make_demo_grid, load_real_grid_from_nc
from arcticroute.core.landmask import load_real_landmask_from_nc
from arcticroute.core.cost import build_demo_cost, build_cost_from_real_env
from arcticroute.core.env_real import load_real_env_for_grid
from arcticroute.core.astar import plan_route_latlon
from arcticroute.core.analysis import compute_route_cost_breakdown
from arcticroute.core.eco.vessel_profiles import get_default_profiles


ModeName = Literal["efficient", "edl_safe", "edl_robust"]


@dataclass
class SingleRunResult:
    """单次规划运行的结果数据类。
    
    Attributes:
        scenario: 场景名称
        mode: 规划模式（efficient / edl_safe / edl_robust）
        reachable: 是否可达
        distance_km: 路线距离（km），若不可达则为 None
        total_cost: 总成本，若不可达则为 None
        edl_risk_cost: EDL 风险成本，若不可达则为 None
        edl_unc_cost: EDL 不确定性成本，若不可达则为 None
        ice_cost: 冰风险成本，若不可达则为 None
        wave_cost: 波浪风险成本，若不可达则为 None
        ice_class_soft_cost: 冰级软约束成本，若不可达则为 None
        ice_class_hard_cost: 冰级硬约束成本，若不可达则为 None
        meta: 元数据字典，包含 vessel, cost_mode, use_real_data, ym, edl_backend 等
    """
    
    scenario: str
    mode: ModeName
    reachable: bool
    distance_km: Optional[float]
    total_cost: Optional[float]
    edl_risk_cost: Optional[float]
    edl_unc_cost: Optional[float]
    ice_cost: Optional[float]
    wave_cost: Optional[float]
    ice_class_soft_cost: Optional[float]
    ice_class_hard_cost: Optional[float]
    meta: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，便于导出。"""
        return asdict(self)
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """转换为扁平字典（meta 字段展开）。"""
        result = asdict(self)
        meta = result.pop("meta")
        result.update({f"meta_{k}": v for k, v in meta.items()})
        return result


def run_single_case(
    scenario: str,
    mode: ModeName,
    use_real_data: bool = True,
) -> SingleRunResult:
    """
    运行单次规划案例。
    
    Args:
        scenario: 场景名称（例如 "barents_to_chukchi", "kara_short" 等）
        mode: 规划模式（"efficient", "edl_safe", "edl_robust"）
        use_real_data: 是否使用真实数据（True）或 demo 数据（False）
    
    Returns:
        SingleRunResult 对象
    
    Raises:
        ValueError: 如果场景或模式不存在
    """
    # ========================================================================
    # Step 1: 获取场景配置
    # ========================================================================
    scenario_obj = get_scenario_by_name(scenario)
    if scenario_obj is None:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    # ========================================================================
    # Step 2: 获取 EDL 模式配置
    # ========================================================================
    try:
        mode_config = get_edl_mode_config(mode)
    except ValueError as e:
        raise ValueError(f"Unknown mode: {mode}") from e
    
    # ========================================================================
    # Step 3: 加载网格和陆地掩码
    # ========================================================================
    if use_real_data:
        try:
            grid = load_real_grid_from_nc(scenario_obj.ym)
            if grid is None:
                raise ValueError("Failed to load real grid")
            # 加载 landmask，需要先有 grid
            land_mask = load_real_landmask_from_nc(grid)
            if land_mask is None:
                print("[ENV] real landmask not available, using demo landmask")
                _, land_mask = make_demo_grid(ny=grid.shape()[0], nx=grid.shape()[1])
        except Exception as e:
            # 回退到 demo 网格
            print(f"[ENV] failed to load real grid/landmask: {e}, falling back to demo")
            grid, land_mask = make_demo_grid()
            use_real_data = False
    else:
        grid, land_mask = make_demo_grid()
    
    # ========================================================================
    # Step 4: 获取船舶配置
    # ========================================================================
    vessel_profiles = get_default_profiles()
    vessel = vessel_profiles.get(scenario_obj.vessel_profile)
    if vessel is None:
        vessel = vessel_profiles.get("panamax")  # 默认船舶
    
    # ========================================================================
    # Step 5: 构建成本场
    # ========================================================================
    cost_mode = "real_sic_if_available" if use_real_data else "demo_icebelt"
    real_env = None
    fallback_reason = None
    
    if use_real_data:
        try:
            real_env = load_real_env_for_grid(grid)
        except Exception:
            real_env = None
    
    # 根据模式配置构建成本场
    use_edl = mode_config.get("use_edl", False)
    w_edl = mode_config.get("w_edl", 0.0)
    use_edl_uncertainty = mode_config.get("use_edl_uncertainty", False)
    edl_uncertainty_weight = mode_config.get("edl_uncertainty_weight", 0.0)
    ice_penalty = mode_config.get("ice_penalty", 4.0)
    
    if use_real_data and real_env is not None and (real_env.sic is not None or real_env.wave_swh is not None):
        cost_field = build_cost_from_real_env(
            grid,
            land_mask,
            real_env,
            ice_penalty=ice_penalty,
            wave_penalty=0.0,
            vessel_profile=vessel,
            use_edl=use_edl,
            w_edl=w_edl,
            use_edl_uncertainty=use_edl_uncertainty,
            edl_uncertainty_weight=edl_uncertainty_weight,
        )
    else:
        # 回退到 demo 成本
        cost_field = build_demo_cost(
            grid, land_mask, ice_penalty=ice_penalty, ice_lat_threshold=75.0
        )
        if use_real_data:
            fallback_reason = "真实环境数据不可用"
    
    # ========================================================================
    # Step 6: 规划路线
    # ========================================================================
    path = plan_route_latlon(
        cost_field,
        scenario_obj.start_lat,
        scenario_obj.start_lon,
        scenario_obj.end_lat,
        scenario_obj.end_lon,
        neighbor8=True,
    )
    
    # ========================================================================
    # Step 7: 计算成本分解
    # ========================================================================
    if path:
        # 计算路线长度
        distance_km = _compute_path_length_km(path)
        
        # 计算成本分解
        breakdown = compute_route_cost_breakdown(grid, cost_field, path)
        total_cost = breakdown.total_cost
        
        # 提取各成本分量
        edl_risk_cost = breakdown.component_totals.get("edl_risk", None)
        edl_unc_cost = breakdown.component_totals.get("edl_uncertainty", None)
        ice_cost = breakdown.component_totals.get("ice", None)
        wave_cost = breakdown.component_totals.get("wave", None)
        ice_class_soft_cost = breakdown.component_totals.get("ice_class_soft", None)
        ice_class_hard_cost = breakdown.component_totals.get("ice_class_hard", None)
        
        reachable = True
    else:
        # 不可达
        distance_km = None
        total_cost = None
        edl_risk_cost = None
        edl_unc_cost = None
        ice_cost = None
        wave_cost = None
        ice_class_soft_cost = None
        ice_class_hard_cost = None
        reachable = False
    
    # ========================================================================
    # Step 8: 构建元数据
    # ========================================================================
    meta = {
        "ym": scenario_obj.ym,
        "use_real_data": use_real_data,
        "cost_mode": cost_mode,
        "fallback_reason": fallback_reason,
        "vessel_profile": scenario_obj.vessel_profile,
        "edl_backend": "miles" if use_edl else "none",
        "grid_shape": grid.shape(),
        "w_edl": float(w_edl),
        "use_edl_uncertainty": bool(use_edl_uncertainty),
        "edl_uncertainty_weight": float(edl_uncertainty_weight),
        "ice_penalty": float(ice_penalty),
    }
    
    # ========================================================================
    # Step 9: 返回结果
    # ========================================================================
    return SingleRunResult(
        scenario=scenario,
        mode=mode,
        reachable=reachable,
        distance_km=distance_km,
        total_cost=total_cost,
        edl_risk_cost=edl_risk_cost,
        edl_unc_cost=edl_unc_cost,
        ice_cost=ice_cost,
        wave_cost=wave_cost,
        ice_class_soft_cost=ice_class_soft_cost,
        ice_class_hard_cost=ice_class_hard_cost,
        meta=meta,
    )


def run_case_grid(
    scenarios: List[str],
    modes: List[ModeName],
    use_real_data: bool = True,
) -> pd.DataFrame:
    """
    逐个调用 run_single_case，返回一个长表格。
    
    Args:
        scenarios: 场景名称列表
        modes: 规划模式列表
        use_real_data: 是否使用真实数据
    
    Returns:
        DataFrame，列包括 scenario/mode + SingleRunResult 的各字段
    """
    results = []
    
    for scenario in scenarios:
        for mode in modes:
            try:
                result = run_single_case(scenario, mode, use_real_data)
                results.append(result)
            except Exception as e:
                # 记录错误但继续
                print(f"Warning: Failed to run {scenario} with {mode}: {e}")
                # 可选：添加一个失败的记录
                results.append(SingleRunResult(
                    scenario=scenario,
                    mode=mode,
                    reachable=False,
                    distance_km=None,
                    total_cost=None,
                    edl_risk_cost=None,
                    edl_unc_cost=None,
                    ice_cost=None,
                    wave_cost=None,
                    ice_class_soft_cost=None,
                    ice_class_hard_cost=None,
                    meta={"error": str(e)},
                ))
    
    # 转换为 DataFrame
    df = pd.DataFrame([result.to_flat_dict() for result in results])
    
    return df


# ============================================================================
# 辅助函数
# ============================================================================

def _compute_path_length_km(path: List[tuple]) -> float:
    """
    计算路径的总长度（单位：km）。
    
    Args:
        path: [(lat, lon), ...] 路径列表
    
    Returns:
        总长度（km）
    """
    if len(path) < 2:
        return 0.0
    
    total_dist = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(path[:-1], path[1:]):
        total_dist += _haversine_km(lat1, lon1, lat2, lon2)
    
    return total_dist


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    计算两点间的大圆距离（单位：km）。
    
    Args:
        lat1, lon1: 起点纬度、经度（度）
        lat2, lon2: 终点纬度、经度（度）
    
    Returns:
        距离（km）
    """
    import math
    
    R = 6371.0  # 地球平均半径
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

