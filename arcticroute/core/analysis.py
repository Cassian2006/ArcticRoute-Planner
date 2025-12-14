"""
成本分解与路线剖面分析工具。

提供沿路径的成本分解、剖面数据等功能。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cost import CostField
from .grid import Grid2D


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    计算两点间的大圆距离（单位：km）。
    
    Args:
        lat1, lon1: 起点纬度、经度（度）
        lat2, lon2: 终点纬度、经度（度）
    
    Returns:
        距离（km）
    """
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


def simplify_path_by_distance(
    path: List[Tuple[float, float]],
    min_step_km: float = 10.0,
    max_points: int = 200,
) -> List[Tuple[float, float]]:
    """
    简化 (lat, lon) 路径，仅用于可视化：
    - 保留首尾点
    - 与上一个保留点的球面距离 >= min_step_km 时才保留
    - 若结果仍超过 max_points，则等间隔抽样
    """
    if len(path) <= 2:
        return list(path)

    simplified: List[Tuple[float, float]] = [path[0]]
    last_lat, last_lon = path[0]
    for lat, lon in path[1:-1]:
        if haversine_km(last_lat, last_lon, lat, lon) >= min_step_km:
            simplified.append((lat, lon))
            last_lat, last_lon = lat, lon
    simplified.append(path[-1])

    if len(simplified) > max_points:
        step = max(1, len(simplified) // max_points)
        simplified = simplified[::step]
        if simplified[-1] != path[-1]:
            simplified.append(path[-1])

    return simplified


@dataclass
class RouteCostBreakdown:
    """路线成本分解结果数据类。"""

    total_cost: float
    component_totals: Dict[str, float]
    component_fractions: Dict[str, float]
    # 沿程信息
    s_km: List[float]
    component_along_path: Dict[str, List[float]]


@dataclass
class RouteScore:
    """路线综合评分数据类。
    
    包含原始指标、归一化指标和根据用户权重计算的综合分数。
    """

    # 原始指标
    distance_km: float
    total_cost: float
    fuel_t: Optional[float]
    co2_t: Optional[float]
    edl_risk_cost: float
    edl_uncertainty_cost: float

    # 归一化指标（0..1，1 表示最差）
    norm_distance: float
    norm_fuel: float
    norm_edl_risk: float
    norm_edl_uncertainty: float

    # 综合分数（根据用户权重计算）
    composite_score: float


@dataclass
class RouteCostProfile:
    """路线成本剖面数据类。"""

    distance_km: np.ndarray  # 累计距离，shape (n_points,)
    total_cost: np.ndarray  # 总成本沿程值，shape (n_points,)
    components: Dict[str, np.ndarray]  # 各成本分量沿程值
    edl_uncertainty: Optional[np.ndarray] = None  # EDL 不确定性沿程值，shape (n_points,)
    ais_density: Optional[np.ndarray] = None  # AIS 拥挤度沿程值，shape (n_points,)


def compute_route_cost_breakdown(
    grid: Grid2D,
    cost_field: CostField,
    route_latlon: Sequence[Tuple[float, float]],
) -> RouteCostBreakdown:
    """
    计算沿路径的成本分解与剖面数据。

    功能：
    - 使用与 A* 一致的最近邻映射，将 route_latlon 映射成 (i, j) 索引序列。
    - 沿着路径采样：
      - total_cost: 对 cost_field.cost 沿路径求和。
      - 对每个组件 name, arr in cost_field.components 中：
        - 计算该组件沿路径的累积值 component_totals[name]。
      - 计算 component_fractions[name] = component_totals[name] / total_cost（total_cost<=0 时全设 0）。
    - 生成剖面数据：
      - s_km：从起点开始的累计距离（km），长度与路径长度一致。
      - component_along_path[name]: 该组件在每一步的值（与 s_km 对齐）。

    Args:
        grid: Grid2D 对象
        cost_field: CostField 对象
        route_latlon: [(lat, lon), ...] 路径列表

    Returns:
        RouteCostBreakdown 对象
    """
    # 处理空路径
    if not route_latlon:
        return RouteCostBreakdown(
            total_cost=0.0,
            component_totals={},
            component_fractions={},
            s_km=[],
            component_along_path={},
        )

    # 将 route_latlon 映射到 (i, j) 索引
    lat2d = grid.lat2d
    lon2d = grid.lon2d
    ny, nx = grid.shape()

    ij_path = []
    for lat, lon in route_latlon:
        # 计算到所有网格点的距离
        dist = np.sqrt((lat2d - lat) ** 2 + (lon2d - lon) ** 2)
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        # 确保索引在范围内
        i = np.clip(i, 0, ny - 1)
        j = np.clip(j, 0, nx - 1)
        ij_path.append((i, j))

    # 计算沿路径的成本
    cost = cost_field.cost
    total_cost = 0.0
    for i, j in ij_path:
        if np.isfinite(cost[i, j]):
            total_cost += cost[i, j]

    # 计算各组件的总贡献
    component_totals: Dict[str, float] = {}
    for comp_name, comp_array in cost_field.components.items():
        comp_total = 0.0
        for i, j in ij_path:
            if np.isfinite(comp_array[i, j]):
                comp_total += comp_array[i, j]
        component_totals[comp_name] = comp_total

    # 计算各组件的占比
    component_fractions: Dict[str, float] = {}
    if total_cost > 0:
        for comp_name, comp_total in component_totals.items():
            component_fractions[comp_name] = comp_total / total_cost
    else:
        for comp_name in component_totals:
            component_fractions[comp_name] = 0.0

    # 生成剖面数据：s_km（累计距离）
    s_km = [0.0]
    for i in range(1, len(route_latlon)):
        lat1, lon1 = route_latlon[i - 1]
        lat2, lon2 = route_latlon[i]
        dist = haversine_km(lat1, lon1, lat2, lon2)
        s_km.append(s_km[-1] + dist)

    # 生成各组件沿路径的值
    component_along_path: Dict[str, List[float]] = {}
    for comp_name, comp_array in cost_field.components.items():
        comp_values = []
        for i, j in ij_path:
            val = comp_array[i, j]
            # 如果是 inf，记为 0（或者可以记为 NaN）
            if not np.isfinite(val):
                val = 0.0
            comp_values.append(val)
        component_along_path[comp_name] = comp_values

    return RouteCostBreakdown(
        total_cost=total_cost,
        component_totals=component_totals,
        component_fractions=component_fractions,
        s_km=s_km,
        component_along_path=component_along_path,
    )


def compute_route_profile(
    route_latlon: Sequence[Tuple[float, float]],
    cost_field: CostField,
) -> RouteCostProfile:
    """
    计算沿路径的成本剖面数据。

    功能：
    - 将 route_latlon 映射到 (i, j) 索引序列
    - 沿路径采样总成本和各成本分量
    - 采样 EDL 不确定性（如果可用）
    - 生成累计距离数组

    Args:
        route_latlon: [(lat, lon), ...] 路径列表
        cost_field: CostField 对象

    Returns:
        RouteCostProfile 对象
    """
    # 处理空路径
    if not route_latlon:
        return RouteCostProfile(
            distance_km=np.array([], dtype=float),
            total_cost=np.array([], dtype=float),
            components={},
            edl_uncertainty=None,
        )

    # 将 route_latlon 映射到 (i, j) 索引
    lat2d = cost_field.grid.lat2d
    lon2d = cost_field.grid.lon2d
    ny, nx = cost_field.grid.shape()

    ij_path = []
    for lat, lon in route_latlon:
        # 计算到所有网格点的距离
        dist = np.sqrt((lat2d - lat) ** 2 + (lon2d - lon) ** 2)
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        # 确保索引在范围内
        i = np.clip(i, 0, ny - 1)
        j = np.clip(j, 0, nx - 1)
        ij_path.append((i, j))

    # 计算累计距离
    distance_km = [0.0]
    for i in range(1, len(route_latlon)):
        lat1, lon1 = route_latlon[i - 1]
        lat2, lon2 = route_latlon[i]
        dist = haversine_km(lat1, lon1, lat2, lon2)
        distance_km.append(distance_km[-1] + dist)
    distance_km_array = np.asarray(distance_km, dtype=float)

    # 采样总成本
    cost = cost_field.cost
    total_cost_values = []
    for i, j in ij_path:
        val = cost[i, j]
        if not np.isfinite(val):
            val = 0.0
        total_cost_values.append(val)
    total_cost_array = np.asarray(total_cost_values, dtype=float)

    # 采样各成本分量
    components = {}
    for comp_name, comp_array in cost_field.components.items():
        comp_values = []
        for i, j in ij_path:
            val = comp_array[i, j]
            if not np.isfinite(val):
                val = 0.0
            comp_values.append(val)
        components[comp_name] = np.asarray(comp_values, dtype=float)

    # 采样 EDL 不确定性（如果可用）
    edl_uncertainty = None
    if cost_field.edl_uncertainty is not None:
        edl_unc_values = []
        for i, j in ij_path:
            if 0 <= i < ny and 0 <= j < nx:
                val = float(cost_field.edl_uncertainty[i, j])
                # clip 到 [0, 1]，避免异常值
                val = float(np.clip(val, 0.0, 1.0))
                edl_unc_values.append(val)
            else:
                edl_unc_values.append(np.nan)
        edl_uncertainty = np.asarray(edl_unc_values, dtype=float)

    # 采样 AIS 拥挤度（如果可用）
    ais_density_profile = None
    if "ais_density" in cost_field.components:
        ais_vals = []
        comp_array = cost_field.components["ais_density"]
        for i, j in ij_path:
            val = comp_array[i, j]
            if not np.isfinite(val):
                val = 0.0
            ais_vals.append(val)
        ais_density_profile = np.asarray(ais_vals, dtype=float)

    return RouteCostProfile(
        distance_km=distance_km_array,
        total_cost=total_cost_array,
        components=components,
        edl_uncertainty=edl_uncertainty,
        ais_density=ais_density_profile,
    )


def compute_route_scores(
    breakdowns: Dict[str, RouteCostBreakdown],
    eco_by_key: Dict[str, Optional[Dict[str, float]]],
    weight_risk: float,
    weight_uncertainty: float,
    weight_fuel: float,
) -> Dict[str, RouteScore]:
    """
    根据三条路线的指标，计算归一化和综合分数。

    功能：
    - 从 breakdowns 中提取：distance_km、total_cost、edl_risk、edl_uncertainty_penalty
    - 从 eco_by_key 中提取：fuel_t、co2_t（缺失则为 None）
    - 对每项进行 min-max 归一化（越大越"差"，归一化后 1 表示最差）
    - 计算综合分数：composite = weight_fuel * norm_fuel + weight_risk * norm_edl_risk + weight_uncertainty * norm_edl_uncertainty

    Args:
        breakdowns: 例如 {"efficient": RouteCostBreakdown(...), "edl_safe": ..., "edl_robust": ...}
        eco_by_key: 例如 {"efficient": {"fuel_total_t": ..., "co2_total_t": ...}, ...}
                    缺失的键或 None 值会被跳过
        weight_risk: EDL 风险权重（0..1）
        weight_uncertainty: EDL 不确定性权重（0..1）
        weight_fuel: 燃油权重（0..1）

    Returns:
        {"efficient": RouteScore(...), "edl_safe": ..., "edl_robust": ...}
    """
    # 第一步：收集原始指标
    raw_data = {}
    
    for key, breakdown in breakdowns.items():
        # 从 breakdown 中提取指标
        distance_km = breakdown.s_km[-1] if breakdown.s_km else 0.0
        total_cost = breakdown.total_cost
        edl_risk_cost = breakdown.component_totals.get("edl_risk", 0.0)
        edl_uncertainty_cost = breakdown.component_totals.get("edl_uncertainty_penalty", 0.0)
        
        # 从 eco_by_key 中提取燃油和 CO2
        eco_data = eco_by_key.get(key)
        fuel_t = None
        co2_t = None
        if eco_data is not None:
            fuel_t = eco_data.get("fuel_total_t")
            co2_t = eco_data.get("co2_total_t")
        
        raw_data[key] = {
            "distance_km": distance_km,
            "total_cost": total_cost,
            "fuel_t": fuel_t,
            "co2_t": co2_t,
            "edl_risk_cost": edl_risk_cost,
            "edl_uncertainty_cost": edl_uncertainty_cost,
        }
    
    # 第二步：计算 min-max 归一化
    # 对每个指标，找出最小值和最大值，然后归一化到 [0, 1]
    # 注意：如果所有值相同或只有一条路线，需要特殊处理
    
    def minmax_normalize(values: List[float]) -> Dict[int, float]:
        """
        对一组值进行 min-max 归一化。
        
        Args:
            values: 数值列表
        
        Returns:
            {index: normalized_value} 字典
        """
        min_val = min(values)
        max_val = max(values)
        
        # 如果所有值相同，返回全 0（最优）
        if max_val == min_val:
            return {key: 0.0 for key in range(len(values))}
        
        # 否则进行标准 min-max 归一化
        return {
            key: (values[key] - min_val) / (max_val - min_val)
            for key in range(len(values))
        }
    
    # 收集各指标的值列表
    keys_list = list(raw_data.keys())
    distance_values = [raw_data[k]["distance_km"] for k in keys_list]
    fuel_values = [raw_data[k]["fuel_t"] if raw_data[k]["fuel_t"] is not None else 0.0 for k in keys_list]
    edl_risk_values = [raw_data[k]["edl_risk_cost"] for k in keys_list]
    edl_uncertainty_values = [raw_data[k]["edl_uncertainty_cost"] for k in keys_list]
    
    # 进行归一化
    norm_distance_dict = minmax_normalize(distance_values)
    norm_fuel_dict = minmax_normalize(fuel_values)
    norm_edl_risk_dict = minmax_normalize(edl_risk_values)
    norm_edl_uncertainty_dict = minmax_normalize(edl_uncertainty_values)
    
    # 第三步：计算综合分数并构造 RouteScore 对象
    scores = {}
    
    for idx, key in enumerate(keys_list):
        norm_distance = norm_distance_dict[idx]
        norm_fuel = norm_fuel_dict[idx]
        norm_edl_risk = norm_edl_risk_dict[idx]
        norm_edl_uncertainty = norm_edl_uncertainty_dict[idx]
        
        # 综合分数：加权和
        # 注意：这里不包含 distance，因为 distance 已经在 fuel 中反映（燃油与距离相关）
        composite_score = (
            weight_fuel * norm_fuel
            + weight_risk * norm_edl_risk
            + weight_uncertainty * norm_edl_uncertainty
        )
        
        scores[key] = RouteScore(
            distance_km=raw_data[key]["distance_km"],
            total_cost=raw_data[key]["total_cost"],
            fuel_t=raw_data[key]["fuel_t"],
            co2_t=raw_data[key]["co2_t"],
            edl_risk_cost=raw_data[key]["edl_risk_cost"],
            edl_uncertainty_cost=raw_data[key]["edl_uncertainty_cost"],
            norm_distance=norm_distance,
            norm_fuel=norm_fuel,
            norm_edl_risk=norm_edl_risk,
            norm_edl_uncertainty=norm_edl_uncertainty,
            composite_score=composite_score,
        )
    
    return scores
