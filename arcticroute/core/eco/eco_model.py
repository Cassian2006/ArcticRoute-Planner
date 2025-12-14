"""
简化版 ECO（能耗）估算模块。

提供：
- EcoRouteEstimate 数据类
- estimate_route_eco(): 基于路线和船舶参数估算能耗
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from .vessel_profiles import VesselProfile


@dataclass
class EcoRouteEstimate:
    """单条路线的 ECO 估算结果。"""

    distance_km: float
    travel_time_h: float
    fuel_total_t: float
    co2_total_t: float


def _haversine_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """计算两点间的大圆距离（单位：km）。

    Args:
        lat1, lon1: 起点纬度、经度（度）
        lat2, lon2: 终点纬度、经度（度）

    Returns:
        距离（km）
    """
    R = 6371.0  # 地球平均半径
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def estimate_route_eco(
    route_latlon: List[Tuple[float, float]],
    vessel: VesselProfile,
    co2_per_ton_fuel: float = 3.114,
) -> EcoRouteEstimate:
    """估算航程的 ECO（能耗）指标。

    使用简化模型：
    - 距离：沿 route_latlon 使用 Haversine 计算
    - 航速：使用 vessel.design_speed_kn（节），换算成 km/h
    - 燃油：distance_km * vessel.base_fuel_per_km
    - CO2：fuel_total_t * co2_per_ton_fuel

    Args:
        route_latlon: 路线点列表 [(lat, lon), ...]
        vessel: VesselProfile 船舶参数
        co2_per_ton_fuel: CO2 排放系数（t CO2 / t fuel），默认 3.114

    Returns:
        EcoRouteEstimate 对象，包含距离、时间、燃油、CO2 等指标
    """
    # 若路线为空，返回全 0
    if not route_latlon or len(route_latlon) < 2:
        return EcoRouteEstimate(
            distance_km=0.0,
            travel_time_h=0.0,
            fuel_total_t=0.0,
            co2_total_t=0.0,
        )

    # 计算总距离
    total_distance_km = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(route_latlon[:-1], route_latlon[1:]):
        total_distance_km += _haversine_km(lat1, lon1, lat2, lon2)

    # 计算航行时间（小时）
    # 航速：节 -> km/h，1 节 = 1.852 km/h
    speed_kmh = vessel.design_speed_kn * 1.852
    travel_time_h = total_distance_km / speed_kmh if speed_kmh > 0 else 0.0

    # 计算燃油消耗（吨）
    fuel_total_t = total_distance_km * vessel.base_fuel_per_km

    # 计算 CO2 排放（吨）
    co2_total_t = fuel_total_t * co2_per_ton_fuel

    return EcoRouteEstimate(
        distance_km=total_distance_km,
        travel_time_h=travel_time_h,
        fuel_total_t=fuel_total_t,
        co2_total_t=co2_total_t,
    )

