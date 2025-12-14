# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np

# 简单地球距离（km）
def haversine_km(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    R = 6371.0
    rad = np.pi / 180.0
    dlat = (lat2 - lat1) * rad
    dlon = (lon2 - lon1) * rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1 * rad) * np.cos(lat2 * rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c


def _ij_to_lonlat(env_lat: np.ndarray, env_lon: np.ndarray, i: int, j: int) -> Tuple[float, float]:
    # env_lat/env_lon 支持 1D 或 2D
    if env_lat.ndim == 1 and env_lon.ndim == 1:
        i2 = int(np.clip(i, 0, len(env_lat) - 1))
        j2 = int(np.clip(j, 0, len(env_lon) - 1))
        return float(env_lat[i2]), float(env_lon[j2])
    # 2D
    i2 = int(np.clip(i, 0, env_lat.shape[0] - 1))
    j2 = int(np.clip(j, 0, env_lat.shape[1] - 1))
    return float(env_lat[i2, j2]), float(env_lon[i2, j2])


_WARNED_FUEL_FALLBACK = False

def _resolve_base_fuel_per_km(vp: Dict[str, Any] | None) -> float:
    """统一推断 base_fuel_per_km（t/km）。
    优先顺序：
    1) 显式 base_fuel_per_km；
    2) 显式 base_fuel_per_nm → 换算为 t/km（除以 1.852）；
    3) 由 dwt 线性估算：1.5e-5 * dwt（与现有预设量级一致：40k→0.6~0.8, 80k→1.2~1.3）；
    4) 保守兜底 0.05。
    缺失时仅打印一次警告（不抛异常）。
    """
    global _WARNED_FUEL_FALLBACK
    try:
        if isinstance(vp, dict):
            if "base_fuel_per_km" in vp:
                return float(vp.get("base_fuel_per_km", 0.05))
            if "base_fuel_per_nm" in vp:
                try:
                    v_nm = float(vp.get("base_fuel_per_nm", 0.05))
                    return max(0.05, v_nm / 1.852)
                except Exception:
                    pass
            if "dwt" in vp:
                val = float(1.5e-5 * float(vp.get("dwt", 0.0)))
                if not _WARNED_FUEL_FALLBACK:
                    print(f"[WARN] vessel_profile 缺少 base_fuel_per_km，按 dwt 估算为 {val:.3f} t/km")
                    _WARNED_FUEL_FALLBACK = True
                return max(0.05, val)
    except Exception:
        pass
    if not _WARNED_FUEL_FALLBACK:
        print("[WARN] vessel_profile 缺少 base_fuel_per_km/dwt，使用保守缺省 0.05 t/km")
        _WARNED_FUEL_FALLBACK = True
    return 0.05

def compute_fuel_along_route(
    path_ij: List[Tuple[int, int]],
    env_lat: np.ndarray,
    env_lon: np.ndarray,
    newenv_for_eco: Dict[str, Any],
    vessel_profile: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    """
    基于 newenv 的简化燃油模型：
      - base_fuel_per_km：吨/km，默认 0.05，可由 vessel_profile 覆盖
      - 冰修正：f_ice = 1 + a1 * sic^2 + a2 * sithick
      - 浪修正：f_wave = 1 + b1 * max(0, swh - 1.0)
      - 总系数 f = f_ice * f_wave
    返回：{"fuel_tons", "co2_tons", "base_fuel_tons"}
    """
    if path_ij is None or len(path_ij) < 2:
        return {"fuel_tons": 0.0, "co2_tons": 0.0, "base_fuel_tons": 0.0}

    vp = vessel_profile or {}
    base_fuel_per_km = _resolve_base_fuel_per_km(vp)

    # 系数（占位可调）
    a1 = float(vp.get("a1_sic_quad", 1.2))
    a2 = float(vp.get("a2_sithick", 0.2))
    b1 = float(vp.get("b1_wave_swh", 0.12))
    wave_thr = float(vp.get("wave_threshold_m", 1.0))
    ice_sens = float(vp.get("ice_sensitivity", 1.0))
    wave_sens = float(vp.get("wave_sensitivity", 1.0))

    sic_da = newenv_for_eco.get("sic", None)
    sth_da = newenv_for_eco.get("sithick", None)
    swh_da = newenv_for_eco.get("wave_swh", None)

    # 逐步计算距离与修正
    lat_list = []
    lon_list = []
    for (i, j) in path_ij:
        lat, lon = _ij_to_lonlat(env_lat, env_lon, int(i), int(j))
        lat_list.append(lat)
        lon_list.append(lon)
    lat_arr = np.asarray(lat_list, dtype=float)
    lon_arr = np.asarray(lon_list, dtype=float)

    # 邻段距离（km）
    if len(lat_arr) >= 2:
        seg_dist = haversine_km(lon_arr[:-1], lat_arr[:-1], lon_arr[1:], lat_arr[1:])
    else:
        seg_dist = np.zeros(0, dtype=float)

    # 栅格点环境值（近邻取值：用每段起点的格点值）
    def _sample_da(da) -> np.ndarray:
        if da is None:
            return np.zeros_like(seg_dist)
        arr = np.asarray(da.values)
        # 期望 dims=(y,x)
        H, W = arr.shape[-2], arr.shape[-1]
        vals = []
        for (i, j) in path_ij[:-1]:
            ii = int(np.clip(i, 0, H - 1))
            jj = int(np.clip(j, 0, W - 1))
            vals.append(float(arr[ii, jj]))
        return np.asarray(vals, dtype=float)

    sic_v = _sample_da(sic_da)
    sth_v = _sample_da(sth_da)
    swh_v = _sample_da(swh_da)

    # 修正系数
    sic_v = np.clip(sic_v, 0.0, 1.0)
    sth_v = np.clip(sth_v, 0.0, None)  # m
    swh_v = np.clip(swh_v, 0.0, None)  # m

    f_ice = 1.0 + ice_sens * (a1 * (sic_v ** 2.0) + a2 * sth_v)
    f_wave = 1.0 + wave_sens * (b1 * np.clip(swh_v - wave_thr, 0.0, None))
    f_total = f_ice * f_wave

    base_fuel_seg = base_fuel_per_km * seg_dist
    fuel_seg = base_fuel_seg * f_total

    base_fuel_tons = float(np.nansum(base_fuel_seg))
    fuel_tons = float(np.nansum(fuel_seg))
    co2_tons = fuel_tons * 3.114

    return {
        "fuel_tons": round(fuel_tons, 4),
        "co2_tons": round(co2_tons, 4),
        "base_fuel_tons": round(base_fuel_tons, 4),
    }

