"""通用网格/时间对齐工具（Phase A 最小实现）

- ensure_common_grid(ds, grid_spec) -> xr.Dataset
  目标：若 ds 已与 grid_spec 契约一致，则原样返回（不复制，不修改）。
  说明：Phase A 仅做探测与等价返回；若不一致，当前不做重采样/重投影，直接原样返回。

- align_time(ds, freq) -> xr.Dataset
  目标：若时间轴已是指定频率或不足以推断（<2 步），原样返回。
  说明：Phase A 不执行插值/重采样；若检测到显著不匹配，仍原样返回（最小实现）。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr


def _get_name(ds: xr.Dataset, cands: Tuple[str, ...]) -> Optional[str]:
    for n in cands:
        if n in ds.coords or n in ds.variables:
            return n
    return None


def _median_abs_step(a: np.ndarray) -> float:
    if a.size <= 1:
        return float("nan")
    diffs = np.diff(a.astype(float))
    diffs = np.abs(diffs[np.nonzero(diffs)])
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs))


def _approx_equal(a: Optional[float], b: Optional[float], tol: float = 1e-3) -> bool:
    if a is None or b is None:
        return False
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    return abs(a - b) <= max(tol, 0.01 * max(abs(a), abs(b), 1.0))


def ensure_common_grid(ds: xr.Dataset, grid_spec: Dict[str, Any]) -> xr.Dataset:
    """最小实现：若 ds 的经纬网格与 grid_spec 一致，则原样返回；否则也原样返回。

    grid_spec 兼容 A-03 的 JSON 结构，优先读取：
    - contract.grid_resolution.lat_deg, lon_deg
    - sic_summary.coords.lat/lon 或 cost_summary.coords
    - sic_summary.grid.grid_type
    """
    try:
        lat_name = _get_name(ds, ("lat", "latitude", "y"))
        lon_name = _get_name(ds, ("lon", "longitude", "x"))
        # 目标分辨率
        gs = grid_spec or {}
        contract = (gs.get("contract") or {})
        gres = (contract.get("grid_resolution") or {})
        lat_deg_tgt = gres.get("lat_deg")
        lon_deg_tgt = gres.get("lon_deg")

        # 当前分辨率（仅 1D rectilinear 可靠）
        lat_deg = None
        lon_deg = None
        if lat_name and lon_name:
            lat = ds[lat_name]
            lon = ds[lon_name]
            if lat.ndim == 1:
                try:
                    lat_deg = _median_abs_step(lat.values)
                except Exception:
                    pass
            if lon.ndim == 1:
                try:
                    lon_deg = _median_abs_step(lon.values)
                except Exception:
                    pass
        # 若已匹配 -> 原样返回
        if (_approx_equal(lat_deg, lat_deg_tgt) or lat_deg_tgt is None) and (
            _approx_equal(lon_deg, lon_deg_tgt) or lon_deg_tgt is None
        ):
            return ds  # 不变
        # Phase A：不做重采样/重投影，直接返回原 ds
        return ds
    except Exception:
        # 容错：任何异常都不改变输入
        return ds


def align_time(ds: xr.Dataset, freq: Optional[str]) -> xr.Dataset:
    """最小实现：若时间轴已与给定 freq 对齐或无法推断，则原样返回。

    行为：
    - 若 ds 无 time 或 freq 为空 -> 原样返回
    - 若 time 步数 < 2 -> 原样返回
    - 若推断频率与 freq 近似一致 -> 原样返回
    - 其他情况：Phase A 不做重采样，原样返回
    """
    if ("time" not in ds.coords) or not freq:
        return ds
    try:
        tindex = ds["time"].to_index() if hasattr(ds["time"], "to_index") else pd.Index(ds["time"].values)
    except Exception:
        return ds
    if len(tindex) < 2:
        return ds

    # 推断现有频率
    inferred: Optional[str] = None
    try:
        inferred = pd.infer_freq(tindex)
    except Exception:
        inferred = None

    if inferred:
        # 简单等价判断（忽略大小写）
        if str(inferred).upper() == str(freq).upper():
            return ds
    else:
        # 使用中位步长（小时）做近似判断
        try:
            deltas = np.diff(tindex.values.astype("datetime64[h]"))
            hours = float(np.median(deltas.astype("timedelta64[h]") / np.timedelta64(1, "h")))
            # 将 freq 映射到小时
            f = str(freq).upper()
            target_h = None
            if f.endswith("H") and f[:-1].isdigit():
                target_h = float(int(f[:-1]))
            elif f == "H":
                target_h = 1.0
            elif f.endswith("D") and f[:-1].isdigit():
                target_h = float(int(f[:-1]) * 24)
            elif f == "D":
                target_h = 24.0
            if target_h is not None and abs(hours - target_h) <= 0.1:
                return ds
        except Exception:
            return ds

    # Phase A：不实际对齐，原样返回
    return ds

