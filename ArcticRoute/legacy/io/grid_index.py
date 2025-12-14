"""AIS 网格落点（lat/lon → y/x 索引）

前提：grid_spec.type == rectilinear_1d，对应网格坐标 lat1d(y), lon1d(x)。

实现：
- 加载网格坐标（优先从 ArcticRoute/data_processed/env_clean.nc via xarray；或从指定 grid_path 的 netcdf/zarr）
- 使用 numpy.searchsorted 找近邻索引，并做边界裁剪
- 为 parquet 添加 iy, ix 列，并统计 OOB 情况

约束：
- 非 dry-run 写盘并通过 register_artifact() 登记
- Windows 路径使用 os.path.join
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

from ArcticRoute.cache.index_util import register_artifact


def _read_parquet(path: str):
    if pl is not None:
        return pl.read_parquet(path)
    if pd is not None:
        return pd.read_parquet(path)  # type: ignore
    raise RuntimeError("No dataframe engine available (polars/pandas)")


def _write_parquet(df_any: Any, out_path: str) -> None:
    if pl and isinstance(df_any, pl.DataFrame):  # type: ignore[attr-defined]
        df_any.write_parquet(out_path)
        return
    if pd and isinstance(df_any, pd.DataFrame):  # type: ignore[attr-defined]
        df_any.to_parquet(out_path)
        return
    raise RuntimeError("Unsupported dataframe type for writing")


def _load_rect_grid(grid_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """加载 rectilinear_1d 网格坐标，返回 (lat1d, lon1d)。"""
    # 优先从默认 env_clean.nc 读取
    cand = grid_path or os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "env_clean.nc")
    if xr is None:
        raise RuntimeError("xarray is required to load rectilinear grid")
    if not os.path.exists(cand):
        raise FileNotFoundError(f"Grid file not found: {cand}")
    ds = xr.open_dataset(cand)
    # 容错坐标名
    lat_name = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else None)
    lon_name = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
    if not lat_name or not lon_name:
        raise ValueError("Grid dataset missing lat/lon coordinates")
    lat = ds[lat_name].values
    lon = ds[lon_name].values
    ds.close()
    # 允许 1D（理想）或 2D（取轴）
    if lat.ndim == 1 and lon.ndim == 1:
        return np.asarray(lat), np.asarray(lon)
    # 若为 2D，尝试取唯一下标轴
    if lat.ndim == 2 and lon.ndim == 2:
        # 常见情形：lat(y,x)、lon(y,x)；退化为取均值后唯一化坐标
        lat1d = np.nanmean(lat, axis=1)
        lon1d = np.nanmean(lon, axis=0)
        return np.asarray(lat1d), np.asarray(lon1d)
    raise ValueError("Unsupported grid coordinate dimensions")


def _nearest_index(axis_vals: np.ndarray, values: np.ndarray) -> np.ndarray:
    # searchsorted 返回插入点，取左右近邻中更近者，再裁剪到 [0, n-1]
    idx = np.searchsorted(axis_vals, values, side='left')
    idx_right = np.clip(idx, 0, len(axis_vals) - 1)
    idx_left = np.clip(idx - 1, 0, len(axis_vals) - 1)
    # 选择更近者
    dist_right = np.abs(axis_vals[idx_right] - values)
    dist_left = np.abs(axis_vals[idx_left] - values)
    choose_left = dist_left <= dist_right
    out = np.where(choose_left, idx_left, idx_right)
    # 边界裁剪
    return np.clip(out, 0, len(axis_vals) - 1)


def annotate_grid_index(parquet_in: str, parquet_out: str, grid_path: Optional[str] = None, dry_run: bool = True) -> Dict[str, Any]:
    """将经纬度映射到网格索引 iy/ix；返回统计信息。"""
    df = _read_parquet(parquet_in)
    if df.is_empty() if hasattr(df, 'is_empty') else df.empty:
        return {"out": parquet_out, "stats": {"rows": 0, "oob_pct": 0.0}}

    lat1d, lon1d = _load_rect_grid(grid_path)
    # 取列
    if isinstance(df, pd.DataFrame):
        latv = df["lat"].to_numpy()
        lonv = df["lon"].to_numpy()
    else:
        latv = df.get_column("lat").to_numpy()  # type: ignore
        lonv = df.get_column("lon").to_numpy()  # type: ignore

    # 近邻索引
    iy = _nearest_index(lat1d, latv)
    ix = _nearest_index(lon1d, lonv)

    # OOB 统计：由于近邻+裁剪，理论 OOB 应接近 0（仅考虑 NaN 情况）
    import numpy as np
    oob_mask = np.isnan(latv) | np.isnan(lonv)
    oob_pct = float(oob_mask.sum()) / float(len(latv)) if len(latv) > 0 else 0.0
    stats = {
        "rows": int(len(latv)),
        "iy_min": int(iy.min()) if len(iy) else None,
        "iy_max": int(iy.max()) if len(iy) else None,
        "ix_min": int(ix.min()) if len(ix) else None,
        "ix_max": int(ix.max()) if len(ix) else None,
        "oob_pct": round(100.0 * oob_pct, 4),
    }

    if not dry_run:
        if isinstance(df, pd.DataFrame):
            dfo = df.copy()
            dfo["iy"] = iy
            dfo["ix"] = ix
        else:
            dfo = df.with_columns([
                pl.Series("iy", iy),  # type: ignore
                pl.Series("ix", ix),  # type: ignore
            ])
        os.makedirs(os.path.dirname(parquet_out), exist_ok=True)
        _write_parquet(dfo, parquet_out)
        try:
            register_artifact(run_id=os.environ.get("RUN_ID", ""), kind="ais_grid_indexed", path=parquet_out, attrs=stats)
        except Exception:
            pass

    return {"out": parquet_out, "stats": stats}


__all__ = ["annotate_grid_index"]

