from __future__ import annotations
"""
按整图生成 bucket 栅格：bucket_{ym}.nc
- 依据 Bucketer（region/season/vessel）逐像元推断 bucket
- 输出变量：bucket (int32)，并在 attrs 中写入 {"mapping": {int_id: bucket_str}}

REUSE: 网格选择优先使用 R_ice_eff_{ym}.nc 的 y/x；否则回退 ArcticRoute/data_processed/env_clean.nc。
"""
from typing import Dict, Any, Tuple
import os
import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

from .bucketer import Bucketer

ROOT = os.path.join(os.getcwd(), "ArcticRoute")
RISK_DIR = os.path.join(ROOT, "data_processed", "risk")
ENV_PATH = os.path.join(ROOT, "data_processed", "env_clean.nc")


def _season_from_ym(ym: str) -> str:
    y = int(ym[:4]); m = int(ym[4:6])
    if m in (12,1,2): return "DJF"
    if m in (3,4,5): return "MAM"
    if m in (6,7,8): return "JJA"
    return "SON"


def _grid_ref(ym: str) -> Tuple[np.ndarray, np.ndarray]:
    if xr is None:
        raise RuntimeError("xarray required")
    # 优先 risk 冰图网格
    cand = os.path.join(RISK_DIR, f"R_ice_eff_{ym}.nc")
    if os.path.exists(cand):
        ds = xr.open_dataset(cand)
        da = ds[list(ds.data_vars)[0]]
        y = np.asarray(da[da.dims[-2]].values)
        x = np.asarray(da[da.dims[-1]].values)
        try:
            ds.close()
        except Exception:
            pass
        return y, x
    # 回退 env_clean
    if os.path.exists(ENV_PATH):
        ds = xr.open_dataset(ENV_PATH)
        # 兼容 lat/lon/y/x 命名
        if "y" in ds.dims and "x" in ds.dims:
            y = np.asarray(ds["y"].values); x = np.asarray(ds["x"].values)
        else:
            y = np.asarray((ds.coords.get("lat") or ds.coords.get("latitude")).values)
            x = np.asarray((ds.coords.get("lon") or ds.coords.get("longitude")).values)
        try:
            ds.close()
        except Exception:
            pass
        return y, x
    # 最小回退
    return np.linspace(50, 85, 128, dtype=np.float32), np.linspace(-180, 180, 256, dtype=np.float32)


def build_bucket_grid(ym: str, bucketer: Bucketer, default_vessel: str = "cargo") -> str:
    if xr is None:
        raise RuntimeError("xarray required")
    y, x = _grid_ref(ym)
    H = int(y.shape[0]); W = int(x.shape[0])
    # 网格经纬：假设 y 为纬度，x 为经度，构造 mesh
    # 注意：多数数据 y 从北到南递减，但 bucketer 仅用数值不受顺序影响
    lat = y.reshape(-1, 1).repeat(W, axis=1)
    lon = x.reshape(1, -1).repeat(H, axis=0)
    import pandas as pd
    ts = pd.Timestamp(f"{ym}01")
    # 遍历生成字符串 bucket，并映射到整数 ID
    buckets_str = np.empty((H, W), dtype=object)
    for i in range(H):
        for j in range(W):
            buckets_str[i, j] = bucketer.infer_bucket(float(lat[i, j]), float(lon[i, j]), ts, default_vessel)
    # 映射到 int32
    uniq = sorted({str(b) for b in buckets_str.ravel()})
    id_map: Dict[str, int] = {b: k for k, b in enumerate(uniq)}
    grid = np.zeros((H, W), dtype=np.int32)
    for i in range(H):
        for j in range(W):
            grid[i, j] = id_map[str(buckets_str[i, j])]
    # 写文件
    os.makedirs(RISK_DIR, exist_ok=True)
    out = os.path.join(RISK_DIR, f"bucket_{ym}.nc")
    da = xr.DataArray(grid, dims=("y", "x"), coords={"y": y, "x": x}, name="bucket")
    ds = xr.Dataset({"bucket": da})
    ds["bucket"].attrs.update({"long_name": "Domain bucket id (int)", "mapping": id_map})
    ds.to_netcdf(out)
    try:
        ds.close()
    except Exception:
        pass
    return out

__all__ = ["build_bucket_grid"]

