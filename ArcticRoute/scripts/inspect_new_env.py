"""
体检 data_processed/newenv 下的标准化 NetCDF 文件，打印变量统计摘要。

用法示例：
python ArcticRoute/scripts/inspect_new_env.py
"""
from __future__ import annotations

from pathlib import Path
import argparse
from typing import Optional, List, Tuple

import numpy as np
import xarray as xr


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


_LAT_CANDS = ["lat", "latitude", "nav_lat", "y"]
_LON_CANDS = ["lon", "longitude", "nav_lon", "x"]


def detect_lat_lon_names(ds: xr.Dataset) -> Tuple[str, str]:
    lat = None
    lon = None
    for c in _LAT_CANDS:
        if c in ds.coords or c in ds.dims:
            lat = c; break
    for c in _LON_CANDS:
        if c in ds.coords or c in ds.dims:
            lon = c; break
    if lat is None or lon is None:
        # 兜底从任意变量找
        if len(ds.data_vars) > 0:
            dv = ds[list(ds.data_vars)[0]]
            for c in _LAT_CANDS:
                if c in dv.coords:
                    lat = lat or c
                    break
            for c in _LON_CANDS:
                if c in dv.coords:
                    lon = lon or c
                    break
    if lat is None or lon is None:
        raise ValueError("未找到经纬度坐标名")
    return lat, lon


def _value_stats(da: xr.DataArray):
    vals = da.values
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return dict(vmin=np.nan, vmax=np.nan, vmean=np.nan, p95=np.nan, p5=np.nan, spread=np.nan, constant_like=True)
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    vmean = float(np.nanmean(vals))
    p5 = float(np.nanpercentile(vals, 5))
    p95 = float(np.nanpercentile(vals, 95))
    spread = p95 - p5
    constant_like = bool(spread < 1e-6)
    return dict(vmin=vmin, vmax=vmax, vmean=vmean, p95=p95, p5=p5, spread=spread, constant_like=constant_like)


def inspect_file(p: Path):
    try:
        ds = xr.open_dataset(p)
    except Exception as e:
        print(f"[INSPECT] 打开失败 {p.name}: {e}")
        return
    lat_name, lon_name = detect_lat_lon_names(ds)
    for var in ds.data_vars:
        da = ds[var]
        # 尽量 squeeze 到 2D
        extra = [d for d in da.dims if d not in (lat_name, lon_name)]
        if extra:
            da = da.mean(dim=extra, skipna=True)
            da = da.squeeze(drop=True)
        try:
            latv = da[lat_name].values
            lonv = da[lon_name].values
            lat_min, lat_max = float(np.nanmin(latv)), float(np.nanmax(latv))
            lon_min, lon_max = float(np.nanmin(lonv)), float(np.nanmax(lonv))
        except Exception:
            lat_min=lat_max=lon_min=lon_max=np.nan
        s = _value_stats(da)
        print(f"{p.name:30s} {var:22s} lat[{lat_min:6.2f},{lat_max:6.2f}] lon[{lon_min:6.2f},{lon_max:6.2f}] "
              f"min={s['vmin']:.3f} max={s['vmax']:.3f} mean={s['vmean']:.3f} spread(p95-p5)={s['spread']:.3e} const={s['constant_like']}")
    try:
        ds.close()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None, help="覆盖默认目录 ArcticRoute/data_processed/newenv")
    args = parser.parse_args()

    root = get_project_root()
    default_dir = root / "ArcticRoute" / "data_processed" / "newenv"
    target = Path(args.dir) if args.dir else default_dir

    print("[INSPECT] dir:", target)
    if not target.exists():
        print("[INSPECT] 目标目录不存在")
        return

    files = sorted(target.glob("*.nc"))
    if not files:
        print("[INSPECT] 无 .nc 文件")
        return

    print("文件名                         变量名                 范围/统计摘要")
    for p in files:
        inspect_file(p)


if __name__ == "__main__":
    main()






















