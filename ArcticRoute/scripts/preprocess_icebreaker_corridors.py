# -*- coding: utf-8 -*-
"""
基于破冰船 AIS 轨迹生成“安全走廊字段”（icebreaker_corridor），用于在成本构建中对冰险折扣。

用法示例：
  python ArcticRoute/scripts/preprocess_icebreaker_corridors.py --ym 202412 \
      --in-dir ArcticRoute/data_raw/ais \
      --out-dir ArcticRoute/data_processed/risk

说明：
  - 仅筛选船型为破冰船的 AIS 记录（通过常见列 ship_type/shiptype/type/vessel_type 等模糊匹配）。
  - 将经纬度映射到 env_clean.nc 的 1D 网格，统计密度并归一化到 0..1。
  - 输出变量 icebreaker_corridor (latitude, longitude)。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import re
import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _find_env_clean() -> Path | None:
    candidates = [
        PROJECT_ROOT / "ArcticRoute" / "data_processed" / "env" / "env_clean.nc",
        PROJECT_ROOT / "ArcticRoute" / "data_processed" / "env_clean.nc",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_env_grid() -> tuple[np.ndarray, np.ndarray]:
    env_path = _find_env_clean()
    if env_path is None:
        raise FileNotFoundError("未找到 env_clean.nc（ArcticRoute/data_processed/env/env_clean.nc 或同级兜底）")
    with xr.open_dataset(env_path) as ds:
        lat_name = next((n for n in ("latitude", "lat") if n in ds.coords or n in ds.variables), None)
        lon_name = next((n for n in ("longitude", "lon") if n in ds.coords or n in ds.variables), None)
        if lat_name is None or lon_name is None:
            raise RuntimeError("env_clean.nc 缺少纬度/经度坐标")
        lat = ds[lat_name].values.astype(float)
        lon = ds[lon_name].values.astype(float)
    if lat.ndim != 1 or lon.ndim != 1:
        raise RuntimeError("env_clean 网格必须为 1D")
    if float(lat[0]) > float(lat[-1]):
        lat = lat[::-1].copy()
        print("[GRID] latitude 为降序，已翻转为升序")
    if float(lon[0]) > float(lon[-1]):
        lon = lon[::-1].copy()
        print("[GRID] longitude 为降序，已翻转为升序")
    return lat, lon


def detect_lat_lon_columns(df: pd.DataFrame) -> tuple[str, str]:
    cand_lat = ["lat", "latitude", "LAT", "Latitude"]
    cand_lon = ["lon", "longitude", "LON", "Longitude"]
    lat_col = next((c for c in cand_lat if c in df.columns), None)
    lon_col = next((c for c in cand_lon if c in df.columns), None)
    if lat_col is None or lon_col is None:
        raise ValueError(f"未找到经纬度列, columns={list(df.columns)}")
    return lat_col, lon_col


def is_icebreaker_row(df: pd.DataFrame) -> np.ndarray:
    """
    返回布尔掩膜，标记破冰船记录。
    规则：
      - 在 ship_type/shiptype/type/vessel_type/ais_type 字段中包含 'ice' 与 'break'（不区分大小写）即认为是破冰船。
      - 数值类型的 ShipTypeCode 无标准映射，此处不作强约束（可后续扩展）。
    """
    cols = [c for c in df.columns if c.lower() in ("ship_type", "shiptype", "type", "vessel_type", "ais_type", "ais_type_summary")]
    if not cols:
        return np.zeros(len(df), dtype=bool)
    mask = np.zeros(len(df), dtype=bool)
    pat = re.compile(r"ice\s*breaker|icebreaker|ice-?break", re.IGNORECASE)
    for c in cols:
        try:
            v = df[c].astype(str).fillna("")
            mask |= v.str.contains(pat, regex=True, na=False).values
        except Exception:
            continue
    return mask


def _wrap_lon_to_grid(lon_vals: np.ndarray, grid_lon: np.ndarray) -> np.ndarray:
    gmin, gmax = float(np.nanmin(grid_lon)), float(np.nanmax(grid_lon))
    if gmax > 180.0 + 1e-3 or gmin >= 0.0:
        out = np.mod(lon_vals, 360.0)
        out[out < 0] += 360.0
        return out
    return ((lon_vals + 180.0) % 360.0) - 180.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ym", required=True, help="YYYYMM")
    ap.add_argument("--in-dir", default=str(PROJECT_ROOT / "ArcticRoute" / "data_raw" / "ais"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "ArcticRoute" / "data_processed" / "risk"))
    args = ap.parse_args()

    ym = str(args.ym)
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lat_arr, lon_arr = load_env_grid()
    H, W = int(lat_arr.size), int(lon_arr.size)
    counts = np.zeros((H, W), dtype=np.float32)

    files = sorted(in_dir.glob(f"*{ym}*.parquet")) + sorted(in_dir.glob(f"*{ym}*.csv"))
    if not files:
        print(f"[IB] 未找到 {ym} 的 AIS 文件于 {in_dir}, 跳过")
        return

    for fp in files:
        print("[IB] 读取", fp)
        try:
            if fp.suffix.lower() == ".parquet":
                df = pd.read_parquet(fp)
            else:
                df = pd.read_csv(fp)
        except Exception as e:
            print(f"[IB] 读取失败 {fp}: {e}")
            continue
        if df is None or len(df) == 0:
            continue

        # 破冰船筛选
        try:
            m_ice = is_icebreaker_row(df)
            if not np.any(m_ice):
                continue
            df = df.loc[m_ice]
        except Exception as e:
            print(f"[IB] 破冰船筛选失败 {fp}: {e}")
            continue

        try:
            lat_col, lon_col = detect_lat_lon_columns(df)
        except Exception as e:
            print(f"[IB] 列名识别失败 {fp}: {e}")
            continue

        lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy()
        lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy()
        m = np.isfinite(lat) & np.isfinite(lon)
        if not np.any(m):
            continue
        lat = lat[m].astype(float)
        lon = lon[m].astype(float)
        m2 = (lat >= -90.0) & (lat <= 90.0)
        lat, lon = lat[m2], lon[m2]
        if lat.size == 0:
            continue

        lon = _wrap_lon_to_grid(lon, lon_arr)
        lat_idx = np.searchsorted(lat_arr, lat, side="right") - 1
        lon_idx = np.searchsorted(lon_arr, lon, side="right") - 1
        lat_idx = np.clip(lat_idx, 0, H - 1)
        lon_idx = np.clip(lon_idx, 0, W - 1)
        flat = (lat_idx.astype(np.int64) * W + lon_idx.astype(np.int64))
        binc = np.bincount(flat, minlength=H * W).astype(np.float32)
        counts += binc.reshape(H, W)

    # 归一化
    da = xr.DataArray(
        counts,
        dims=("latitude", "longitude"),
        coords={"latitude": lat_arr, "longitude": lon_arr},
        name="icebreaker_corridor_raw",
    )
    logc = np.log1p(da)
    denom = float(logc.max() - logc.min())
    if not np.isfinite(denom) or denom <= 0:
        denom = 1e-6
    norm = (logc - float(logc.min())) / denom
    corridor = norm.astype("float32").rename("icebreaker_corridor")

    ds = xr.Dataset({"icebreaker_corridor": corridor})
    out_path = out_dir / f"icebreaker_corridor_{ym}.nc"
    ds.to_netcdf(out_path)
    print("[IB] 写出:", out_path)
    try:
        print("[IB] corridor range", float(corridor.min()), float(corridor.max()))
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

