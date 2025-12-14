# -*- coding: utf-8 -*-
"""
将原始 AIS csv/parquet 转为月度 2D 栅格交通密度，并派生拥挤风险。

用法示例：
  python ArcticRoute/scripts/preprocess_ais_to_density.py --ym 202412 \
      --in-dir ArcticRoute/data_raw/ais \
      --out-dir ArcticRoute/data_processed/risk

输出：out-dir/traffic_density_YYYYMM.nc
  - 变量：traffic_density (latitude, longitude), congestion_risk (latitude, longitude)

说明：
  - 网格来源优先 env_clean.nc（与 planner/grid 对齐），若无则报错。
  - 经纬度列自动识别：lat/latitude/LAT/Latitude 与 lon/longitude/LON/Longitude。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_dilation  # 仅当存在陆地掩膜时用于安全缓冲

# -------------------------
# 网格加载：优先 env_clean.nc
# -------------------------
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


def _load_runtime_flags() -> dict:
    try:
        path = PROJECT_ROOT / "ArcticRoute" / "config" / "runtime.yaml"
        if path.exists():
            import yaml
            obj = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return (obj.get("flags") or {}) if isinstance(obj, dict) else {}
    except Exception:
        pass
    return {}


def _load_land_mask_like(H: int, W: int) -> np.ndarray | None:
    """加载与规划一致的陆地掩膜，返回 (H,W) 的 bool 数组（True=陆地）。失败返回 None。"""
    candidates = [
        PROJECT_ROOT / "ArcticRoute" / "data_processed" / "env" / "env_clean.nc",
        PROJECT_ROOT / "ArcticRoute" / "data_processed" / "env_clean.nc",
        PROJECT_ROOT / "ArcticRoute" / "data_processed" / "env" / "land_mask.nc",
    ]
    var_candidates = ["land_mask", "lm", "land", "lsm", "landsea_mask", "LSM", "ocean_mask"]
    for p in candidates:
        if not p.exists():
            continue
        try:
            with xr.open_dataset(p) as ds:
                var = next((v for v in var_candidates if v in ds.data_vars), None)
                if var is None:
                    continue
                da = ds[var]
                if "time" in da.dims:
                    da = da.isel(time=0)
                arr = da.values
                if arr.ndim >= 2:
                    a2 = arr.reshape(arr.shape[-2], arr.shape[-1]).astype(float)
                else:
                    continue
                if str(var).lower() == "ocean_mask":
                    a2 = 1.0 - a2
                a2 = (a2 > 0.5).astype(bool)
                # 最邻近重采样到 (H,W)
                si, sj = a2.shape
                if (si, sj) != (H, W):
                    yi = (np.linspace(0, si - 1, H)).astype(int)
                    xj = (np.linspace(0, sj - 1, W)).astype(int)
                    a2 = a2[yi[:, None], xj[None, :]]
                # 安全缓冲 1 像素
                try:
                    a2 = binary_dilation(a2, iterations=1)
                except Exception:
                    pass
                return a2
        except Exception:
            continue
    return None


def load_env_grid() -> tuple[np.ndarray, np.ndarray]:
    """
    返回 (lat_arr, lon_arr) 两个 1D 单调数组。
    采用 env_clean.nc 中的 latitude/longitude。
    若为降序，则自动翻转到升序并提醒。
    若经度在 0..360，则保持 0..360 范围；若在 -180..180，则保持该范围。
    """
    env_path = _find_env_clean()
    if env_path is None:
        raise FileNotFoundError("未找到 env_clean.nc（ArcticRoute/data_processed/env/env_clean.nc 或同级兜底）")
    with xr.open_dataset(env_path) as ds:
        lat_name = next((n for n in ("latitude", "lat") if n in ds.coords or n in ds.variables), None)
        lon_name = next((n for n in ("longitude", "lon") if n in ds.coords or n in ds.variables), None)
        if lat_name is None or lon_name is None:
            raise RuntimeError(f"env_clean.nc 中缺少纬度/经度坐标: coords={list(ds.coords)} vars={list(ds.data_vars)}")
        lat = ds[lat_name].values.astype(float)
        lon = ds[lon_name].values.astype(float)
    # 只接受 1D
    if lat.ndim != 1 or lon.ndim != 1:
        raise RuntimeError(f"env_clean 网格必须为 1D：lat.ndim={lat.ndim} lon.ndim={lon.ndim}")
    # 统一单调性：升序
    if lat.size >= 2 and float(lat[0]) > float(lat[-1]):
        lat = lat[::-1].copy()
        print("[GRID] latitude 为降序，已翻转为升序")
    if lon.size >= 2 and float(lon[0]) > float(lon[-1]):
        # 允许 0..360；若第一项>最后一项，可能是降序，需要翻转
        lon = lon[::-1].copy()
        print("[GRID] longitude 为降序，已翻转为升序")
    return lat, lon


# -------------------------
# AIS 列名探测
# -------------------------

def detect_lat_lon_columns(df: pd.DataFrame) -> tuple[str, str]:
    cand_lat = ["lat", "latitude", "LAT", "Latitude"]
    cand_lon = ["lon", "longitude", "LON", "Longitude"]
    lat_col = next((c for c in cand_lat if c in df.columns), None)
    lon_col = next((c for c in cand_lon if c in df.columns), None)
    if lat_col is None or lon_col is None:
        raise ValueError(f"未找到经纬度列, columns={list(df.columns)}")
    return lat_col, lon_col


# -------------------------
# 主过程
# -------------------------

def _wrap_lon_to_grid(lon_vals: np.ndarray, grid_lon: np.ndarray) -> np.ndarray:
    """
    将输入经度数组映射到与 grid_lon 一致的范围：
      - 若网格经度范围主要在 [0,360]，则映射到 [0,360)
      - 否则映射到 [-180,180)
    """
    lon_vals = lon_vals.astype(float)
    gmin, gmax = float(np.nanmin(grid_lon)), float(np.nanmax(grid_lon))
    if gmax > 180.0 + 1e-3 or gmin >= 0.0:
        # 使用 0..360
        out = np.mod(lon_vals, 360.0)
        out[out < 0] += 360.0
        return out
    # 使用 -180..180
    out = ((lon_vals + 180.0) % 360.0) - 180.0
    return out


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

    # 1) 读取 env 网格
    lat_arr, lon_arr = load_env_grid()
    H, W = int(lat_arr.size), int(lon_arr.size)
    counts = np.zeros((H, W), dtype=np.float32)

    # 2) 遍历本月所有 AIS 文件
    files = sorted(in_dir.glob(f"*{ym}*.parquet")) + sorted(in_dir.glob(f"*{ym}*.csv"))
    if not files:
        print(f"[AIS] 未找到 {ym} 的 AIS 文件于 {in_dir}, 跳过")
        return

    # 运行时开关与陆地掩膜
    flags = _load_runtime_flags()
    do_filter = bool(flags.get("ais_filter_enabled", True))
    lm = _load_land_mask_like(H, W) if do_filter else None
    if lm is not None:
        print(f"[AIS_FILTER] 使用 land_mask 进行陆地点过滤；shape={lm.shape}; frac_land={float(lm.mean()):.4f}")

    # BBOX 范围（基于网格坐标）
    lat_min, lat_max = float(lat_arr.min()), float(lat_arr.max())
    lon_min, lon_max = float(lon_arr.min()), float(lon_arr.max())

    total_all = on_land_total = oob_total = 0

    for fp in files:
        print("[AIS] 读取", fp)
        try:
            if fp.suffix.lower() == ".parquet":
                df = pd.read_parquet(fp)
            else:
                df = pd.read_csv(fp)
        except Exception as e:
            print(f"[AIS] 读取失败 {fp}: {e}")
            continue
        if df is None or len(df) == 0:
            continue

        try:
            lat_col, lon_col = detect_lat_lon_columns(df)
        except Exception as e:
            print(f"[AIS] 列名识别失败 {fp}: {e}")
            continue

        lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy()
        lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy()

        # 丢掉 NaN
        m = np.isfinite(lat) & np.isfinite(lon)
        if not np.any(m):
            continue
        lat = lat[m].astype(float)
        lon = lon[m].astype(float)
        total_all += int(lat.size)

        # 经纬度粗过滤
        m2 = (lat >= -90.0) & (lat <= 90.0)
        lat = lat[m2]
        lon = lon[m2]
        if lat.size == 0:
            continue

        # 经度映射到网格同一范围
        lon = _wrap_lon_to_grid(lon, lon_arr)

        # BBOX 过滤（严格在网格范围内；越界直接丢弃）
        m_bbox = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
        oob_total += int((~m_bbox).sum())
        lat = lat[m_bbox]
        lon = lon[m_bbox]
        if lat.size == 0:
            continue

        # 最近邻映射到网格（升序的 1D 单调数组）
        lat_idx = np.searchsorted(lat_arr, lat, side="right") - 1
        lon_idx = np.searchsorted(lon_arr, lon, side="right") - 1
        lat_idx = np.clip(lat_idx, 0, H - 1)
        lon_idx = np.clip(lon_idx, 0, W - 1)

        # 陆地过滤（可选）
        if lm is not None:
            on_land_mask = lm[lat_idx, lon_idx]
            on_land_total += int(on_land_mask.sum())
            keep = ~on_land_mask
            if not np.any(keep):
                continue
            lat_idx = lat_idx[keep]
            lon_idx = lon_idx[keep]

        # 向量化累计计数
        flat = (lat_idx.astype(np.int64) * W + lon_idx.astype(np.int64))
        binc = np.bincount(flat, minlength=H * W).astype(np.float32)
        counts += binc.reshape(H, W)

    if total_all > 0:
        print(f"[AIS_FILTER] total={total_all} on_land={on_land_total} ({on_land_total/max(1,total_all):.2%}) out_of_bbox={oob_total}")

    # 3) 构造 DataArray 并归一化为 0..1 的拥挤程度
    counts_da = xr.DataArray(
        counts,
        dims=("latitude", "longitude"),
        coords={"latitude": lat_arr, "longitude": lon_arr},
        name="traffic_density",
    )

    # 简单对数 + 归一
    logc = np.log1p(counts_da)
    denom = float(logc.max() - logc.min())
    if not np.isfinite(denom) or denom <= 0:
        denom = 1e-6
    norm = (logc - float(logc.min())) / denom
    congestion = norm.astype("float32").rename("congestion_risk")

    ds = xr.Dataset({"traffic_density": counts_da.astype("float32"), "congestion_risk": congestion})
    out_path = out_dir / f"traffic_density_{ym}.nc"
    ds.to_netcdf(out_path)
    print("[AIS] 写出:", out_path)
    try:
        print("[AIS] traffic_density range", float(counts_da.min()), float(counts_da.max()))
        print("[AIS] congestion_risk range", float(congestion.min()), float(congestion.max()))
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

