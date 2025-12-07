from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

from ArcticRoute.core.prior.geo import to_xy
from ArcticRoute.cache.index_util import register_artifact

ROOT = Path(__file__).resolve().parents[2]
CFG_DIR = ROOT / "ArcticRoute" / "config"
OUT_AIS = ROOT / "ArcticRoute" / "data_processed" / "ais"
OUT_PRIOR = ROOT / "ArcticRoute" / "data_processed" / "prior"
ENV_CLEAN = ROOT / "ArcticRoute" / "data_processed" / "env_clean.nc"


@dataclass
class RasterConfig:
    ym: str
    method: str = "transformer"
    band_quantile: float = 0.75
    dry_run: bool = False


def _load_centerlines(ym: str) -> Dict[str, Any]:
    # 首选路径：reports/phaseE/center/
    path1 = ROOT / "reports" / "phaseE" / "center" / f"prior_centerlines_{ym}.geojson"
    if path1.exists():
        with open(path1, "r", encoding="utf-8") as f:
            return json.load(f)
    # 备选路径：data_processed/prior/centerlines/
    alt = OUT_PRIOR / "centerlines" / f"prior_centerlines_{ym}.geojson"
    if alt.exists():
        with open(alt, "r", encoding="utf-8") as f:
            return json.load(f)
    # 两处都不存在时，保持原有报错文案（指向首选路径）
    raise FileNotFoundError(f"缺少中心线: {path1}")


def _load_grid_coords() -> Tuple[np.ndarray, np.ndarray]:
    """返回 (lat[y,x], lon[y,x])。优先 grid_spec.json；否则从 env_clean.nc 读取 lat/lon 变量。"""
    spec = CFG_DIR / "grid_spec.json"
    if spec.exists():
        try:
            spec_obj = json.loads(spec.read_text(encoding="utf-8"))
            # 兼容两种：1) 2D lat/lon；2) 1D 向量可网格化
            if isinstance(spec_obj.get("lat"), list) and isinstance(spec_obj.get("lon"), list):
                lat = np.array(spec_obj["lat"])  # type: ignore
                lon = np.array(spec_obj["lon"])  # type: ignore
                if lat.ndim == 2 and lon.ndim == 2 and lat.shape == lon.shape:
                    return lat.astype(float), lon.astype(float)
                if lat.ndim == 1 and lon.ndim == 1:
                    Lon, Lat = np.meshgrid(lon.astype(float), lat.astype(float))
                    return Lat, Lon
        except Exception:
            pass
    # fallback: env_clean.nc
    if xr is None:
        raise ImportError("需要 xarray 以从 env_clean.nc 读取网格")
    if not ENV_CLEAN.exists():
        raise FileNotFoundError(f"缺少网格参考: {ENV_CLEAN}")
    ds = xr.open_dataset(ENV_CLEAN)
    # 尝试变量名
    lat_name = None
    lon_name = None
    for cand in ("lat", "latitude"):
        if cand in ds.variables:
            lat_name = cand
            break
    for cand in ("lon", "longitude"):
        if cand in ds.variables:
            lon_name = cand
            break
    if lat_name and lon_name:
        lat = ds[lat_name].values
        lon = ds[lon_name].values
        if lat.ndim == 2 and lon.ndim == 2:
            return lat.astype(float), lon.astype(float)
    # 无 lat/lon 变量则无法稳健推导
    raise ValueError("env_clean.nc 中未发现 2D lat/lon 变量；请提供 grid_spec.json")


def _project_points(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H, W = lat.shape
    xs = np.zeros_like(lat, dtype=float)
    ys = np.zeros_like(lat, dtype=float)
    for i in range(H):
        for j in range(W):
            try:
                x, y = to_xy(float(lat[i, j]), float(lon[i, j]))
            except Exception:
                x, y = np.nan, np.nan
            xs[i, j] = x
            ys[i, j] = y
    return xs, ys


def _polyline_min_distance_xy(px: float, py: float, coords_xy: List[Tuple[float, float]]) -> float:
    """点到折线最短距离（平面），米。"""
    dmin = float("inf")
    for k in range(len(coords_xy) - 1):
        x1, y1 = coords_xy[k]
        x2, y2 = coords_xy[k + 1]
        vx, vy = x2 - x1, y2 - y1
        wx, wy = px - x1, py - y1
        vv = vx * vx + vy * vy
        t = 0.0 if vv == 0 else max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
        projx = x1 + t * vx
        projy = y1 + t * vy
        dx = px - projx
        dy = py - projy
        d = (dx * dx + dy * dy) ** 0.5
        if d < dmin:
            dmin = d
    return dmin if np.isfinite(dmin) else float("inf")


def export_prior_raster(ym: str, method: str = "transformer", band_quantile: float = 0.75, dry_run: bool = False) -> Path:
    """将中心线/带宽转为 P_prior 与 PriorPenalty 的网格栅格（与 P1 网格一致）。"""
    cl = _load_centerlines(ym)
    feats = cl.get("features", [])
    lat, lon = _load_grid_coords()  # [H,W]
    H, W = lat.shape

    # 预投影网格点
    grid_x, grid_y = _project_points(lat, lon)

    # 初始化概率栅格
    P = np.zeros((H, W), dtype=np.float32)

    # 逐簇计算
    for ft in feats:
        props = ft.get("properties", {}) or {}
        coords = ft.get("geometry", {}).get("coordinates", []) or []
        if not coords:
            continue
        # 折线坐标投影
        line_xy: List[Tuple[float, float]] = []
        for (LON, LAT) in coords:
            try:
                x, y = to_xy(float(LAT), float(LON))
                line_xy.append((float(x), float(y)))
            except Exception:
                continue
        if len(line_xy) < 2:
            continue
        # 局部带宽：若 properties 提供 per-vertex bw_profile 则使用；否则用 bw_stats[quantile]
        bw_sigma = None
        if "bw_profile" in props and isinstance(props.get("bw_profile"), list) and len(props["bw_profile"]) >= 2:
            # 取最近折线点索引对应的带宽（简化）
            bw_profile = np.array(props["bw_profile"], dtype=float)
        else:
            # 使用全局统计作为常数 sigma（米）
            bw_stats = props.get("bw_stats") or {}
            # 选择与 band_quantile 一致的统计或使用 p75
            if band_quantile >= 0.9 and "p90" in bw_stats:
                bw_sigma = float(bw_stats.get("p90", 1000.0))
            elif band_quantile >= 0.75 and "p75" in bw_stats:
                bw_sigma = float(bw_stats.get("p75", 1000.0))
            elif "p50" in bw_stats:
                bw_sigma = float(bw_stats.get("p50", 1000.0))
            else:
                bw_sigma = float(bw_stats.get("mean", 1000.0) or 1000.0)
            if not np.isfinite(bw_sigma) or bw_sigma <= 0:
                bw_sigma = 1000.0
            bw_profile = None  # type: ignore

        # 扫描全图（如需加速，可限制在折线包围盒的扩展区域）
        # 计算 P_i 并取 max
        for i in range(H):
            for j in range(W):
                px = float(grid_x[i, j])
                py = float(grid_y[i, j])
                if not np.isfinite(px) or not np.isfinite(py):
                    continue
                d = _polyline_min_distance_xy(px, py, line_xy)  # 米
                # 选择 sigma
                if bw_profile is not None:
                    # 选择最近顶点索引（简化近似）
                    # 找最近线段端点
                    idx_min = 0
                    dmin = float("inf")
                    for k, (xk, yk) in enumerate(line_xy):
                        dd = (px - xk) ** 2 + (py - yk) ** 2
                        if dd < dmin:
                            dmin = dd; idx_min = k
                    sigma = float(bw_profile[min(idx_min, len(bw_profile) - 1)])
                    if not np.isfinite(sigma) or sigma <= 0:
                        sigma = bw_sigma if bw_sigma is not None else 1000.0
                else:
                    sigma = bw_sigma if bw_sigma is not None else 1000.0
                val = float(np.exp(- (d * d) / (2.0 * sigma * sigma)))
                if val > P[i, j]:
                    P[i, j] = val

    # 归一化到 [0,1]
    P = np.clip(P, 0.0, 1.0)
    vmax = float(np.max(P)) if np.isfinite(P).any() else 0.0
    if vmax > 0:
        P = (P / vmax).astype(np.float32)
    PriorPenalty = (1.0 - P).astype(np.float32)

    # 写 NetCDF
    out_dir = OUT_PRIOR
    out_dir.mkdir(parents=True, exist_ok=True)
    nc_path = out_dir / f"prior_transformer_{ym}.nc"

    if xr is None:
        raise ImportError("需要 xarray/netCDF4 以写 NetCDF")

    ds = xr.Dataset({
        "P_prior": (("y", "x"), P),
        "PriorPenalty": (("y", "x"), PriorPenalty),
        "lat": (("y", "x"), lat.astype(np.float32)),
        "lon": (("y", "x"), lon.astype(np.float32)),
    }, coords={
        "y": np.arange(lat.shape[0], dtype=np.int32),
        "x": np.arange(lat.shape[1], dtype=np.int32),
    })
    ds["lat"].attrs.update({"long_name": "latitude", "units": "degrees_north"})
    ds["lon"].attrs.update({"long_name": "longitude", "units": "degrees_east"})
    ds["P_prior"].attrs.update({
        "long_name": "Transformer prior probability",
        "units": "1",
        "method": method,
        "band_quantile": float(band_quantile),
        "source": f"prior_centerlines_{ym}.geojson",
    })
    ds["PriorPenalty"].attrs.update({
        "long_name": "Prior penalty (1 - P_prior)",
        "units": "1",
    })

    # 压缩与分块（确保 chunksize 不超过维度）
    H, W = lat.shape
    chunk_y = min(H, 128)
    chunk_x = min(W, 256)
    encoding = {
        "P_prior": {"zlib": True, "complevel": 4},
        "PriorPenalty": {"zlib": True, "complevel": 4},
    }

    if not dry_run:
        ds.to_netcdf(str(nc_path), engine="netcdf4", encoding=encoding)
        try:
            register_artifact(run_id=ym, kind="prior_raster", path=str(nc_path), attrs={"ym": ym, "method": method})
        except Exception:
            pass

    return nc_path


__all__ = ["RasterConfig", "export_prior_raster"]

