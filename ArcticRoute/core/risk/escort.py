from __future__ import annotations

"""
Phase F｜护航走廊 KDE 与护航折减（真实最小实现）

- build_escort_corridor(ym) -> xarray.DataArray[name='P']
  使用 AIS 编队 episodes 的经纬点，离散到网格并做 2D 盒式平滑，归一到 [0,1]
- apply_escort(ym, eta) -> xarray.DataArray[name='risk']
  读取 R_ice 与 P_escort_corridor，计算 m = 1 - eta * P（eta<=0.3），R_ice_eff = R_ice * m

REUSE:
- 变量选择复用 fusion._pick_var；网格兜底复用 congest.encounter._infer_grid_from_any。
"""

import os
from typing import Optional, Tuple

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from ArcticRoute.io.ais_pairs import detect_convoy_episodes  # type: ignore

try:
    from pyproj import Transformer  # type: ignore
except Exception:  # pragma: no cover
    Transformer = None  # type: ignore

try:
    from sklearn.neighbors import KernelDensity  # type: ignore
except Exception:  # pragma: no cover
    KernelDensity = None  # type: ignore

RISK_DIR = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "risk")


def _grid_lat_lon(ref_da: "xr.DataArray") -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    lat = None
    lon = None
    for c in ("lat", "latitude"):
        if c in ref_da.coords:
            lat = np.asarray(ref_da.coords[c].values)
            break
    for c in ("lon", "longitude"):
        if c in ref_da.coords:
            lon = np.asarray(ref_da.coords[c].values)
            break
    return lat, lon


def _box_blur(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return arr
    # 简单累计和实现 O(N) 均值滤波
    out = arr.copy()
    # 水平
    csum = np.cumsum(out, axis=1)
    w = k
    left = np.concatenate([np.zeros((out.shape[0], 1), dtype=out.dtype), csum[:, :-w]], axis=1)
    right = csum
    out = (right - left) / float(w)
    # 垂直
    csum2 = np.cumsum(out, axis=0)
    top = np.concatenate([np.zeros((1, out.shape[1]), dtype=out.dtype), csum2[:-w, :]], axis=0)
    bottom = csum2
    out = (bottom - top) / float(w)
    return out


def _estimate_kernel_size_km(lat: np.ndarray, lon: np.ndarray, bandwidth_km: float = 20.0) -> int:
    # 估算网格分辨率（km/像素）
    if lat is None or lon is None:
        return 5
    try:
        from math import radians, cos
        dlat = abs(float(lat[1] - lat[0])) if lat.size >= 2 else 0.1
        dlon = abs(float(lon[1] - lon[0])) if lon.size >= 2 else 0.2
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * cos(radians(float(lat[int(lat.size/2)]))) if lat.size else 60.0
        dy_km = max(1e-6, dlat * km_per_deg_lat)
        dx_km = max(1e-6, dlon * km_per_deg_lon)
        px = int(max(1, round(bandwidth_km / dx_km)))
        py = int(max(1, round(bandwidth_km / dy_km)))
        return int(max(1, (px + py) // 2))
    except Exception:
        return 5


def build_escort_corridor(ym: str) -> "xr.DataArray":
    if xr is None:
        raise RuntimeError("xarray is required")

    # 读取参考网格（优先冰层）
    p_rice = os.path.join(RISK_DIR, f"risk_ice_{ym}.nc")
    ref_da = None
    try:
        if os.path.exists(p_rice):
            ds = xr.open_dataset(p_rice)
            from ArcticRoute.core.risk.fusion import _pick_var  # REUSE
            var, _, _ = _pick_var(ds, "ice")
            if var:
                ref_da = ds[var]
    except Exception:
        ref_da = None
    if ref_da is None:
        # 兜底用融合层或网格推断
        from ArcticRoute.core.congest.encounter import _infer_grid_from_any  # REUSE
        Ty, Tx, ref = _infer_grid_from_any(ym)
        arr = np.zeros((Ty, Tx), dtype=np.float32)
        da = xr.DataArray(arr, dims=("y", "x"))
        if ref is not None:
            try:
                ref_da2 = ref[list(ref.data_vars)[0]]
                for c in ("lat", "latitude"):
                    if c in ref_da2.coords:
                        da = da.assign_coords({c: ref_da2.coords[c]})
                        break
                for c in ("lon", "longitude"):
                    if c in ref_da2.coords:
                        da = da.assign_coords({c: ref_da2.coords[c]})
                        break
            except Exception:
                pass
            finally:
                try:
                    ref.close()
                except Exception:
                    pass
        ref_da = da

    lat, lon = _grid_lat_lon(ref_da)
    Ty = int(ref_da.sizes.get("y") or ref_da.sizes.get("lat") or ref_da.sizes.get("latitude"))
    Tx = int(ref_da.sizes.get("x") or ref_da.sizes.get("lon") or ref_da.sizes.get("longitude"))

    # episodes
    episodes = detect_convoy_episodes(ym)
    if not episodes:
        H = np.zeros((Ty, Tx), dtype=np.float32)
        P = xr.DataArray(H, dims=ref_da.dims, coords=ref_da.coords, name="P")
        P.attrs.update({"long_name": "Escort corridor probability (KDE)", "source": "escort_kde", "samples": 0})
        out_path = os.path.join(RISK_DIR, f"P_escort_corridor_{ym}.nc")
        os.makedirs(RISK_DIR, exist_ok=True)
        P.to_dataset().to_netcdf(out_path)
        return P

    # 收集样本点
    lats_all: list[float] = []
    lons_all: list[float] = []
    for ep in episodes:
        lats = ep.get("lats") or []
        lons = ep.get("lons") or []
        if not lats or not lons:
            continue
        lats_all.extend([float(v) for v in lats])
        lons_all.extend([float(v) for v in lons])

    if Transformer is None or KernelDensity is None or lat is None or lon is None:
        # 回退：若依赖缺失或坐标缺失，写零场
        H = np.zeros((Ty, Tx), dtype=np.float32)
        P = xr.DataArray(H, dims=ref_da.dims, coords=ref_da.coords, name="P")
        P.attrs.update({"long_name": "Escort corridor probability (KDE)", "source": "escort_kde_fallback", "samples": len(lats_all)})
        out_path = os.path.join(RISK_DIR, f"P_escort_corridor_{ym}.nc")
        os.makedirs(RISK_DIR, exist_ok=True)
        P.to_dataset().to_netcdf(out_path)
        return P

    transformer = Transformer.from_crs(4326, 3413, always_xy=True)
    xs, ys = transformer.transform(np.asarray(lons_all), np.asarray(lats_all))
    samples = np.stack([xs, ys], axis=1)

    # 带宽估计：20km（可按样本方差自适应，这里固定以满足文档建议）
    bandwidth = 20_000.0
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(samples)

    # 将网格经纬度投影为米，做评估
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    Xg, Yg = transformer.transform(lon_grid, lat_grid)
    grid_points = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
    log_dens = kde.score_samples(grid_points)
    dens = np.exp(log_dens).reshape(len(lat), len(lon))

    # 归一化到 [0,1]
    if np.max(dens) > 0:
        dens = dens / float(np.max(dens))
    P = xr.DataArray(dens.astype(np.float32), dims=ref_da.dims, coords=ref_da.coords, name="P")
    P.attrs.update({"long_name": "Escort corridor probability (KDE)", "source": "escort_kde", "bandwidth_m": bandwidth, "samples": len(lats_all)})

    os.makedirs(RISK_DIR, exist_ok=True)
    out_path = os.path.join(RISK_DIR, f"P_escort_corridor_{ym}.nc")
    P.to_dataset().to_netcdf(out_path)
    return P


def _load_risk_ice(ym: str) -> "xr.DataArray":
    # REUSE 路径与变量选择
    p_rice = os.path.join(RISK_DIR, f"risk_ice_{ym}.nc")
    da_ice = None
    ds_ice = None
    if os.path.exists(p_rice):
        ds_ice = xr.open_dataset(p_rice)
        from ArcticRoute.core.risk.fusion import _pick_var
        var_ice, _, _ = _pick_var(ds_ice, 'ice')
        if var_ice:
            da_ice = ds_ice[var_ice]
    if da_ice is None:
        from ArcticRoute.core.risk.fuse_prep import find_layer_paths
        paths = find_layer_paths(ym)
        p_rice2 = paths.get("ice")
        if p_rice2 and os.path.exists(p_rice2):
            ds_ice = xr.open_dataset(p_rice2)
            from ArcticRoute.core.risk.fusion import _pick_var
            var_ice, _, _ = _pick_var(ds_ice, 'ice')
            if var_ice:
                da_ice = ds_ice[var_ice]
    if da_ice is None:
        from ArcticRoute.core.congest.encounter import _infer_grid_from_any  # REUSE grid inference
        Ty, Tx, ref = _infer_grid_from_any(ym)
        arr = np.zeros((Ty, Tx), dtype=np.float32)
        da_ice = xr.DataArray(arr, dims=("y", "x"))
        if ref is not None:
            try:
                ref_da = ref[list(ref.data_vars)[0]]
                for c in ("lat", "latitude"):
                    if c in ref_da.coords:
                        da_ice = da_ice.assign_coords({c: ref_da.coords[c]})
                        break
                for c in ("lon", "longitude"):
                    if c in ref_da.coords:
                        da_ice = da_ice.assign_coords({c: ref_da.coords[c]})
                        break
            except Exception:
                pass
            finally:
                try:
                    ref.close()  # type: ignore
                except Exception:
                    pass
    return da_ice


def apply_escort(ym: str, eta: float) -> "xr.DataArray":
    if xr is None:
        raise RuntimeError("xarray is required")

    eta = float(np.clip(eta, 0.0, 0.3))

    da_ice = _load_risk_ice(ym)
    # 读取或生成 P_escort_corridor
    p_corr = os.path.join(RISK_DIR, f"P_escort_corridor_{ym}.nc")
    if os.path.exists(p_corr):
        try:
            with xr.open_dataset(p_corr) as ds:
                P = ds["P"] if "P" in ds else ds[list(ds.data_vars)[0]]
        except Exception:
            P = build_escort_corridor(ym)
    else:
        P = build_escort_corridor(ym)

    # 对齐时间/维度（若 ice 有 time 则广播 P）
    if "time" in da_ice.dims and "time" not in P.dims:
        P = P.expand_dims({"time": da_ice.coords["time"]})  # type: ignore
        P = P.transpose(*da_ice.dims, missing_dims="ignore")
    elif "time" in P.dims and "time" not in da_ice.dims:
        da_ice = da_ice.expand_dims({"time": P.coords["time"]})  # type: ignore
        da_ice = da_ice.transpose(*P.dims, missing_dims="ignore")
    else:
        # 仅空间维，尝试 broadcast
        try:
            P = P.broadcast_like(da_ice)
        except Exception:
            pass

    m = (1.0 - eta * P.astype("float32")).clip(0.0, 1.0)
    da_eff = (da_ice.astype("float32") * m).clip(0.0, 1.0)
    da_eff.name = "risk"
    da_eff.attrs.update({
        "long_name": "Effective ice risk with escort factor",
        "source": "escort",
        "eta": float(eta),
        "m_escort_mode": "kde_corridor",
    })

    os.makedirs(RISK_DIR, exist_ok=True)
    out_path = os.path.join(RISK_DIR, f"R_ice_eff_{ym}.nc")
    da_eff.to_dataset().to_netcdf(out_path)
    return da_eff


__all__ = ["build_escort_corridor", "apply_escort"]
