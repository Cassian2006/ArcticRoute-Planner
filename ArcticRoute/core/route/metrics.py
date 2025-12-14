from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dl / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(R * c)


def _grid_lat_lon(da: "xr.DataArray") -> Tuple[np.ndarray, np.ndarray]:
    if xr is None:
        raise RuntimeError("xarray required")
    lat = None; lon = None
    # 1) 优先从 coords 取
    for latn in ("lat", "latitude"):
        if latn in da.coords:
            lat = np.asarray(da.coords[latn].values)
            break
    for lonn in ("lon", "longitude"):
        if lonn in da.coords:
            lon = np.asarray(da.coords[lonn].values)
            break
    # 2) 次选从 variables 取
    if lat is None:
        for latn in ("lat", "latitude"):
            if latn in da._to_temp_dataset():  # type: ignore[attr-defined]
                lat = np.asarray(da._to_temp_dataset()[latn].values)  # type: ignore[index]
                break
    if lon is None:
        for lonn in ("lon", "longitude"):
            if lonn in da._to_temp_dataset():  # type: ignore[attr-defined]
                lon = np.asarray(da._to_temp_dataset()[lonn].values)  # type: ignore[index]
                break
    # 3) 最后回退：从标准网格加载并适配当前尺寸
    if lat is None or lon is None:
        try:
            from ArcticRoute.io.grid_index import _load_rect_grid  # REUSE
            Ty = int(da.sizes.get('y') or da.shape[-2])
            Tx = int(da.sizes.get('x') or da.shape[-1])
            lat1d, lon1d = _load_rect_grid(None)
            lat = lat1d.astype('float32')
            lon = lon1d.astype('float32')
            # 近邻重采样到目标长度
            if lat.shape[0] != Ty:
                idx_y = np.linspace(0, max(1, lat.shape[0]-1), Ty).round().astype(int)
                lat = lat[idx_y]
            if lon.shape[0] != Tx:
                idx_x = np.linspace(0, max(1, lon.shape[0]-1), Tx).round().astype(int)
                lon = lon[idx_x]
        except Exception:
            pass
    if lat is None or lon is None:
        raise KeyError("lat/latitude or lon/longitude not found")
    # 若 lat/lon 为 2D，则提取 1D 轴（lat: y 轴，lon: x 轴）
    if lat.ndim == 2:
        lat = lat[:, 0]
    if lon.ndim == 2:
        lon = lon[0, :]
    # 保证升序（searchsorted 期望）
    if lat[0] > lat[-1]:
        lat = lat[::-1]
    if lon[0] > lon[-1]:
        lon = lon[::-1]
    return lat.astype('float32'), lon.astype('float32')


def _arr2d_from_da(da: "xr.DataArray") -> np.ndarray:
    arr = da
    if "time" in arr.dims and arr.sizes.get("time", 0) > 0:
        arr = arr.isel(time=0)
    A = np.asarray(arr.values, dtype=float)
    if A.ndim == 2:
        return A
    A2 = np.squeeze(A)
    if A2.ndim > 2:
        axes = tuple(range(0, A2.ndim - 2))
        A2 = A2.mean(axis=axes)
    return A2


def integrate_field_along_path(field_da: "xr.DataArray", path_lonlat: Sequence[Tuple[float, float]]) -> float:
    lat, lon = _grid_lat_lon(field_da)
    A = _arr2d_from_da(field_da)
    total = 0.0
    for i in range(len(path_lonlat) - 1):
        lon1, lat1 = path_lonlat[i]
        lon2, lat2 = path_lonlat[i + 1]
        iy = int(np.clip(np.searchsorted(lat, lat1) - 1, 0, len(lat) - 1))
        ix = int(np.clip(np.searchsorted(lon, lon1) - 1, 0, len(lon) - 1))
        v = float(np.nan_to_num(A[iy, ix], nan=0.0))
        total += v * haversine_m(lat1, lon1, lat2, lon2)
    return total


def compute_distance_km(path_lonlat: Sequence[Tuple[float, float]]) -> float:
    dist = 0.0
    for i in range(len(path_lonlat) - 1):
        lon1, lat1 = path_lonlat[i]
        lon2, lat2 = path_lonlat[i + 1]
        dist += haversine_m(lat1, lon1, lat2, lon2)
    return dist / 1000.0


def summarize_route(path_lonlat: Sequence[Tuple[float, float]], risk: Optional["xr.DataArray"], prior_penalty: Optional["xr.DataArray"], interact: Optional["xr.DataArray"]) -> Dict[str, float]:
    out = {"distance_km": compute_distance_km(path_lonlat)}
    if risk is not None:
        out["risk_integral"] = integrate_field_along_path(risk, path_lonlat)
    if prior_penalty is not None:
        try:
            out["prior_integral"] = integrate_field_along_path(prior_penalty, path_lonlat)
        except Exception:
            pass
    if interact is not None:
        try:
            out["congest_integral"] = integrate_field_along_path(interact, path_lonlat)
        except Exception:
            pass
    return out


__all__ = ["integrate_field_along_path", "compute_distance_km", "summarize_route"]

