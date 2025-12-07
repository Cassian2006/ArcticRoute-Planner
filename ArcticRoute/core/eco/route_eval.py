from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

from ArcticRoute.core.route.metrics import haversine_m  # REUSE


def _grid_coords(da: "xr.DataArray") -> Tuple[np.ndarray, np.ndarray]:
    latn = "lat" if "lat" in da.coords else ("latitude" if "latitude" in da.coords else None)
    lonn = "lon" if "lon" in da.coords else ("longitude" if "longitude" in da.coords else None)
    if not (latn and lonn):
        raise KeyError("lat/lon not found in eco grid")
    return np.asarray(da.coords[latn].values), np.asarray(da.coords[lonn].values)


def _arr2d(da: "xr.DataArray") -> np.ndarray:
    arr = da
    if "time" in arr.dims and int(arr.sizes.get("time", 0)) > 0:
        arr = arr.isel(time=0)
    v = np.asarray(arr.values, dtype=float)
    if v.ndim == 2:
        return v
    v2 = np.squeeze(v)
    if v2.ndim > 2:
        axes = tuple(range(0, v2.ndim - 2))
        v2 = v2.mean(axis=axes)
    return v2


def eval_route_eco(
    route_lonlat: Sequence[Tuple[float, float]],
    eco_cost_nm_t: "xr.DataArray",
    ef_co2_t_per_t_fuel: float,
) -> Dict[str, Any]:
    """
    计算路线燃油与 CO2：
    - eco_cost_nm_t: 每海里燃油消耗（t/nm）
    - ef_co2_t_per_t_fuel: 排放因子（tCO2 / t 燃油）
    返回：{fuel_total_t, co2_total_t, per_segment: [{i, j, d_nm, fuel_t, co2_t}...]}
    """
    if xr is None:
        raise RuntimeError("xarray required")
    lat, lon = _grid_coords(eco_cost_nm_t)
    A = _arr2d(eco_cost_nm_t)

    per_seg: List[Dict[str, float]] = []
    fuel_total = 0.0
    for k in range(len(route_lonlat) - 1):
        lon1, lat1 = route_lonlat[k]
        lon2, lat2 = route_lonlat[k + 1]
        # nearest cell at segment start
        iy = int(np.clip(np.searchsorted(lat, lat1) - 1, 0, len(lat) - 1))
        ix = int(np.clip(np.searchsorted(lon, lon1) - 1, 0, len(lon) - 1))
        f_per_nm = float(np.nan_to_num(A[iy, ix], nan=0.0))
        d_m = haversine_m(lat1, lon1, lat2, lon2)
        d_nm = d_m / 1852.0
        fuel_t = f_per_nm * d_nm
        co2_t = fuel_t * float(ef_co2_t_per_t_fuel)
        fuel_total += fuel_t
        per_seg.append({
            "seg": int(k),
            "i": int(iy),
            "j": int(ix),
            "d_nm": float(d_nm),
            "fuel_t": float(fuel_t),
            "co2_t": float(co2_t),
        })
    co2_total = fuel_total * float(ef_co2_t_per_t_fuel)
    return {"fuel_total_t": float(fuel_total), "co2_total_t": float(co2_total), "per_segment": per_seg}


__all__ = ["eval_route_eco"]

