from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, Optional
import json
import os
import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

from ArcticRoute.core.route.metrics import haversine_m  # REUSE


def _grid_lat_lon(da: "xr.DataArray") -> Tuple[np.ndarray, np.ndarray]:
    latn = "lat" if "lat" in da.coords else ("latitude" if "latitude" in da.coords else None)
    lonn = "lon" if "lon" in da.coords else ("longitude" if "longitude" in da.coords else None)
    if not (latn and lonn):
        raise RuntimeError("mask missing lat/lon")
    return np.asarray(da.coords[latn].values), np.asarray(da.coords[lonn].values)


def _arr2d(da: "xr.DataArray") -> np.ndarray:
    a = da
    if "time" in a.dims and int(a.sizes.get("time", 0)) > 0:
        a = a.isel(time=0)
    v = np.asarray(a.values, dtype=float)
    if v.ndim == 2:
        return v
    v2 = np.squeeze(v)
    if v2.ndim > 2:
        axes = tuple(range(0, v2.ndim - 2))
        v2 = v2.mean(axis=axes)
    return v2


def _load_route_lonlat(route_path: str) -> List[Tuple[float, float]]:
    data = json.loads(open(route_path, "r", encoding="utf-8").read())
    feat = (data.get("features") or [{}])[0]
    coords = (feat.get("geometry") or {}).get("coordinates") or []
    return [(float(x), float(y)) for x, y in coords]


def check_route_against_mask(route_lonlat: Sequence[Tuple[float, float]], mask_da: "xr.DataArray") -> Dict[str, Any]:
    lat, lon = _grid_lat_lon(mask_da)
    M = _arr2d(mask_da)
    violated_len_m = 0.0
    violated_segments = 0
    for i in range(len(route_lonlat) - 1):
        lon1, lat1 = route_lonlat[i]
        lon2, lat2 = route_lonlat[i + 1]
        iy = int(np.clip(np.searchsorted(lat, lat1) - 1, 0, len(lat) - 1))
        ix = int(np.clip(np.searchsorted(lon, lon1) - 1, 0, len(lon) - 1))
        v = float(np.nan_to_num(M[iy, ix], nan=0.0))
        d = haversine_m(lat1, lon1, lat2, lon2)
        if v > 0.5:
            violated_segments += 1
            violated_len_m += d
    return {
        "violated_segments": int(violated_segments),
        "violated_length_km": float(violated_len_m / 1000.0),
        "has_violation": bool(violated_segments > 0),
    }


__all__ = ["check_route_against_mask"]
