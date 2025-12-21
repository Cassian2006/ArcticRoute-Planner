from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from arcticroute.core.grid import Grid2D


SIT_KEYWORDS = ("thickness", "sithick", "sit", "ice_thickness", "ice_thk")
DRIFT_U_KEYWORDS = (
    "drift_u",
    "u_drift",
    "uice",
    "u_ice",
    "ice_u",
    "ice_drift_x",
    "drift_x",
    "x_drift",
    "u",
    "u_velocity",
)
DRIFT_V_KEYWORDS = (
    "drift_v",
    "v_drift",
    "vice",
    "v_ice",
    "ice_v",
    "ice_drift_y",
    "drift_y",
    "y_drift",
    "v",
    "v_velocity",
)
DRIFT_SPEED_KEYWORDS = ("drift_speed", "ice_drift_speed", "ice_speed", "speed")


def _pick_var_name(
    candidates: Iterable[str] | None, available: Iterable[str], keywords: Iterable[str]
) -> str | None:
    available_set = {name for name in available}
    if candidates:
        for name in candidates:
            if name in available_set:
                return name
    lower_map = {name.lower(): name for name in available}
    for key in keywords:
        for low_name, orig in lower_map.items():
            if key in low_name:
                return orig
    return None


def _select_time_slice(da: Any, time_index: int) -> Any:
    if hasattr(da, "dims") and "time" in da.dims:
        idx = min(time_index, da.sizes.get("time", 1) - 1)
        return da.isel(time=idx)
    return da


def _attach_latlon_coords(da: Any, ds: Any) -> Any:
    try:
        import xarray as xr  # noqa: WPS433
    except Exception:
        return da

    if not isinstance(da, xr.DataArray):
        return da

    has_lat = any(name in da.coords for name in ("lat", "latitude", "LAT"))
    has_lon = any(name in da.coords for name in ("lon", "longitude", "LON"))
    if has_lat and has_lon:
        return da

    lat_da = None
    lon_da = None
    for name in ("lat", "latitude", "LAT"):
        if name in ds.coords:
            lat_da = ds.coords[name]
            break
        if name in ds.data_vars:
            lat_da = ds[name]
            break
    for name in ("lon", "longitude", "LON"):
        if name in ds.coords:
            lon_da = ds.coords[name]
            break
        if name in ds.data_vars:
            lon_da = ds[name]
            break
    if lat_da is None or lon_da is None:
        return da

    try:
        return da.assign_coords(lat=lat_da, lon=lon_da)
    except Exception:
        return da


def align_to_grid(
    da: Any,
    grid: Grid2D | None,
) -> tuple[np.ndarray | None, str | None]:
    if da is None:
        return None, "no_data"
    try:
        arr = np.asarray(getattr(da, "values", da), dtype=float)
    except Exception as exc:
        return None, f"to_array_failed: {exc}"

    if grid is None:
        return arr, "no_grid"

    if arr.shape == grid.shape():
        return arr, None

    try:
        import xarray as xr  # noqa: WPS433
    except Exception:
        return None, "xarray_missing_for_interp"

    if not isinstance(da, xr.DataArray):
        return None, "shape_mismatch_no_coords"

    lat = None
    lon = None
    for name in ("lat", "latitude", "LAT"):
        if name in da.coords:
            lat = da.coords[name]
            break
    for name in ("lon", "longitude", "LON"):
        if name in da.coords:
            lon = da.coords[name]
            break
    if lat is None or lon is None:
        return None, "shape_mismatch_no_latlon"

    try:
        target = da.interp(
            lat=(("y", "x"), grid.lat2d),
            lon=(("y", "x"), grid.lon2d),
            method="nearest",
        )
        return np.asarray(target.values, dtype=float), None
    except Exception as exc:
        return None, f"interp_failed: {exc}"


def load_sit_from_nc(
    nc_path: str | Path,
    *,
    grid: Grid2D | None = None,
    var_candidates: Iterable[str] | None = None,
    time_index: int = 0,
) -> tuple[np.ndarray | None, dict]:
    meta = {"status": "skipped", "reason": "", "var": None, "aligned": False}
    path = Path(nc_path)
    if not path.exists():
        meta["status"] = "skipped"
        meta["reason"] = "file_missing"
        return None, meta
    try:
        import xarray as xr  # noqa: WPS433
    except Exception:
        meta["status"] = "skipped"
        meta["reason"] = "xarray_missing"
        return None, meta

    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            var_name = _pick_var_name(var_candidates, ds.data_vars, SIT_KEYWORDS)
            if var_name is None:
                meta["status"] = "skipped"
                meta["reason"] = "no_sit_variable"
                return None, meta
            da = _select_time_slice(ds[var_name], time_index)
            da = _attach_latlon_coords(da, ds)
            aligned, reason = align_to_grid(da, grid)
            if aligned is None:
                meta["status"] = "skipped"
                meta["reason"] = reason or "align_failed"
                return None, meta
            meta["status"] = "ok"
            meta["var"] = var_name
            meta["aligned"] = reason is None
            if reason:
                meta["reason"] = reason
            return aligned, meta
    except Exception as exc:
        meta["status"] = "skipped"
        meta["reason"] = f"open_failed: {exc}"
        return None, meta


def load_ice_drift_from_nc(
    nc_path: str | Path,
    *,
    grid: Grid2D | None = None,
    u_candidates: Iterable[str] | None = None,
    v_candidates: Iterable[str] | None = None,
    speed_candidates: Iterable[str] | None = None,
    time_index: int = 0,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict]:
    meta = {
        "status": "skipped",
        "reason": "",
        "u_var": None,
        "v_var": None,
        "speed_var": None,
        "aligned": False,
    }
    path = Path(nc_path)
    if not path.exists():
        meta["status"] = "skipped"
        meta["reason"] = "file_missing"
        return None, None, None, meta
    try:
        import xarray as xr  # noqa: WPS433
    except Exception:
        meta["status"] = "skipped"
        meta["reason"] = "xarray_missing"
        return None, None, None, meta

    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            var_names = list(ds.data_vars)
            u_name = _pick_var_name(u_candidates, var_names, DRIFT_U_KEYWORDS)
            v_name = _pick_var_name(v_candidates, var_names, DRIFT_V_KEYWORDS)
            speed_name = _pick_var_name(speed_candidates, var_names, DRIFT_SPEED_KEYWORDS)

            u_da = _select_time_slice(ds[u_name], time_index) if u_name else None
            v_da = _select_time_slice(ds[v_name], time_index) if v_name else None
            speed_da = _select_time_slice(ds[speed_name], time_index) if speed_name else None

            if u_da is not None:
                u_da = _attach_latlon_coords(u_da, ds)
            if v_da is not None:
                v_da = _attach_latlon_coords(v_da, ds)
            if speed_da is not None:
                speed_da = _attach_latlon_coords(speed_da, ds)

            u_aligned = v_aligned = speed_aligned = None
            if u_da is not None and v_da is not None:
                u_aligned, u_reason = align_to_grid(u_da, grid)
                v_aligned, v_reason = align_to_grid(v_da, grid)
                if u_aligned is None or v_aligned is None:
                    meta["status"] = "skipped"
                    meta["reason"] = u_reason or v_reason or "align_failed"
                    return None, None, None, meta
                speed_aligned = np.sqrt(u_aligned ** 2 + v_aligned ** 2)
                meta["aligned"] = (u_reason is None and v_reason is None)
            elif speed_da is not None:
                speed_aligned, reason = align_to_grid(speed_da, grid)
                if speed_aligned is None:
                    meta["status"] = "skipped"
                    meta["reason"] = reason or "align_failed"
                    return None, None, None, meta
                meta["aligned"] = reason is None
            else:
                meta["status"] = "skipped"
                meta["reason"] = "no_drift_variable"
                return None, None, None, meta

            meta["status"] = "ok"
            meta["u_var"] = u_name
            meta["v_var"] = v_name
            meta["speed_var"] = speed_name
            return u_aligned, v_aligned, speed_aligned, meta
    except Exception as exc:
        meta["status"] = "skipped"
        meta["reason"] = f"open_failed: {exc}"
        return None, None, None, meta
