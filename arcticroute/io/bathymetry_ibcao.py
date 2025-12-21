from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


DEPTH_VAR_KEYWORDS = ["depth", "z", "elevation", "band1"]
LAT_NAMES = ["lat", "latitude", "y"]
LON_NAMES = ["lon", "longitude", "x"]


def detect_depth_var(ds: xr.Dataset) -> str:
    for name in ds.data_vars:
        name_lower = name.lower()
        if any(key in name_lower for key in DEPTH_VAR_KEYWORDS):
            return name
    vars_list = list(ds.data_vars.keys())
    raise ValueError(f"depth variable not found; vars={vars_list}")


def _find_coord(ds: xr.Dataset, names: list[str]) -> xr.DataArray | None:
    for name in names:
        if name in ds.coords:
            return ds.coords[name]
        if name in ds.variables:
            return ds[name]
    return None


def _interp_with_xarray(depth: xr.DataArray, grid_lat: np.ndarray, grid_lon: np.ndarray) -> np.ndarray | None:
    lat = _find_coord(depth.to_dataset(name="depth"), LAT_NAMES)
    lon = _find_coord(depth.to_dataset(name="depth"), LON_NAMES)
    if lat is None or lon is None:
        return None
    if lat.ndim != 1 or lon.ndim != 1:
        return None

    target_lat = xr.DataArray(grid_lat, dims=("y", "x"))
    target_lon = xr.DataArray(grid_lon, dims=("y", "x"))
    try:
        aligned = depth.interp({lat.name: target_lat, lon.name: target_lon}, method="nearest")
        return np.asarray(aligned.values)
    except Exception:
        return None


def _nearest_neighbor(
    depth_vals: np.ndarray,
    src_lat2d: np.ndarray,
    src_lon2d: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
) -> np.ndarray:
    ny, nx = grid_lat.shape
    output = np.full((ny, nx), np.nan, dtype=float)
    flat_vals = depth_vals.ravel()
    flat_lat = src_lat2d.ravel()
    flat_lon = src_lon2d.ravel()

    for iy in range(ny):
        for ix in range(nx):
            lat = grid_lat[iy, ix]
            lon = grid_lon[iy, ix]
            d2 = (flat_lat - lat) ** 2 + (flat_lon - lon) ** 2
            idx = int(np.argmin(d2))
            output[iy, ix] = flat_vals[idx]
    return output


def load_depth_to_grid(nc_path: str, grid: Any) -> tuple[np.ndarray, dict[str, Any]]:
    ds = xr.open_dataset(nc_path)
    try:
        depth_var = detect_depth_var(ds)
        depth_da = ds[depth_var]
        original_shape = tuple(int(s) for s in depth_da.shape)

        aligned = _interp_with_xarray(depth_da, grid.lat2d, grid.lon2d)
        resampled = False
        if aligned is None:
            lat = _find_coord(ds, LAT_NAMES)
            lon = _find_coord(ds, LON_NAMES)
            if lat is None or lon is None:
                raise ValueError("lat/lon coordinates not found in dataset")
            if lat.ndim == 1 and lon.ndim == 1:
                src_lon2d, src_lat2d = np.meshgrid(lon.values, lat.values)
            elif lat.ndim == 2 and lon.ndim == 2:
                src_lat2d = np.asarray(lat.values)
                src_lon2d = np.asarray(lon.values)
            else:
                raise ValueError("lat/lon coordinates have unsupported dimensions")

            aligned = _nearest_neighbor(
                np.asarray(depth_da.values),
                src_lat2d,
                src_lon2d,
                grid.lat2d,
                grid.lon2d,
            )
            resampled = True
        else:
            if aligned.shape != grid.shape():
                resampled = True

        depth_grid = np.asarray(aligned, dtype=float)
        finite = np.isfinite(depth_grid)
        meta = {
            "source_path": str(Path(nc_path)),
            "original_shape": original_shape,
            "resampled": resampled or (original_shape != grid.shape()),
            "min": float(np.nanmin(depth_grid)) if np.any(finite) else None,
            "max": float(np.nanmax(depth_grid)) if np.any(finite) else None,
            "mean": float(np.nanmean(depth_grid)) if np.any(finite) else None,
            "nan_fraction": float(1.0 - (np.sum(finite) / depth_grid.size)) if depth_grid.size else 1.0,
            "depth_sign_assumption": "negative_is_deeper",
        }
        return depth_grid, meta
    finally:
        ds.close()
