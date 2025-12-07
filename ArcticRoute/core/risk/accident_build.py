from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import numpy as np  # type: ignore
import xarray as xr  # type: ignore

# repo root
REPO = Path(__file__).resolve().parents[3]


def _select_month(ds: xr.Dataset, ym: str) -> xr.DataArray:
    # Expect a time dimension; select month by matching YYYYMM
    var = list(ds.data_vars)[0]
    da = ds[var]
    if "time" in da.dims:
        idx = [i for i, t in enumerate(da["time"].values) if str(t)[:7].replace("-", "") == ym]
        if idx:
            return da.isel(time=idx[0])
        # fallback: mean over time
        return da.mean("time")
    return da


def build_accident_risk(ym: str) -> Tuple[xr.DataArray, Path]:
    """
    Real-data build: derive accident risk from precomputed accident density cube if available,
    otherwise fail-soft by raising FileNotFoundError (caller may decide).
    """
    dens_time = REPO / "data_processed" / "accident_density_time.nc"
    dens_static = REPO / "data_processed" / "accident_density_static.nc"
    if dens_time.exists():
        ds = xr.open_dataset(dens_time)
        da = _select_month(ds, ym)
    elif dens_static.exists():
        ds = xr.open_dataset(dens_static)
        var = list(ds.data_vars)[0]
        da = ds[var]
    else:
        raise FileNotFoundError("accident density dataset not found")
    # normalize to [0,1]
    arr = da.values.astype(float)
    m, s = np.nanmean(arr), np.nanstd(arr)
    if s == 0 or not np.isfinite(s):
        # spread a bit to avoid degenerate std
        arr = (arr - np.nanmin(arr))
    vmin, vmax = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
    out = (arr - vmin) / max(1e-9, (vmax - vmin))
    out = np.clip(out, 0.0, 1.0)
    da_out = xr.DataArray(out, coords=da.coords, dims=da.dims, name="risk")
    out_path = REPO / "ArcticRoute" / "data_processed" / "risk" / f"risk_accident_{ym}.nc"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    da_out.to_dataset(name="risk").to_netcdf(out_path)
    return da_out, out_path

