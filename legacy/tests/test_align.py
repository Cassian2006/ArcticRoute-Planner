from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ArcticRoute.io.align import ensure_common_grid, align_time


def _make_rect_ds(lat_step: float = 0.25, lon_step: float = 0.25, nt: int = 4) -> xr.Dataset:
    lat = np.arange(60.0, 62.0 + 1e-9, lat_step)
    lon = np.arange(10.0, 12.0 + 1e-9, lon_step)
    time = pd.date_range("2024-01-01", periods=nt, freq="H")
    da = xr.DataArray(
        np.zeros((len(time), len(lat), len(lon))),
        dims=("time", "y", "x"),
        coords={"time": time, "y": lat, "x": lon},
        name="v",
    )
    return xr.Dataset({"v": da})


def _grid_spec_from(ds: xr.Dataset) -> dict:
    # Mimic A-03 spec minimal
    dlat = float(np.median(np.abs(np.diff(ds["y"].values)))) if ds["y"].size > 1 else None
    dlon = float(np.median(np.abs(np.diff(ds["x"].values)))) if ds["x"].size > 1 else None
    return {
        "contract": {
            "grid_resolution": {
                "lat": None,
                "lon": None,
                "lat_deg": dlat,
                "lon_deg": dlon,
                "type": "rectilinear_1d",
            }
        }
    }


def test_ensure_common_grid_returns_same_when_aligned():
    ds = _make_rect_ds(0.25, 0.25, nt=3)
    spec = _grid_spec_from(ds)
    out = ensure_common_grid(ds, spec)
    # Phase A：对齐时返回等价对象（当前实现：即原对象）
    assert out is ds


def test_align_time_returns_same_when_matching_freq():
    ds = _make_rect_ds(nt=6)
    out = align_time(ds, "H")
    assert out is ds


def test_align_time_returns_same_when_mismatch_in_phase_a():
    ds = _make_rect_ds(nt=6)
    # 即使请求 6H，Phase A 也不做实际对齐，返回原对象
    out = align_time(ds, "6H")
    assert out is ds
