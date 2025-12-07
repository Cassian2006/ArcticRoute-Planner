from __future__ import annotations

import numpy as np
import xarray as xr

from ArcticRoute.core.online.blend import blend_components, fuse_live  # REUSE


def _mk_da(shape=(2, 8, 8), seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange(shape[0])
    y = np.arange(shape[1])
    x = np.arange(shape[2])
    lat = np.linspace(60.0, 80.0, shape[1])
    lon = np.linspace(-40.0, 40.0, shape[2])
    arr = rng.random(shape)
    da = xr.DataArray(arr, dims=("time","y","x"), coords={"time": time, "y": y, "x": x, "lat": ("y", lat), "lon": ("x", lon)}, name="risk")
    return da


def test_fuse_live_bounds_and_time_align():
    # 组件A含time，组件B为静态（无time），应广播
    a = _mk_da(shape=(3, 10, 12), seed=42)
    b = _mk_da(shape=(1, 10, 12), seed=7).isel(time=0)
    comp = {"ice": a, "wave": b}
    blend = blend_components(comp, conf={"ice": 0.8, "wave": 0.6}, norm="quantile")  # REUSE
    fused = fuse_live(blend, method="stacking")
    v = np.asarray(fused.values)
    assert np.nanmin(v) >= 0.0 - 1e-8
    assert np.nanmax(v) <= 1.0 + 1e-8
    # time 维存在且长度与 A 相同
    assert "time" in fused.dims
    assert int(fused.sizes["time"]) == int(a.sizes["time"])  # B 被广播

