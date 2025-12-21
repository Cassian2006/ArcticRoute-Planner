from __future__ import annotations

import numpy as np
import xarray as xr

from arcticroute.core.grid import make_demo_grid
from arcticroute.io.bathymetry_ibcao import load_depth_to_grid


def test_load_depth_to_grid(tmp_path):
    lat = np.linspace(60.0, 61.0, 5)
    lon = np.linspace(-10.0, -6.0, 5)
    depth = np.linspace(-50.0, -10.0, 25).reshape(5, 5)

    ds = xr.Dataset(
        {"depth": (("lat", "lon"), depth)},
        coords={"lat": lat, "lon": lon},
    )
    path = tmp_path / "bathy.nc"
    ds.to_netcdf(path)

    grid, _ = make_demo_grid(ny=4, nx=6)
    depth_grid, meta = load_depth_to_grid(str(path), grid)

    assert depth_grid.shape == grid.shape()
    assert meta["source_path"].endswith("bathy.nc")
    assert meta["original_shape"] == (5, 5)
    assert "min" in meta and "max" in meta and "mean" in meta
    assert "nan_fraction" in meta
    assert meta["depth_sign_assumption"] == "negative_is_deeper"
