from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arcticroute.core.grid import Grid2D
from arcticroute.io.cmems_loader import load_ice_drift_from_nc, load_sit_from_nc


def _make_grid(ny: int, nx: int) -> Grid2D:
    lat_1d = np.linspace(65.0, 70.0, ny)
    lon_1d = np.linspace(0.0, 10.0, nx)
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
    return Grid2D(lat2d=lat2d, lon2d=lon2d)


def test_load_sit_and_drift(tmp_path: Path) -> None:
    xr = pytest.importorskip("xarray")

    ny, nx = 4, 6
    grid = _make_grid(ny, nx)

    sit = np.random.uniform(0.0, 2.0, (1, ny, nx))
    ds_sit = xr.Dataset(
        {
            "ice_thickness": (["time", "y", "x"], sit),
            "lat": (["y"], np.linspace(65, 70, ny)),
            "lon": (["x"], np.linspace(0, 10, nx)),
        }
    )
    sit_path = tmp_path / "sit.nc"
    ds_sit.to_netcdf(sit_path)

    u = np.random.uniform(-0.3, 0.3, (1, ny, nx))
    v = np.random.uniform(-0.2, 0.2, (1, ny, nx))
    ds_drift = xr.Dataset(
        {
            "ice_drift_x": (["time", "y", "x"], u),
            "ice_drift_y": (["time", "y", "x"], v),
            "lat": (["y"], np.linspace(65, 70, ny)),
            "lon": (["x"], np.linspace(0, 10, nx)),
        }
    )
    drift_path = tmp_path / "drift.nc"
    ds_drift.to_netcdf(drift_path)

    sit_arr, sit_meta = load_sit_from_nc(sit_path, grid=grid)
    assert sit_arr is not None
    assert sit_arr.shape == (ny, nx)
    assert sit_meta["status"] == "ok"

    u_arr, v_arr, speed_arr, drift_meta = load_ice_drift_from_nc(drift_path, grid=grid)
    assert u_arr is not None and v_arr is not None and speed_arr is not None
    assert u_arr.shape == (ny, nx)
    assert v_arr.shape == (ny, nx)
    assert speed_arr.shape == (ny, nx)
    assert drift_meta["status"] == "ok"


def test_load_sit_missing_variable(tmp_path: Path) -> None:
    xr = pytest.importorskip("xarray")

    ny, nx = 3, 5
    grid = _make_grid(ny, nx)
    ds = xr.Dataset(
        {
            "foo": (["y", "x"], np.zeros((ny, nx))),
            "lat": (["y"], np.linspace(65, 70, ny)),
            "lon": (["x"], np.linspace(0, 10, nx)),
        }
    )
    path = tmp_path / "missing_sit.nc"
    ds.to_netcdf(path)

    sit_arr, meta = load_sit_from_nc(path, grid=grid)
    assert sit_arr is None
    assert meta["status"] == "skipped"
