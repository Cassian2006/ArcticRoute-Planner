from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arcticroute.core.env_real import RealEnvLayers


def test_env_real_from_cmems_sit_drift(tmp_path: Path) -> None:
    xr = pytest.importorskip("xarray")

    ny, nx = 4, 6
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

    u = np.random.uniform(-0.4, 0.4, (1, ny, nx))
    v = np.random.uniform(-0.4, 0.4, (1, ny, nx))
    ds_drift = xr.Dataset(
        {
            "uice": (["time", "y", "x"], u),
            "vice": (["time", "y", "x"], v),
            "lat": (["y"], np.linspace(65, 70, ny)),
            "lon": (["x"], np.linspace(0, 10, nx)),
        }
    )
    drift_path = tmp_path / "drift.nc"
    ds_drift.to_netcdf(drift_path)

    env = RealEnvLayers.from_cmems(
        sit_nc=sit_path,
        drift_nc=drift_path,
        allow_partial=True,
    )

    assert env is not None
    assert env.grid is not None
    assert env.sit is not None
    assert env.ice_drift_speed is not None
