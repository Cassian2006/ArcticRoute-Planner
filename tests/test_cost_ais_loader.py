import numpy as np
import xarray as xr

from arcticroute.core.cost import _add_ais_cost_component, AIS_DENSITY_PATH_REAL
from arcticroute.core.grid import Grid2D


def test_add_ais_component_prefers_real_file(monkeypatch, tmp_path):
    ny, nx = 5, 7
    lat1d = np.linspace(60.0, 70.0, ny)
    lon1d = np.linspace(-10.0, 10.0, nx)
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)
    grid = Grid2D(lat2d=lat2d, lon2d=lon2d)

    ais_values = np.linspace(0, 1, num=ny * nx, dtype=float).reshape(ny, nx)
    ds = xr.Dataset({"ais_density": (("y", "x"), ais_values)})
    real_path = tmp_path / "ais_density_2024_real.nc"
    ds.to_netcdf(real_path)

    from arcticroute.core import cost as cost_module
    monkeypatch.setattr(cost_module, "AIS_DENSITY_PATH_REAL", real_path)

    base_cost = np.zeros((ny, nx), dtype=float)
    components = {}
    _add_ais_cost_component(
        base_cost,
        components,
        ais_density=None,
        weight_ais=2.0,
        grid=grid,
        prefer_real=True,
    )

    assert "ais_density" in components
    assert np.sum(components["ais_density"]) > 0
    assert np.sum(base_cost) > 0
