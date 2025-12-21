from __future__ import annotations

import numpy as np

from arcticroute.core.cost import build_cost_from_real_env
from arcticroute.core.env_real import RealEnvLayers
from arcticroute.core.grid import Grid2D


def _make_grid(ny: int, nx: int) -> Grid2D:
    lat_1d = np.linspace(65.0, 70.0, ny)
    lon_1d = np.linspace(0.0, 10.0, nx)
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
    return Grid2D(lat2d=lat2d, lon2d=lon2d)


def test_sit_drift_weights_zero_no_change() -> None:
    ny, nx = 4, 6
    grid = _make_grid(ny, nx)
    land_mask = np.zeros((ny, nx), dtype=bool)
    sic = np.full((ny, nx), 0.3)
    sit = np.linspace(0.0, 1.0, ny * nx).reshape(ny, nx)
    drift_speed = np.linspace(0.0, 0.5, ny * nx).reshape(ny, nx)
    env = RealEnvLayers(sic=sic, sit=sit, ice_drift_speed=drift_speed)

    base = build_cost_from_real_env(grid, land_mask, env, wave_penalty=0.0, w_sit=0.0, w_drift=0.0)
    assert "sit_cost" not in base.components
    assert "drift_cost" not in base.components


def test_sit_cost_monotonic() -> None:
    ny, nx = 3, 4
    grid = _make_grid(ny, nx)
    land_mask = np.zeros((ny, nx), dtype=bool)
    sic = np.zeros((ny, nx))
    sit = np.zeros((ny, nx))
    sit[0, 0] = 0.1
    sit[-1, -1] = 1.0
    env = RealEnvLayers(sic=sic, sit=sit)

    cost_field = build_cost_from_real_env(grid, land_mask, env, w_sit=2.0, w_drift=0.0)
    sit_cost = cost_field.components["sit_cost"]
    assert sit_cost[0, 0] <= sit_cost[-1, -1]


def test_drift_cost_missing_outputs_zero() -> None:
    ny, nx = 3, 4
    grid = _make_grid(ny, nx)
    land_mask = np.zeros((ny, nx), dtype=bool)
    sic = np.zeros((ny, nx))
    env = RealEnvLayers(sic=sic, ice_drift_speed=None)

    cost_field = build_cost_from_real_env(grid, land_mask, env, w_sit=0.0, w_drift=1.0)
    drift_cost = cost_field.components["drift_cost"]
    assert np.allclose(drift_cost, 0.0)
