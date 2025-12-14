import numpy as np
import xarray as xr

from arcticroute.core.cost import build_demo_cost
from arcticroute.core.grid import Grid2D


def _make_grid(ny: int = 3, nx: int = 3):
    lat = np.linspace(60.0, 60.0 + (ny - 1) * 0.5, ny)
    lon = np.linspace(0.0, (nx - 1) * 0.5, nx)
    lon2d, lat2d = np.meshgrid(lon, lat)
    land_mask = np.zeros((ny, nx), dtype=bool)
    return Grid2D(lat2d=lat2d, lon2d=lon2d), land_mask


def test_corridor_prefers_high_density():
    grid, land_mask = _make_grid()
    density = np.array(
        [
            [0.1, 0.9, 0.1],
            [0.1, 0.9, 0.1],
            [0.1, 0.9, 0.1],
        ],
        dtype=float,
    )

    cost_field = build_demo_cost(
        grid,
        land_mask,
        ice_penalty=0.0,
        ice_lat_threshold=100.0,
        w_ais_corridor=2.0,
        ais_density=density,
    )

    corridor = cost_field.components.get("ais_corridor")
    assert corridor is not None
    assert corridor.shape == density.shape
    # 高密度主航道的成本更低
    assert corridor[0, 1] < corridor[0, 0]


def test_congestion_penalizes_only_high_quantile():
    grid, land_mask = _make_grid()
    dense = np.full((3, 3), 0.1, dtype=float)
    dense[1, 1] = 0.9  # 单个高峰

    cost_field = build_demo_cost(
        grid,
        land_mask,
        ice_penalty=0.0,
        ice_lat_threshold=100.0,
        w_ais_congestion=3.0,
        ais_density=dense,
    )

    congestion = cost_field.components.get("ais_congestion")
    assert congestion is not None
    assert congestion[1, 1] > 0  # 高峰点被惩罚
    positive_mask = np.isfinite(congestion) & (congestion > 0)
    assert np.count_nonzero(positive_mask) == 1  # 仅高分位热点产生惩罚


def test_legacy_w_ais_maps_to_corridor_component():
    grid, land_mask = _make_grid()
    density = np.full((3, 3), 0.5, dtype=float)

    cost_field = build_demo_cost(
        grid,
        land_mask,
        ice_penalty=0.0,
        ice_lat_threshold=100.0,
        w_ais=1.2,
        ais_density=density,
    )

    assert "ais_corridor" in cost_field.components
    assert cost_field.components["ais_corridor"].shape == density.shape


def test_resampling_aligns_ais_to_grid_shape():
    grid, land_mask = _make_grid(3, 3)
    src_lat = np.array([[60.0, 60.0], [60.5, 60.5]])
    src_lon = np.array([[0.0, 0.5], [0.0, 0.5]])
    ais_da = xr.DataArray(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        dims=("y", "x"),
        coords={"lat": (("y", "x"), src_lat), "lon": (("y", "x"), src_lon)},
        name="ais_density",
    )

    cost_field = build_demo_cost(
        grid,
        land_mask,
        ice_penalty=0.0,
        ice_lat_threshold=100.0,
        w_ais_corridor=1.0,
        ais_density=ais_da,
    )

    corridor = cost_field.components.get("ais_corridor")
    assert corridor is not None
    assert corridor.shape == grid.shape()
    assert np.isfinite(corridor).any()
