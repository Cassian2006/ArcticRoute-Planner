from __future__ import annotations

import numpy as np

from arcticroute.core.grid import Grid2D
from arcticroute.core.cost import build_cost_from_real_env
from arcticroute.core.env_real import RealEnvLayers


def test_cost_components_change_with_weights():
    lat2d = np.array([[70.0, 70.0], [71.0, 71.0]])
    lon2d = np.array([[0.0, 1.0], [0.0, 1.0]])
    grid = Grid2D(lat2d=lat2d, lon2d=lon2d)
    land_mask = np.zeros_like(lat2d, dtype=bool)

    env = RealEnvLayers(
        grid=grid,
        sic=np.zeros_like(lat2d),
        wave_swh=None,
        land_mask=land_mask,
        ice_thickness_m=np.ones_like(lat2d),  # 1m 冰厚
        ice_drift=np.ones_like(lat2d) * 0.5,  # 0.5 m/s
        bathymetry_depth_m=np.ones_like(lat2d) * 10.0,  # 10m 深度
        meta={},
    )

    cost_zero = build_cost_from_real_env(
        grid,
        land_mask,
        env,
        ice_penalty=0.0,
        wave_penalty=0.0,
        w_sit=0.0,
        w_drift=0.0,
        w_shallow=0.0,
        min_depth_m=20.0,
    )
    cost_weighted = build_cost_from_real_env(
        grid,
        land_mask,
        env,
        ice_penalty=0.0,
        wave_penalty=0.0,
        w_sit=2.0,
        w_drift=1.5,
        w_shallow=1.0,
        min_depth_m=20.0,
    )

    assert cost_weighted.cost.sum() > cost_zero.cost.sum()
    for key in ["sit_cost", "drift_cost", "shallow_cost"]:
        assert key in cost_weighted.components
        assert np.isfinite(cost_weighted.components[key]).any()

