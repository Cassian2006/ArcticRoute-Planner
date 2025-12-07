from pathlib import Path
import sys

import numpy as np
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ArcticRoute.core.cost.env_risk_cost import EnvRiskCostProvider  # noqa: E402

pytestmark = pytest.mark.p0


def _make_risk() -> xr.DataArray:
    time = xr.DataArray(["2023-07-15"], dims="time")
    lat = xr.DataArray([60.0, 60.5], dims="latitude")
    lon = xr.DataArray([-10.0, -9.5], dims="longitude")
    data = np.array([[0.8, 0.6], [0.4, 0.2]], dtype=np.float32).reshape(1, 2, 2)
    return xr.DataArray(
        data,
        coords={"time": time, "latitude": lat, "longitude": lon},
        dims=("time", "latitude", "longitude"),
        name="risk_env",
    )


def _make_ice() -> xr.DataArray:
    lat = xr.DataArray([60.0, 60.5], dims="latitude")
    lon = xr.DataArray([-10.0, -9.5], dims="longitude")
    data = np.array([[0.0, 0.6], [np.nan, 1.2]], dtype=np.float32)
    return xr.DataArray(
        data,
        coords={"latitude": lat, "longitude": lon},
        dims=("latitude", "longitude"),
        name="ice_prob",
    )


def test_blend_with_ice_clips_and_preserves_nan():
    risk = _make_risk()
    ice = _make_ice()
    fused = EnvRiskCostProvider.blend_with_ice(risk, ice, alpha_ice=0.25)

    assert fused.dims == risk.dims
    assert fused.shape == risk.shape

    expected = np.array(
        [
            [
                [0.25 * 0.8 + 0.75 * 0.0, 0.25 * 0.6 + 0.75 * 0.6],
                [0.25 * 0.4 + 0.75 * 0.4, min(1.0, 0.25 * 0.2 + 0.75 * 1.2)],
            ]
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(fused.values, expected)

    # Ensure NaN in ice falls back to original risk
    assert fused.values[0, 1, 0] == risk.values[0, 1, 0]
