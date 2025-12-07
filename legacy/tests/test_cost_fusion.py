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


def _make_da(name: str, values: np.ndarray) -> xr.DataArray:
    time = xr.DataArray(["2024-01-01"], dims="time")
    lat = xr.DataArray([60.0, 60.5], dims="latitude")
    lon = xr.DataArray([-10.0, -9.5], dims="longitude")
    data = values.astype(np.float32).reshape(1, 2, 2)
    return xr.DataArray(
        data,
        coords={"time": time, "latitude": lat, "longitude": lon},
        dims=("time", "latitude", "longitude"),
        name=name,
    )


def test_blend_with_ice_respects_formula_and_clipping():
    risk_vals = np.array([[0.8, 0.2], [0.5, 1.3]], dtype=np.float32)
    ice_vals = np.array([[0.1, np.nan], [0.4, 0.9]], dtype=np.float32)
    risk_da = _make_da("risk_env", risk_vals)
    ice_da = xr.DataArray(
        ice_vals,
        coords={"latitude": risk_da.latitude, "longitude": risk_da.longitude},
        dims=("latitude", "longitude"),
        name="ice_prob",
    )

    alpha = 0.6
    fused = EnvRiskCostProvider.blend_with_ice(risk_da, ice_da, alpha_ice=alpha)

    risk_values = risk_da.values
    ice_values = np.expand_dims(ice_vals, axis=0)
    ice_values = np.where(np.isnan(ice_values), risk_values, ice_values)
    expected = alpha * risk_values + (1.0 - alpha) * ice_values
    expected = np.clip(expected, 0.0, 1.0)

    np.testing.assert_allclose(fused.values, expected)
    assert fused.name == "final_risk"
    assert fused.attrs.get("alpha_ice") == alpha
    assert np.all((fused.values >= 0.0) & (fused.values <= 1.0))
