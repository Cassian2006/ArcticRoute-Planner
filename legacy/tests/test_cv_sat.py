from pathlib import Path
import sys

import numpy as np
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ArcticRoute.core.predictors.cv_sat import SatCVPredictor  # noqa: E402

try:  # pragma: no cover - optional dependency check
    import skimage  # type: ignore  # noqa: F401

    SKIMAGE_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency check
    SKIMAGE_AVAILABLE = False

pytestmark = pytest.mark.p0


@pytest.mark.skipif(not SKIMAGE_AVAILABLE, reason="skimage not available")
def test_run_otsu_generates_binary_mask(tmp_path):
    env_path = tmp_path / "env.nc"
    env_path.write_bytes(b"")  # placeholder to satisfy path operations

    predictor = SatCVPredictor(env_path=env_path, var_name="risk_env")
    lat = xr.DataArray([60.0, 60.5, 61.0], dims="latitude")
    lon = xr.DataArray([-10.0, -9.5, -9.0], dims="longitude")
    values = np.linspace(-1.0, 1.0, 9, dtype=np.float32).reshape(3, 3)
    candidate = xr.DataArray(
        values,
        coords={"latitude": lat, "longitude": lon},
        dims=("latitude", "longitude"),
        name="ndwi",
    )

    result = predictor._run_otsu(candidate, "test")
    assert result is not None
    ice_da, info = result

    assert ice_da.dims == candidate.dims
    assert ice_da.shape == candidate.shape
    assert ice_da.name == "ice_prob"
    finite = ice_da.values[np.isfinite(ice_da.values)]
    assert finite.size == candidate.size
    assert np.all((finite >= 0.0) & (finite <= 1.0))
    assert isinstance(info, dict)
