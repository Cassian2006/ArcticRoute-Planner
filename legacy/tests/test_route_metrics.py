from __future__ import annotations

import numpy as np
import xarray as xr

from ArcticRoute.core.route.metrics import compute_distance_km, integrate_field_along_path


def test_distance_and_integral_monotonicity():
    # Path 1: short
    path1 = [(10.0, 70.0), (10.5, 70.0)]
    # Path 2: longer
    path2 = [(10.0, 70.0), (10.5, 70.0), (11.0, 70.0)]

    dist1 = compute_distance_km(path1)
    dist2 = compute_distance_km(path2)
    assert dist2 > dist1

    # Field: constant value 1.0
    lat = np.array([69.0, 71.0])
    lon = np.array([9.0, 12.0])
    field_da = xr.DataArray(np.ones((2, 2)), coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))

    integ1 = integrate_field_along_path(field_da, path1)
    integ2 = integrate_field_along_path(field_da, path2)
    # Integral should be proportional to distance for constant field
    assert integ2 > integ1
    assert np.isclose(integ1 / 1000.0, dist1, rtol=1e-3)









