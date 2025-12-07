from __future__ import annotations

import numpy as np
import xarray as xr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RISK_DIR = ROOT / "ArcticRoute" / "data_processed" / "risk"


def test_apply_escort_reduces_when_P_present(tmp_path: Path):
    ym = "209902"
    RISK_DIR.mkdir(parents=True, exist_ok=True)
    # 写入一个简易 R_ice（全 1）
    lat = xr.DataArray([70.0, 70.5], dims="lat")
    lon = xr.DataArray([10.0, 10.5], dims="lon")
    R_ice = xr.DataArray(np.ones((2, 2), dtype=np.float32), coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name="R_ice")
    (RISK_DIR / f"risk_ice_{ym}.nc").write_bytes(R_ice.to_dataset().to_netcdf())

    # 写入一个 P_escort_corridor（非零）
    P = xr.DataArray(np.array([[0.0, 0.5], [1.0, 0.2]], dtype=np.float32), coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name="P")
    (RISK_DIR / f"P_escort_corridor_{ym}.nc").write_bytes(P.to_dataset().to_netcdf())

    from ArcticRoute.core.risk.escort import apply_escort
    da_eff = apply_escort(ym, eta=0.3)

    assert da_eff.name == "risk"
    arr = da_eff.values
    # 根据 m = 1 - eta*P，R_eff = 1 * m
    expected = np.clip(1.0 - 0.3 * P.values, 0.0, 1.0)
    assert np.allclose(arr, expected)

