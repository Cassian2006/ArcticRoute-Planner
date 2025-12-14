from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
RISK_DIR = ROOT / "ArcticRoute" / "data_processed" / "risk"


def _write_nc(path: Path, name: str, arr: np.ndarray, coords: dict[str, xr.DataArray], dims: tuple[str, ...]):
    da = xr.DataArray(arr, coords=coords, dims=dims, name=name)
    da.to_dataset().to_netcdf(path)


def test_fuse_prefers_ice_eff(tmp_path: Path):
    ym = "209901"  # avoid clashing with other smoke
    RISK_DIR.mkdir(parents=True, exist_ok=True)
    # Create base risk_ice file with higher values
    lat = xr.DataArray([70.0, 70.5], dims="lat")
    lon = xr.DataArray([10.0, 10.5], dims="lon")
    base = np.full((2, 2), 0.9, dtype=np.float32)
    eff = np.full((2, 2), 0.1, dtype=np.float32)
    _write_nc(RISK_DIR / f"risk_ice_{ym}.nc", "R_ice", base, {"lat": lat, "lon": lon}, ("lat", "lon"))
    # Create R_ice_eff with much lower values
    _write_nc(RISK_DIR / f"R_ice_eff_{ym}.nc", "risk", eff, {"lat": lat, "lon": lon}, ("lat", "lon"))

    # Run fuse
    from ArcticRoute.core.risk.fusion import fuse_risk
    payload = fuse_risk(ym)
    out = Path(payload["out"]) if isinstance(payload.get("out"), str) else ROOT / "ArcticRoute" / "data_processed" / "risk" / f"risk_fused_{ym}.nc"
    assert out.exists()

    with xr.open_dataset(out) as ds:
        da = ds["Risk"]
        if "time" in da.dims:
            da = da.isel(time=0)
        # After quantile norm, both map to [0,1], but since only one layer and both constant, it becomes zeros.
        # The key is that source path should be R_ice_eff
        pass

    # Ensure source is ice_eff
    assert payload["sources"]["ice"]["path"].endswith(f"R_ice_eff_{ym}.nc")

