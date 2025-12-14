from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
DP_RISK = ROOT / "ArcticRoute" / "data_processed" / "risk"


def test_build_interact_layer_stub(tmp_path: Path):
    ym = "202412"
    # 清理目标文件
    out_path = DP_RISK / f"R_interact_{ym}.nc"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    # 直接调用构建函数
    from ArcticRoute.core.congest.encounter import build_interact_layer

    da = build_interact_layer(ym)
    assert da.name == "risk"
    assert isinstance(da, xr.DataArray)
    assert np.isfinite(da.values).all()
    # 文件应已写盘
    assert out_path.exists()


def test_apply_escort_stub(tmp_path: Path):
    ym = "202412"
    out_path = DP_RISK / f"R_ice_eff_{ym}.nc"
    if out_path.exists():
        out_path.unlink()

    # 构造一个最小的 R_ice 源文件供读取
    src_path = DP_RISK / f"risk_ice_{ym}.nc"
    src_path.parent.mkdir(parents=True, exist_ok=True)
    lat = xr.DataArray([70.0, 70.5], dims="lat")
    lon = xr.DataArray([10.0, 10.5], dims="lon")
    data = xr.DataArray(np.zeros((2, 2), dtype=np.float32), coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name="R_ice")
    data.to_dataset().to_netcdf(src_path)

    from ArcticRoute.core.risk.escort import apply_escort

    da_eff = apply_escort(ym, eta=0.2)
    assert da_eff.name == "risk"
    assert out_path.exists()
    with xr.open_dataset(out_path) as ds:
        assert "risk" in ds
        arr = ds["risk"].values
        assert np.all(arr >= 0.0) and np.all(arr <= 1.0)

