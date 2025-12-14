from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import xarray as xr

from ArcticRoute.core.reporting.uncertainty import build_month


def _ensure_dirs():
    repo = Path(__file__).resolve().parents[1]
    (repo / "ArcticRoute" / "data_processed" / "risk").mkdir(parents=True, exist_ok=True)
    (repo / "ArcticRoute" / "reports" / "d_stage" / "phaseI").mkdir(parents=True, exist_ok=True)
    # features 可选，不强制


def _write_fused_nc(ym: str):
    repo = Path(__file__).resolve().parents[1]
    root = repo / "ArcticRoute" / "data_processed" / "risk"
    root.mkdir(parents=True, exist_ok=True)
    lat = np.linspace(60.0, 61.0, 5)
    lon = np.linspace(10.0, 11.0, 5)
    risk = np.clip(np.outer(np.linspace(0.1, 0.9, 5), np.ones(5)), 0.0, 1.0)
    var = 0.05 * (np.ones_like(risk) * 0.25)  # 小方差
    ds = xr.Dataset(
        {
            "Risk": xr.DataArray(risk.astype("float32"), dims=("lat", "lon"), coords={"lat": lat, "lon": lon}),
            "RiskVar": xr.DataArray(var.astype("float32"), dims=("lat", "lon"), coords={"lat": lat, "lon": lon}),
        }
    )
    out = root / f"risk_fused_{ym}.nc"
    ds.to_netcdf(out)
    ds.close()
    return out


def test_uncertainty_report_builds_json_png(tmp_path: Path, monkeypatch):
    ym = "202401"
    _ensure_dirs()
    fused_path = _write_fused_nc(ym)
    # 执行
    payload = build_month(ym)
    assert os.path.exists(payload["json"]) , payload
    assert os.path.exists(payload["png"]) , payload

