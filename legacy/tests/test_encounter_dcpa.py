from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
AIS_DIR = ROOT / "ArcticRoute" / "data_processed" / "ais"
RISK_DIR = ROOT / "ArcticRoute" / "data_processed" / "risk"


def _make_tracks(ym: str):
    AIS_DIR.mkdir(parents=True, exist_ok=True)
    # 构造两船相向运动，cog: 90(东) 与 270(西)，sog=10kn
    t0 = pd.Timestamp("2024-12-01T00:00:00Z")
    times = pd.date_range(t0, t0 + pd.Timedelta(minutes=30), freq="1min")

    lat_a = np.full(len(times), 70.0)
    lon_a = 10.0 + np.linspace(0.0, 0.1, len(times))  # 向东
    sog_a = np.full(len(times), 10.0)
    cog_a = np.full(len(times), 90.0)
    mmsi_a = np.full(len(times), 111000001)

    lat_b = np.full(len(times), 70.0)
    lon_b = 10.1 - np.linspace(0.0, 0.1, len(times))  # 向西
    sog_b = np.full(len(times), 10.0)
    cog_b = np.full(len(times), 270.0)
    mmsi_b = np.full(len(times), 111000002)

    df_a = pd.DataFrame({
        "time": times,
        "lat": lat_a,
        "lon": lon_a,
        "sog": sog_a,
        "cog": cog_a,
        "mmsi": mmsi_a,
    })
    df_b = pd.DataFrame({
        "time": times,
        "lat": lat_b,
        "lon": lon_b,
        "sog": sog_b,
        "cog": cog_b,
        "mmsi": mmsi_b,
    })
    df = pd.concat([df_a, df_b], ignore_index=True)
    df.to_parquet(AIS_DIR / f"tracks_{ym}.parquet")


def test_encounter_dcpa_basic():
    ym = "209903"
    _make_tracks(ym)

    from ArcticRoute.core.congest.encounter import build_interact_layer
    da = build_interact_layer(ym)

    # 输出检查
    out_path = RISK_DIR / f"R_interact_{ym}.nc"
    assert out_path.exists()
    assert isinstance(da, xr.DataArray)
    assert da.name == "risk"
    arr = da.values
    assert np.isfinite(arr).all()
    assert float(np.nanmax(arr)) > 0.0

