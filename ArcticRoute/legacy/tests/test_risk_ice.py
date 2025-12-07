"""Tests for core.risk.ice"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from ArcticRoute.core.risk.ice import build_risk_ice


@pytest.fixture
def mock_sic_fcst(tmp_path: Path) -> str:
    ym = "202412"
    p = tmp_path / "merged" / f"sic_fcst_{ym}.nc"
    p.parent.mkdir(parents=True, exist_ok=True)

    # Create a simple gradient
    lat = np.linspace(70, 75, 10)
    lon = np.linspace(-160, -150, 20)
    sic_data = np.linspace(0, 1, 20)[None, :] * np.ones((10, 1))
    
    da = xr.DataArray(
        sic_data[np.newaxis, :, :],
        dims=("time", "y", "x"),
        coords={"time": [np.datetime64("2024-12-01")], "y": lat, "x": lon},
        name="sic_pred",
    )
    ds = xr.Dataset({"sic_pred": da})
    ds.to_netcdf(p)
    # Mock the expected path
    os.environ["ARCTICROUTE_DATA_PROCESSED"] = str(tmp_path)
    # For simplicity, we'll mock the path structure
    full_path = tmp_path / "ice_forecast" / "merged" / f"sic_fcst_{ym}.nc"
    full_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(full_path)

    # Also mock the root directory for path lookups
    if "cwd" not in mock_sic_fcst.__dict__:
        mock_sic_fcst.cwd = os.getcwd()
        os.chdir(tmp_path.parent.parent) # a bit ugly, depends on tmp_path structure

    return ym


def test_build_risk_ice_edge_toggle(mock_sic_fcst, tmp_path):
    ym = mock_sic_fcst
    # Change CWD to make paths relative to repo root work
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # Run with edge detection
        res_edge = build_risk_ice(ym, 0.1, 0.75, 1.5, use_edge=True, use_thickness=False, dry_run=False)
        ds_edge = xr.open_dataset(res_edge["out"])
        arr_edge = ds_edge["R_ice"].values

        # Run without edge detection
        res_no_edge = build_risk_ice(ym, 0.1, 0.75, 1.5, use_edge=False, use_thickness=False, dry_run=False)
        ds_no_edge = xr.open_dataset(res_no_edge["out"])
        arr_no_edge = ds_no_edge["R_ice"].values

        # Assertions
        assert not np.allclose(arr_edge, arr_no_edge), "Edge detection should modify the output"
        assert np.all(arr_edge >= 0) and np.all(arr_edge <= 1), "Values must be in [0, 1]"
        assert np.all(arr_no_edge >= 0) and np.all(arr_no_edge <= 1), "Values must be in [0, 1]"
        assert ds_edge.attrs["use_edge"] == 1
        assert ds_no_edge.attrs["use_edge"] == 0

    finally:
        os.chdir(original_cwd)

