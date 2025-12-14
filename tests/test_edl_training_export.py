import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from scripts.export_edl_training_dataset import export_edl_training_dataset


def _make_toy_nc(path: Path, var_name: str, data: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    ds = xr.Dataset(
        {
            var_name: (("y", "x"), data.astype(np.float32)),
        },
        coords={
            "y": ("y", lat.astype(np.float32)),
            "x": ("x", lon.astype(np.float32)),
            "latitude": ("y", lat.astype(np.float32)),
            "longitude": ("x", lon.astype(np.float32)),
        },
    )
    ds.to_netcdf(path)


def test_export_edl_training_dataset_toy(tmp_path: Path):
    # Create a small toy grid 6x6
    ny, nx = 6, 6
    lat = np.linspace(70, 75, ny)
    lon = np.linspace(-20, 10, nx)

    # SIC: left half low (safe), right half high (risky)
    sic = np.zeros((ny, nx), dtype=np.float32)
    sic[:, : nx // 2] = 0.1  # safe side
    sic[:, nx // 2 :] = 0.85  # risky side

    # Wave: mostly low
    wave = np.full((ny, nx), 1.5, dtype=np.float32)

    # AIS density: left-top quadrant high (safe), others low (risky)
    ais = np.zeros((ny, nx), dtype=np.float32)
    ais[: ny // 2, : nx // 2] = 0.5
    ais[:, nx // 2 :] = 0.0

    sic_nc = tmp_path / "sic.nc"
    wave_nc = tmp_path / "wave.nc"
    ais_nc = tmp_path / "ais.nc"

    _make_toy_nc(sic_nc, "sic", sic, lat, lon)
    _make_toy_nc(wave_nc, "wave_swh", wave, lat, lon)
    _make_toy_nc(ais_nc, "ais_density", ais, lat, lon)

    out_parquet = tmp_path / "edl_train.parquet"

    df = export_edl_training_dataset(
        output_path=out_parquet,
        max_samples=10_000,
        nc_sic_path=sic_nc,
        nc_wave_path=wave_nc,
        nc_ais_density_path=ais_nc,
        time_index=0,
        replicate_by_vessel=False,
    )

    assert out_parquet.exists(), "Parquet should be written"

    # 必须含有指定列名
    required_cols = [
        "lat",
        "lon",
        "month",
        "dayofyear",
        "sic",
        "ice_thickness_m",
        "wave_swh",
        "ais_density",
        "vessel_class_id",
        "label_safe_risky",
    ]
    for c in required_cols:
        assert c in df.columns, f"missing column {c}"

    # 至少有 label=1 和 label=0
    assert (df["label_safe_risky"] == 1).any(), "should contain some safe labels"
    assert (df["label_safe_risky"] == 0).any(), "should contain some risky labels"

    # 没有 NaN/Inf
    assert np.isfinite(df[required_cols].to_numpy()).all(), "no NaN/Inf allowed in dataset"

    # 行数 > 0 且与 max_samples 约束一致
    assert len(df) > 0
    assert len(df) <= 10_000

