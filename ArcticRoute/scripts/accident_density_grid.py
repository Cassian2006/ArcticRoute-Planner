#!/usr/bin/env python3
"""Build accident density grid NetCDF from ingested incident data.

@role: pipeline
"""

"""
ACC-3 | Build static and time-varying accident density grids aligned to env_clean.nc.

- Loads incidents_aligned.parquet and env_clean.nc grid definition
- Aggregates incident counts onto the grid, applies lightweight box smoothing, and normalises to [0, 1]
- Persists two products:
    * accident_density_static.nc (variable: acc_density_static)
    * accident_density_time.nc (variable: acc_density_time)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INC_PATH = PROJECT_ROOT / "data_processed" / "incidents_aligned.parquet"
DEFAULT_ENV_PATH = PROJECT_ROOT / "data_processed" / "env_clean.nc"
DEFAULT_STATIC_OUT = PROJECT_ROOT / "data_processed" / "accident_density_static.nc"
DEFAULT_TIME_OUT = PROJECT_ROOT / "data_processed" / "accident_density_time.nc"


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    for base in (Path.cwd(), PROJECT_ROOT):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()


def _resolve_output(path: Path) -> Path:
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _load_incidents(path: Path, time_len: int, lat_len: int, lon_len: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.dropna(subset=["tidx", "ii", "jj"])
    if df.empty:
        return df
    df = df.astype({"tidx": "int64", "ii": "int64", "jj": "int64"})
    valid = (
        (df["tidx"].between(0, time_len - 1))
        & (df["ii"].between(0, lat_len - 1))
        & (df["jj"].between(0, lon_len - 1))
    )
    return df.loc[valid].copy()


def _box_smooth_2d(arr: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return arr.astype(np.float32, copy=True)
    kernel_size = 2 * radius + 1
    padded = np.pad(arr, radius, mode="constant", constant_values=0)
    out = np.zeros_like(arr, dtype=np.float32)
    for di in range(kernel_size):
        for dj in range(kernel_size):
            out += padded[di : di + arr.shape[0], dj : dj + arr.shape[1]]
    out /= float(kernel_size * kernel_size)
    return out


def _box_smooth_3d(arr: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return arr.astype(np.float32, copy=True)
    smoothed = np.zeros_like(arr, dtype=np.float32)
    for idx in range(arr.shape[0]):
        smoothed[idx] = _box_smooth_2d(arr[idx], radius)
    return smoothed


def _normalise_to_unit(arr: np.ndarray, quantile: float) -> tuple[np.ndarray, float]:
    """Normalise array to [0,1] using provided upper quantile as scale."""
    arr = arr.astype(np.float32, copy=True)
    positive = arr[arr > 0]
    if positive.size == 0:
        scale = 1.0
    else:
        scale = float(np.quantile(positive, quantile))
        if scale <= 0:
            scale = float(positive.max())
        if scale <= 0:
            scale = 1.0
    arr = np.clip(arr / scale, 0.0, 1.0)
    return arr, scale


def _describe(label: str, data: np.ndarray) -> None:
    print(
        f"[STAT] {label}: min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="ACC-3 accident density grid generation")
    parser.add_argument("--inc", type=Path, default=DEFAULT_INC_PATH, help="Path to incidents_aligned.parquet")
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV_PATH, help="Path to env_clean.nc")
    parser.add_argument("--out-static", type=Path, default=DEFAULT_STATIC_OUT, help="Output path for static density nc")
    parser.add_argument("--out-time", type=Path, default=DEFAULT_TIME_OUT, help="Output path for time-varying density nc")
    parser.add_argument("--smooth-radius", type=int, default=1, help="Box smoothing radius (cells)")
    parser.add_argument("--quantile", type=float, default=0.99, help="Quantile used for normalisation scaling")
    args = parser.parse_args()

    inc_path = _resolve_path(args.inc)
    env_path = _resolve_path(args.env)
    static_out = _resolve_output(args.out_static)
    time_out = _resolve_output(args.out_time)

    smooth_radius = max(0, int(args.smooth_radius))
    norm_quantile = float(args.quantile)

    with xr.open_dataset(env_path) as ds:
        env_time = ds["time"].values
        env_lat = ds["latitude"].values
        env_lon = ds["longitude"].values

    time_len = len(env_time)
    lat_len = len(env_lat)
    lon_len = len(env_lon)

    incidents = _load_incidents(inc_path, time_len, lat_len, lon_len)
    print(f"[STAT] Aligned incidents available: {len(incidents)}")

    static_counts = np.zeros((lat_len, lon_len), dtype=np.float32)
    time_counts = np.zeros((time_len, lat_len, lon_len), dtype=np.float32)

    if not incidents.empty:
        np.add.at(static_counts, (incidents["ii"].to_numpy(), incidents["jj"].to_numpy()), 1.0)
        np.add.at(
            time_counts,
            (
                incidents["tidx"].to_numpy(),
                incidents["ii"].to_numpy(),
                incidents["jj"].to_numpy(),
            ),
            1.0,
        )

    static_smoothed = _box_smooth_2d(static_counts, smooth_radius)
    time_smoothed = _box_smooth_3d(time_counts, smooth_radius)

    static_norm, static_scale = _normalise_to_unit(static_smoothed, norm_quantile)
    time_norm, time_scale = _normalise_to_unit(time_smoothed, norm_quantile)

    _describe("Static density", static_norm)
    _describe("Time density", time_norm)

    created_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    smoothing_desc = f"box_mean_radius={smooth_radius}"

    static_da = xr.DataArray(
        static_norm.astype(np.float32),
        dims=("latitude", "longitude"),
        coords={"latitude": env_lat, "longitude": env_lon},
        name="acc_density_static",
        attrs={
            "source_incidents": str(inc_path),
            "smoothing": smoothing_desc,
            "normalization_quantile": norm_quantile,
            "normalization_scale": static_scale,
        },
    )
    static_ds = xr.Dataset(
        {"acc_density_static": static_da},
        coords={"latitude": env_lat, "longitude": env_lon},
        attrs={
            "created_at": created_at,
            "smoothing": smoothing_desc,
            "normalization_quantile": norm_quantile,
            "normalization_scale": static_scale,
            "source_incidents": str(inc_path),
        },
    )

    time_da = xr.DataArray(
        time_norm.astype(np.float32),
        dims=("time", "latitude", "longitude"),
        coords={"time": env_time, "latitude": env_lat, "longitude": env_lon},
        name="acc_density_time",
        attrs={
            "source_incidents": str(inc_path),
            "smoothing": smoothing_desc,
            "normalization_quantile": norm_quantile,
            "normalization_scale": time_scale,
        },
    )
    time_ds = xr.Dataset(
        {"acc_density_time": time_da},
        coords={"time": env_time, "latitude": env_lat, "longitude": env_lon},
        attrs={
            "created_at": created_at,
            "smoothing": smoothing_desc,
            "normalization_quantile": norm_quantile,
            "normalization_scale": time_scale,
            "source_incidents": str(inc_path),
        },
    )

    static_out.parent.mkdir(parents=True, exist_ok=True)
    time_out.parent.mkdir(parents=True, exist_ok=True)

    static_ds.to_netcdf(static_out)
    time_ds.to_netcdf(time_out)

    print(f"[OK] Wrote {static_out}")
    print(f"[OK] Wrote {time_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
