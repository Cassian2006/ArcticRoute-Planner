#!/usr/bin/env python3
"""Align incident points to model grid/time index for downstream use.

@role: pipeline
"""

"""
ACC-2 | Align incident points onto the environmental grid/time axis.

- Load incidents_clean.parquet and env_clean.nc
- Filter incidents to the env time extent and map to nearest (time, lat, lon)
- Enforce a maximum temporal gap for alignment
- Persist alignment result to incidents_aligned.parquet and report coverage stats
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INC_PATH = PROJECT_ROOT / "data_processed" / "incidents_clean.parquet"
DEFAULT_ENV_PATH = PROJECT_ROOT / "data_processed" / "env_clean.nc"
DEFAULT_OUT_PATH = PROJECT_ROOT / "data_processed" / "incidents_aligned.parquet"

def _wrap_longitude(values: np.ndarray) -> np.ndarray:
    """Wrap longitudes into [-180, 180)."""
    wrapped = ((values + 180.0) % 360.0) - 180.0
    # Preserve 180 exactly if original was 180
    wrapped[np.isclose(wrapped, -180.0) & (values > 180.0)] = 180.0
    return wrapped


def _nearest_index(sorted_values: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest index (0-based) and absolute distance for sorted coordinate array."""
    insert_pos = np.searchsorted(sorted_values, targets, side="left")
    right = np.clip(insert_pos, 0, len(sorted_values) - 1)
    left = np.clip(insert_pos - 1, 0, len(sorted_values) - 1)

    diff_right = np.abs(sorted_values[right] - targets)
    diff_left = np.abs(sorted_values[left] - targets)

    use_left = diff_left <= diff_right
    indices = np.where(use_left, left, right)
    deltas = np.where(use_left, diff_left, diff_right)
    return indices, deltas


def _align_time(
    env_times: pd.DatetimeIndex, incident_times: pd.Series, max_gap_hours: float
) -> tuple[np.ndarray, np.ndarray]:
    """Map incidents to nearest env time index and return (indices, time_gap_ns)."""
    env_ns = env_times.values.astype("int64", copy=False)
    inc_ns = incident_times.astype("int64").to_numpy()

    insert_pos = np.searchsorted(env_ns, inc_ns, side="left")
    right = np.clip(insert_pos, 0, len(env_ns) - 1)
    left = np.clip(insert_pos - 1, 0, len(env_ns) - 1)

    diff_right = np.abs(env_ns[right] - inc_ns)
    diff_left = np.abs(env_ns[left] - inc_ns)

    use_left = diff_left <= diff_right
    indices = np.where(use_left, left, right)
    gaps = np.where(use_left, diff_left, diff_right)

    max_gap_ns = int(pd.to_timedelta(max_gap_hours, unit="h").value)
    valid = gaps <= max_gap_ns
    return np.where(valid, indices, -1), gaps


def _resolve_input_path(path: Path) -> Path:
    """Resolve a (possibly relative) path against common project roots."""
    if path.is_absolute():
        return path
    for base in (Path.cwd(), PROJECT_ROOT):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()


def _resolve_output_path(path: Path) -> Path:
    """Resolve output path, defaulting to current working directory when relative."""
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _prepare_incidents(path: Path, env_tmin: pd.Timestamp, env_tmax: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    """Load incidents parquet and filter to env time range."""
    df = pd.read_parquet(path)
    total_rows = len(df)
    df["time_dt"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["time_dt"])
    within = (df["time_dt"] >= env_tmin) & (df["time_dt"] <= env_tmax)
    filtered = df.loc[within].copy()
    filtered.reset_index(drop=True, inplace=True)
    return filtered, total_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="ACC-2 incident alignment onto env grid")
    parser.add_argument("--inc", type=Path, default=DEFAULT_INC_PATH, help="Path to incidents_clean.parquet")
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV_PATH, help="Path to env_clean.nc")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help="Output parquet path")
    parser.add_argument(
        "--max-hour-gap",
        type=float,
        default=48.0,
        dest="max_hour_gap",
        help="Maximum allowed time delta (hours) between incident and env grid",
    )
    args = parser.parse_args()

    inc_path: Path = _resolve_input_path(args.inc)
    env_path: Path = _resolve_input_path(args.env)
    out_path: Path = _resolve_output_path(args.out)
    max_hour_gap: float = args.max_hour_gap

    with xr.open_dataset(env_path) as ds:
        env_time = pd.to_datetime(ds["time"].values).tz_localize("UTC")
        env_lat = ds["latitude"].values.astype("float64")
        env_lon = ds["longitude"].values.astype("float64")

    incidents, total_initial = _prepare_incidents(inc_path, env_time.min(), env_time.max())
    total_filtered = len(incidents)
    print(f"[STAT] Incidents loaded: {total_initial}")
    print(f"[STAT] Within env time window: {total_filtered} (dropped {total_initial - total_filtered})")

    if incidents.empty:
        print("[WARN] No incidents within env time range; nothing to align.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        empty_df = pd.DataFrame(columns=["time_utc", "lat", "lon", "tidx", "ii", "jj", "type", "severity"])
        empty_df.to_parquet(out_path, index=False)
        print(f"[OK] Wrote empty alignment parquet to {out_path}")
        return 0

    incidents["lon"] = _wrap_longitude(incidents["lon"].to_numpy())

    lat_values_all = incidents["lat"].to_numpy(dtype="float64")
    lon_values_all = incidents["lon"].to_numpy(dtype="float64")

    lat_in_range = np.isfinite(lat_values_all) & (lat_values_all >= env_lat.min()) & (lat_values_all <= env_lat.max())
    lon_in_range = np.isfinite(lon_values_all) & (lon_values_all >= env_lon.min()) & (lon_values_all <= env_lon.max())
    spatial_mask = lat_in_range & lon_in_range

    spatial_kept = int(spatial_mask.sum())
    print(f"[STAT] Within env spatial coverage: {spatial_kept} (dropped {len(incidents) - spatial_kept})")

    if spatial_kept == 0:
        print("[WARN] No incidents fall inside the env grid coverage; nothing to align.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        empty_df = pd.DataFrame(columns=["time_utc", "lat", "lon", "tidx", "ii", "jj", "type", "severity"])
        empty_df.to_parquet(out_path, index=False)
        print(f"[OK] Wrote empty alignment parquet to {out_path}")
        return 0

    incidents = incidents.loc[spatial_mask].copy()
    lat_values = lat_values_all[spatial_mask]
    lon_values = lon_values_all[spatial_mask]

    time_indices, _ = _align_time(env_time, incidents["time_dt"], max_hour_gap)
    time_valid = time_indices >= 0

    lat_indices, _ = _nearest_index(env_lat, lat_values)
    lon_indices, _ = _nearest_index(env_lon, lon_values)

    success_mask = time_valid
    success_count = int(success_mask.sum())
    failure_count = int(len(incidents) - success_count)
    success_rate = success_count / len(incidents) if len(incidents) else 0.0
    failure_rate = 1.0 - success_rate if len(incidents) else 0.0

    time_miss = int((~time_valid).sum())
    print(f"[STAT] Alignment success: {success_count}/{len(incidents)} ({success_rate:.1%})")
    print(f"[STAT] Alignment failures: {failure_count}")
    print(f"[STAT] Alignment failure rate: {failure_rate:.1%}")
    print(f"[STAT] Failures due to time gap: {time_miss}")

    tidx_series = pd.Series(time_indices, index=incidents.index, dtype="Int64")
    tidx_series.loc[~time_valid] = pd.NA
    incidents["tidx"] = tidx_series

    incidents["ii"] = pd.Series(lat_indices, index=incidents.index, dtype="Int64")
    incidents["jj"] = pd.Series(lon_indices, index=incidents.index, dtype="Int64")

    incidents["time_utc"] = incidents["time_dt"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    incidents.drop(columns=["time_dt", "source"], inplace=True, errors="ignore")

    result = incidents[["time_utc", "lat", "lon", "tidx", "ii", "jj", "type", "severity"]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    print(f"[OK] Wrote aligned incidents to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
