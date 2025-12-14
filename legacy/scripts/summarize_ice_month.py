from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

try:
    import numpy as np
    import xarray as xr
except ImportError:
    print("xarray and numpy are required for this script.")
    exit(1)

REPO_ROOT = Path(__file__).resolve().parents[1]

def summarize(
    month: str,
    warn_nan_pct: float = 30.0,
    warn_sic_max: float = 1.2,
    warn_cost_max: float = 0.3,
) -> Dict[str, Any]:
    """Summarize a month's merged NetCDF files into a one-line JSON.

    Args:
        month: The target month in YYYYMM format.
        warn_nan_pct: Warning threshold for NaN percentage.
        warn_sic_max: Warning threshold for SIC max value.
        warn_cost_max: Warning threshold for cost max value.

    Returns:
        A dictionary with summary statistics and warnings.
    """
    base_merged = REPO_ROOT / "ArcticRoute" / "data_processed" / "ice_forecast" / "merged"
    sic_path = base_merged / f"sic_fcst_{month}.nc"
    cost_path = base_merged / f"ice_cost_{month}.nc"

    result: Dict[str, Any] = {"month": month, "warnings": []}
    stats: Dict[str, Any] = {}

    def _get_stats(da: xr.DataArray, name: str) -> Dict[str, Any]:
        arr = da.values
        if arr.size == 0:
            return {}
        nan_count = np.isnan(arr).sum()
        return {
            f"{name}_nan_pct": (nan_count / arr.size) * 100 if arr.size > 0 else 0,
            f"{name}_min": float(np.nanmin(arr)) if nan_count < arr.size else None,
            f"{name}_max": float(np.nanmax(arr)) if nan_count < arr.size else None,
            f"{name}_mean": float(np.nanmean(arr)) if nan_count < arr.size else None,
        }

    # Process SIC file
    if sic_path.exists():
        with xr.open_dataset(sic_path) as ds:
            stats["shape"] = ds["sic_pred"].shape
            stats["chunks"] = ds["sic_pred"].encoding.get("chunksizes")
            stats["time_coverage"] = str(ds.get("time_coverage", "N/A"))
            stats["snapshot_source"] = str(ds.attrs.get("snapshot_source", "N/A"))
            sic_stats = _get_stats(ds["sic_pred"], "sic")
            stats.update(sic_stats)
    else:
        result["warnings"].append(f"Missing sic_fcst file: {sic_path}")

    # Process cost file
    if cost_path.exists():
        with xr.open_dataset(cost_path) as ds:
            if "ice_cost" in ds:
                cost_stats = _get_stats(ds["ice_cost"], "cost")
                stats.update(cost_stats)
            else:
                 result["warnings"].append(f"ice_cost variable not found in {cost_path}")
    else:
        result["warnings"].append(f"Missing ice_cost file: {cost_path}")

    # Check warnings
    nan_pct = stats.get("sic_nan_pct")
    if nan_pct is not None and nan_pct > warn_nan_pct:
        result["warnings"].append(f"High NaN percentage: {nan_pct:.2f}% > {warn_nan_pct}%")

    sic_max = stats.get("sic_max")
    if sic_max is not None and sic_max > warn_sic_max:
        result["warnings"].append(f"High SIC max value: {sic_max:.3f} > {warn_sic_max}")

    cost_max = stats.get("cost_max")
    if cost_max is not None and cost_max < warn_cost_max:
        result["warnings"].append(f"Low cost max value: {cost_max:.3f} < {warn_cost_max}")

    result.update(stats)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize a month's data.")
    parser.add_argument("--month", required=True, help="Target month in YYYYMM format.")
    parser.add_argument("--warn-nan-pct", type=float, default=30.0)
    parser.add_argument("--warn-sic-max", type=float, default=1.2)
    parser.add_argument("--warn-cost-max", type=float, default=0.3)
    args = parser.parse_args()

    summary_data = summarize(
        args.month,
        warn_nan_pct=args.warn_nan_pct,
        warn_sic_max=args.warn_sic_max,
        warn_cost_max=args.warn_cost_max,
    )

    print(json.dumps(summary_data, ensure_ascii=False, indent=2))
