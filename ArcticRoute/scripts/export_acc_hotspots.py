#!/usr/bin/env python3
"""Export accident hotspot layers derived from analysis to GeoJSON.

@role: pipeline
"""

"""
导出事故热点为 GeoJSON 点集（阈值筛选）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT.parent / path).resolve()


def export_hotspots(nc_path: Path, threshold: float, out_path: Path, limit: int) -> int:
    ds = xr.open_dataset(nc_path)
    try:
        if "acc_density_static" in ds:
            density = ds["acc_density_static"]
        elif "accident_density" in ds:
            density = ds["accident_density"]
        else:
            raise KeyError("accident density variable missing in NetCDF")

        values = density.values
        lat = ds["latitude"].values
        lon = ds["longitude"].values
    finally:
        ds.close()

    mask = np.where(values >= threshold)
    count = len(mask[0])
    if count == 0:
        features = []
    else:
        if limit and count > limit:
            idx = np.random.default_rng(2025).choice(count, size=limit, replace=False)
            mask = (mask[0][idx], mask[1][idx])
        features = []
        for ii, jj in zip(mask[0], mask[1]):
            feat = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lon[jj]), float(lat[ii])],
                },
                "properties": {
                    "value": float(values[ii, jj]),
                },
            }
            features.append(feat)

    geojson = {"type": "FeatureCollection", "features": features}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[HOTSPOTS] exported {len(features)} features to {out_path}")
    return len(features)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export accident hotspots to GeoJSON")
    parser.add_argument("--nc", type=Path, required=True, help="accident density NetCDF")
    parser.add_argument("--th", type=float, default=0.8, help="threshold for hotspot selection")
    parser.add_argument("--out", type=Path, required=True, help="output GeoJSON path")
    parser.add_argument("--limit", type=int, default=200, help="max number of hotspots")
    args = parser.parse_args()

    nc_path = resolve_path(args.nc)
    out_path = resolve_path(args.out)
    export_hotspots(nc_path, args.th, out_path, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
