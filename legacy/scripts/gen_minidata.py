#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


@dataclass
class Paths:
    raster: Path
    ais: Path
    coastline: Path

    @property
    def parent(self) -> Path:
        return self.raster.parent


def _normalised_double_peak(size: int) -> np.ndarray:
    """Create a synthetic raster with two gaussian peaks."""
    y_grid, x_grid = np.mgrid[0:size, 0:size]

    def gaussian(x0: float, y0: float, sigma: float) -> np.ndarray:
        return np.exp(-((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / (2.0 * sigma**2))

    peak_a = gaussian(size * 0.25, size * 0.35, sigma=size * 0.12)
    peak_b = gaussian(size * 0.7, size * 0.6, sigma=size * 0.18)
    valley = gaussian(size * 0.5, size * 0.8, sigma=size * 0.5)

    raster = peak_a + 0.8 * peak_b - 0.2 * valley
    raster -= raster.min()
    raster /= np.ptp(raster) or 1.0
    return raster.astype(np.float32)


def write_demo_raster(output: Path, size: int = 64) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = _normalised_double_peak(size)

    pixel_size_deg = 0.05
    transform = from_origin(-20.0, 80.0, pixel_size_deg, pixel_size_deg)

    profile = {
        "driver": "GTiff",
        "height": size,
        "width": size,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "lzw",
    }

    with rasterio.open(output, "w", **profile) as dst:
        dst.write(data[np.newaxis, ...])


def write_demo_ais(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    features = [
        {
            "type": "Feature",
            "properties": {
                "mmsi": "999000111",
                "speed_kn": 12.5,
                "heading": 45,
            },
            "geometry": {"type": "Point", "coordinates": [-19.5, 79.7]},
        },
        {
            "type": "Feature",
            "properties": {
                "mmsi": "999000222",
                "speed_kn": 8.1,
                "heading": 120,
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [-19.0, 79.8],
                    [-18.7, 79.75],
                    [-18.4, 79.65],
                ],
            },
        },
    ]
    payload = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(payload, indent=2))


def write_demo_coastline(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    polygon = {
        "type": "Feature",
        "properties": {"name": "Stub Coastline"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-19.8, 79.9],
                    [-18.2, 79.9],
                    [-18.2, 79.6],
                    [-19.8, 79.6],
                    [-19.8, 79.9],
                ]
            ],
        },
    }

    multipolygon = {
        "type": "Feature",
        "properties": {"name": "Tiny Island"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-19.3, 79.72],
                    [-19.1, 79.72],
                    [-19.1, 79.65],
                    [-19.3, 79.65],
                    [-19.3, 79.72],
                ]
            ],
        },
    }

    payload = {"type": "FeatureCollection", "features": [polygon, multipolygon]}
    path.write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate minimal offline demo datasets.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "samples",
        help="Directory where demo files will be written (default: data/samples).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=64,
        help="Raster side length in pixels (default: 64).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths(
        raster=args.output_dir / "sat_demo.tif",
        ais=args.output_dir / "ais_demo.geojson",
        coastline=args.output_dir / "coastline_stub.geojson",
    )

    write_demo_raster(paths.raster, size=args.size)
    write_demo_ais(paths.ais)
    write_demo_coastline(paths.coastline)

    total_size = sum(p.stat().st_size for p in (paths.raster, paths.ais, paths.coastline) if p.exists())
    total_mb = total_size / (1024 * 1024)
    print(f"[gen_minidata] Wrote demo data into {paths.parent} (total ~{total_mb:.3f} MB)")
    for out_file in (paths.raster, paths.ais, paths.coastline):
        if out_file.exists():
            size_kb = out_file.stat().st_size / 1024
            print(f"  - {out_file.name}: {size_kb:.1f} KiB")


if __name__ == "__main__":
    main()
