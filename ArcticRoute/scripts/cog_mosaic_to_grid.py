#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mosaic Sentinel COG assets onto the environmental analysis grid (GeoTIFF + sidecar).

@role: pipeline
"""

"""
Mosaic Sentinel COG assets onto the environmental analysis grid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from ArcticRoute.io.stac_ingest import (  # noqa: E402
    build_asset_access_params,
    collect_cog_hrefs,
    extract_env_grid,
)

DEFAULT_BANDS: Dict[str, Tuple[str, ...]] = {
    "S2": ("B02", "B03", "B04", "B08"),
    "S1": ("VV", "VH"),
}
DST_CRS = "EPSG:4326"


def load_env_defaults() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


load_env_defaults()


def resolve_project_path(env_key: str, default: str) -> Path:
    value = os.environ.get(env_key, default)
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


CACHE_DIR = resolve_project_path("SAT_CACHE_DIR", "data_processed/sat_cache")


@contextmanager
def rasterio_auth_env(href: str):
    if href.lower().startswith("s3://"):
        raise RuntimeError(f"S3 access disabled for mosaic inputs (expected HTTPS): {href}")
    headers, auth = build_asset_access_params(href)
    env_kwargs: Dict[str, str] = {}
    if headers:
        header_lines = [f"{key}: {value}" for key, value in headers.items()]
        env_kwargs["GDAL_HTTP_HEADERS"] = "\n".join(header_lines)
    if auth:
        env_kwargs["GDAL_HTTP_USERPWD"] = f"{auth.username}:{auth.password}"
    if env_kwargs:
        with rasterio.Env(**env_kwargs):
            yield
    else:
        yield


def reproject_to_grid(
    href: str,
    dst_transform,
    dst_shape: Tuple[int, int],
    dst_crs: str,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    data = np.full(dst_shape, np.nan, dtype=np.float32)
    with rasterio_auth_env(href):
        with rasterio.open(href) as src:
            if src.count < 1:
                raise ValueError("No bands found in source raster.")
            src_nodata = src.nodata
            profile_nodata = src.profile.get("nodata")
            if src_nodata is None and isinstance(profile_nodata, (int, float)):
                src_nodata = profile_nodata
            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs or dst_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                src_nodata=src_nodata,
                dst_nodata=np.nan,
                resampling=resampling,
                init_dest_nodata=True,
            )
    return data


def _first_valid(stack: np.ndarray) -> np.ndarray:
    result = np.full(stack.shape[1:], np.nan, dtype=np.float32)
    for layer in stack:
        mask = np.isnan(result) & np.isfinite(layer)
        if not np.any(mask):
            continue
        result[mask] = layer[mask].astype(np.float32)
        if not np.isnan(result).any():
            break
    return result


def _reduce_stack(stack: Sequence[np.ndarray], dst_shape: Tuple[int, int], mode: str) -> np.ndarray:
    if not stack:
        return np.full(dst_shape, np.nan, dtype=np.float32)
    stack_arr = np.stack(stack, axis=0).astype(np.float32)
    mode_token = mode.lower()
    with np.errstate(all="ignore"):
        if mode_token == "median":
            reduced = np.nanmedian(stack_arr, axis=0)
        elif mode_token == "mean":
            reduced = np.nanmean(stack_arr, axis=0)
        else:
            reduced = _first_valid(stack_arr)
    return reduced.astype(np.float32)


def mosaic_band(
    hrefs: Sequence[str],
    dst_transform,
    dst_shape: Tuple[int, int],
    dst_crs: str,
    mode: str,
) -> Tuple[np.ndarray, List[str], Dict[str, str]]:
    stack: List[np.ndarray] = []
    used: List[str] = []
    failures: Dict[str, str] = {}
    for href in hrefs:
        if href.lower().startswith("s3://"):
            failures[href] = "S3 access disabled; expected signed HTTPS asset."
            print(f"[WARN] Skipping S3 asset (HTTP required): {href}")
            continue
        try:
            array = reproject_to_grid(href, dst_transform, dst_shape, dst_crs)
        except Exception as err:  # pragma: no cover - IO heavy
            failures[href] = str(err)
            print(f"[WARN] Failed to ingest {href}: {err}")
            continue
        if not np.isfinite(array).any():
            continue
        stack.append(array.astype(np.float32))
        used.append(href)
    result = _reduce_stack(stack, dst_shape, mode)
    return result, used, failures


def compute_band_stats(array: np.ndarray) -> Dict[str, float | int]:
    finite = np.isfinite(array)
    total = int(array.size)
    valid = int(finite.sum())
    if valid == 0:
        return {"valid_pixels": 0, "total_pixels": total, "coverage_pct": 0.0}
    values = array[finite]
    stats: Dict[str, float | int] = {
        "valid_pixels": valid,
        "total_pixels": total,
        "coverage_pct": round((valid / total) * 100.0, 4),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "mean": float(np.nanmean(values)),
    }
    return stats


def compute_sha1(path: Path) -> Tuple[str, Path]:
    sha1 = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if not chunk:
                break
            sha1.update(chunk)
    digest = sha1.hexdigest()
    sha_path = path.with_suffix(path.suffix + ".sha1")
    sha_path.write_text(digest + "\n", encoding="utf-8")
    return digest, sha_path


def write_metadata(
    output_path: Path,
    mission: str,
    band_payload: List[Tuple[str, Sequence[str], Dict[str, float | int]]],
    lat_values: Iterable[float],
    lon_values: Iterable[float],
    transform,
    unused: Sequence[str],
    failures: Dict[str, str],
    sha1_hex: str,
    mosaic_mode: str,
) -> Tuple[Path, Path]:
    lat_list = [float(value) for value in lat_values]
    lon_list = [float(value) for value in lon_values]
    resolution_x = abs(float(getattr(transform, "a", 0.0)))
    resolution_y = abs(float(getattr(transform, "e", 0.0)))
    summary = {
        "output": str(output_path),
        "mission": mission,
        "created": datetime.now(timezone.utc).isoformat(),
        "sha1": sha1_hex,
        "mosaic_mode": mosaic_mode,
        "grid": {
            "rows": len(lat_list),
            "cols": len(lon_list),
            "lat_range": [min(lat_list), max(lat_list)] if lat_list else None,
            "lon_range": [min(lon_list), max(lon_list)] if lon_list else None,
            "resolution_deg": {"lon": resolution_x, "lat": resolution_y},
        },
        "bands": [],
        "unused_inputs": list(dict.fromkeys(unused)),
        "failures": failures,
    }
    for band, sources, stats in band_payload:
        summary["bands"].append(
            {
                "band": band,
                "sources": list(sources),
                "stats": stats,
            }
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{output_path.stem}.json"
    cache_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    sidecar_path = output_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return cache_path, sidecar_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mosaic Sentinel COGs to the environmental grid.")
    parser.add_argument("--cogs", required=True, help="Source glob / JSON payload / STAC log filename.")
    parser.add_argument("--env", required=True, help="Template NetCDF file (e.g. data_processed/env_clean.nc).")
    parser.add_argument("--mission", required=True, choices=["S1", "S2"], help="Mission identifier.")
    parser.add_argument("--out", required=True, help="Output GeoTIFF path.")
    parser.add_argument("--resampling", default="bilinear", choices=["bilinear", "nearest"], help="Resampling kernel.")
    parser.add_argument(
        "--mosaic",
        default=os.getenv("AR_CV_MOSAIC", "median"),
        choices=["median", "mean", "best"],
        help="Aggregation strategy for overlapping pixels (default: env AR_CV_MOSAIC or median).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mission = args.mission.upper()
    band_order = DEFAULT_BANDS.get(mission)
    if not band_order:
        print(f"[ERR] Unsupported mission {mission}")
        return 2

    env_path = (PROJECT_ROOT / args.env).resolve()
    out_path = (PROJECT_ROOT / args.out).resolve()
    resampling = Resampling.nearest if args.resampling == "nearest" else Resampling.bilinear
    mosaic_mode = (args.mosaic or "median").lower()
    print(f"[INFO] Mosaic mode: {mosaic_mode}")

    band_map, unused = collect_cog_hrefs(args.cogs, mission, band_order)
    if unused:
        for entry in dict.fromkeys(unused):
            print(f"[WARN] Unused COG entry ignored: {entry}")

    lat_values, lon_values, dst_transform = extract_env_grid(env_path)
    dst_shape = (len(lat_values), len(lon_values))
    mosaic_payload: List[Tuple[str, np.ndarray, List[str]]] = []
    all_failures: Dict[str, str] = {}

    for band in band_order:
        hrefs = band_map.get(band, [])
        if not hrefs:
            print(f"[WARN] Missing band {band}; skipping.")
            continue
        print(f"[INFO] Processing band {band} ({len(hrefs)} asset(s))")
        array, used, failures = mosaic_band(hrefs, dst_transform, dst_shape, DST_CRS, mosaic_mode)
        all_failures.update(failures)
        if not used:
            print(f"[WARN] Band {band} produced no valid pixels; skipped.")
            continue
        mosaic_payload.append((band, array, used))

    if not mosaic_payload:
        print("[ERR] No usable bands were produced; aborting.")
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        width=dst_shape[1],
        height=dst_shape[0],
        count=len(mosaic_payload),
        dtype="float32",
        crs=DST_CRS,
        transform=dst_transform,
        nodata=np.nan,
        tiled=True,
        compress="deflate",
    ) as dst:
        for index, (band, array, _) in enumerate(mosaic_payload, start=1):
            dst.write(array.astype("float32"), index)
            dst.set_band_description(index, band)

    stats_payload = [
        (band, sources, compute_band_stats(array)) for band, array, sources in mosaic_payload
    ]
    sha1_hex, sha1_path = compute_sha1(out_path)
    cache_path, sidecar_path = write_metadata(
        out_path,
        mission,
        stats_payload,
        lat_values,
        lon_values,
        dst_transform,
        unused,
        all_failures,
        sha1_hex,
        mosaic_mode,
    )

    res_x = abs(float(dst_transform.a))
    res_y = abs(float(dst_transform.e))
    print(f"[OK] Mosaic saved to {out_path}")
    print(f"[OK] Sidecar JSON: {sidecar_path}")
    print(f"[OK] Cache summary: {cache_path}")
    print(f"[OK] SHA1 digest ({sha1_hex}) -> {sha1_path}")
    print(f"[INFO] Output bands: {[band for band, _, _ in mosaic_payload]}")
    print(f"[INFO] Output shape: (rows={dst_shape[0]}, cols={dst_shape[1]})")
    print(f"[INFO] Resolution (deg): lon={res_x}, lat={res_y}; CRS={DST_CRS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
