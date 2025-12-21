"""
Phase 15: End-to-End Demo (offline-safe).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from arcticroute.core.astar import plan_route_latlon
from arcticroute.core.config_paths import get_newenv_path
from arcticroute.core.cost import build_demo_cost
from arcticroute.core.grid import make_demo_grid
from arcticroute.io.cmems_loader import load_ice_drift_from_nc, load_sit_from_nc


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _stats(arr: np.ndarray | None) -> Dict[str, float] | None:
    if arr is None:
        return None
    vals = np.asarray(arr, dtype=float)
    if vals.size == 0:
        return None
    return {
        "min": float(np.nanmin(vals)),
        "mean": float(np.nanmean(vals)),
        "max": float(np.nanmax(vals)),
    }


def export_geojson(path: list, outfile: Path) -> None:
    features = []
    if path:
        coords = [[lon, lat] for lat, lon in path]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {"name": "route"},
            }
        )
    geojson = {"type": "FeatureCollection", "features": features}
    outfile.write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")


def export_summary(meta: Dict[str, Any], path: list, outfile: Path) -> None:
    lines = [
        "=" * 60,
        "Phase 15: End-to-End Demo Summary",
        "=" * 60,
        f"Timestamp: {_now_iso()}",
        "",
        "Route Result:",
        f"  Route Length (points): {len(path)}",
        f"  Route Found: {'Yes' if path else 'No'}",
        "",
        "SIT/Drift Summary:",
        f"  SIT loaded: {meta.get('sit_loaded', False)}",
        f"  Drift loaded: {meta.get('drift_loaded', False)}",
    ]

    sit_stats = meta.get("sit_stats")
    if sit_stats:
        lines.append(
            f"  SIT stats (min/mean/max): {sit_stats['min']:.4f} / "
            f"{sit_stats['mean']:.4f} / {sit_stats['max']:.4f}"
        )
    drift_stats = meta.get("drift_stats")
    if drift_stats:
        lines.append(
            f"  Drift speed stats (min/mean/max): {drift_stats['min']:.4f} / "
            f"{drift_stats['mean']:.4f} / {drift_stats['max']:.4f}"
        )

    lines.append("=" * 60)
    outfile.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 15 end-to-end demo (offline)")
    ap.add_argument("--outdir", default="reports/demo_run_phase15", help="Output directory")
    ap.add_argument("--start-lat", type=float, default=66.0)
    ap.add_argument("--start-lon", type=float, default=5.0)
    ap.add_argument("--end-lat", type=float, default=78.0)
    ap.add_argument("--end-lon", type=float, default=150.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    grid, land_mask = make_demo_grid()
    cf = build_demo_cost(grid, land_mask)
    path = plan_route_latlon(cf, args.start_lat, args.start_lon, args.end_lat, args.end_lon)

    meta: Dict[str, Any] = {}
    try:
        newenv_dir = get_newenv_path()
        sit_nc = newenv_dir / "ice_thickness.nc"
        drift_nc = newenv_dir / "ice_drift.nc"
        if sit_nc.exists():
            sit_arr, _ = load_sit_from_nc(sit_nc, grid=None)
            if sit_arr is not None:
                meta["sit_loaded"] = True
                meta["sit_stats"] = _stats(sit_arr)
        if drift_nc.exists():
            _, _, drift_speed, _ = load_ice_drift_from_nc(drift_nc, grid=None)
            if drift_speed is not None:
                meta["drift_loaded"] = True
                meta["drift_stats"] = _stats(drift_speed)
    except Exception as exc:
        meta["sit_loaded"] = False
        meta["drift_loaded"] = False
        meta["sit_drift_error"] = str(exc)

    export_geojson(path, outdir / "route.geojson")
    export_summary(meta, path, outdir / "summary.txt")
    print(f"[INFO] Results written to {outdir}")


if __name__ == "__main__":
    main()
