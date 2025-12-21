from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from arcticroute.io.cmems_loader import load_ice_drift_from_nc, load_sit_from_nc


def _load_json(path: Path) -> dict | None:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _latest_file(cache_dir: Path, pattern: str) -> Path | None:
    matches = sorted(cache_dir.glob(pattern))
    return matches[-1] if matches else None


def _stats(arr: np.ndarray | None) -> dict | None:
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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    resolved_path = root / "reports" / "cmems_resolved.json"
    cache_dir = root / "data" / "cmems_cache"
    newenv_dir = root / "data_processed" / "newenv"

    resolved = _load_json(resolved_path) or {}
    sit_res = resolved.get("sit", {})
    drift_res = resolved.get("drift", {})

    sit_cache = _latest_file(cache_dir, "sit_*.nc")
    drift_cache = _latest_file(cache_dir, "drift_*.nc")

    sit_newenv = newenv_dir / "ice_thickness.nc"
    drift_newenv = newenv_dir / "ice_drift.nc"

    print("Phase 15 Quick Check")
    print("=" * 60)
    print(f"cmems_resolved.json: {resolved_path if resolved_path.exists() else 'missing'}")
    print(f"sit status: {sit_res.get('status')} reason: {sit_res.get('reason')}")
    print(f"drift status: {drift_res.get('status')} reason: {drift_res.get('reason')}")
    print("")

    print("[CACHE]")
    print(f"sit cache: {sit_cache if sit_cache else 'missing'}")
    print(f"drift cache: {drift_cache if drift_cache else 'missing'}")
    print("")

    print("[NEWENV]")
    print(f"ice_thickness.nc: {sit_newenv if sit_newenv.exists() else 'missing'}")
    print(f"ice_drift.nc: {drift_newenv if drift_newenv.exists() else 'missing'}")

    sit_stats = None
    drift_stats = None
    if sit_newenv.exists():
        sit_arr, _ = load_sit_from_nc(sit_newenv, grid=None)
        sit_stats = _stats(sit_arr)
    if drift_newenv.exists():
        _, _, drift_speed, _ = load_ice_drift_from_nc(drift_newenv, grid=None)
        drift_stats = _stats(drift_speed)

    print("")
    if sit_stats:
        print(f"sit stats: min={sit_stats['min']:.4f} mean={sit_stats['mean']:.4f} max={sit_stats['max']:.4f}")
    else:
        print("sit stats: unavailable")
    if drift_stats:
        print(
            f"drift speed stats: min={drift_stats['min']:.4f} "
            f"mean={drift_stats['mean']:.4f} max={drift_stats['max']:.4f}"
        )
    else:
        print("drift speed stats: unavailable")


if __name__ == "__main__":
    main()
