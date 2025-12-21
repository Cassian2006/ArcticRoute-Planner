from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _latest_file(cache_dir: Path, pattern: str) -> Path | None:
    matches = sorted(cache_dir.glob(pattern))
    return matches[-1] if matches else None


def _list_vars(nc_path: Path) -> list[str]:
    try:
        import xarray as xr  # noqa: WPS433
    except Exception:
        return []
    try:
        with xr.open_dataset(nc_path, decode_times=False) as ds:
            return list(ds.data_vars)
    except Exception:
        return []


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync CMEMS cache to data_processed/newenv")
    ap.add_argument("--dry-run", action="store_true", help="Do not copy files")
    ap.add_argument(
        "--output",
        default="reports/cmems_newenv_index.json",
        help="Output index JSON path",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cache_dir = root / "data" / "cmems_cache"
    newenv_dir = root / "data_processed" / "newenv"
    newenv_dir.mkdir(parents=True, exist_ok=True)

    sit_src = _latest_file(cache_dir, "sit_*.nc")
    drift_src = _latest_file(cache_dir, "drift_*.nc")

    sit_dst = newenv_dir / "ice_thickness.nc"
    drift_dst = newenv_dir / "ice_drift.nc"

    sit_vars = _list_vars(sit_src) if sit_src else []
    drift_vars = _list_vars(drift_src) if drift_src else []

    ok = False
    reason = ""

    if sit_src and not args.dry_run:
        shutil.copy2(sit_src, sit_dst)
        ok = True
    if drift_src and not args.dry_run:
        shutil.copy2(drift_src, drift_dst)
        ok = True

    if not ok:
        reason = "cache_dir_missing_or_empty"

    out = {
        "timestamp": _now_iso(),
        "cache_dir": str(cache_dir),
        "newenv_dir": str(newenv_dir),
        "sic_src": None,
        "swh_src": None,
        "sic_dst": None,
        "swh_dst": None,
        "sit_src": str(sit_src) if sit_src else None,
        "drift_src": str(drift_src) if drift_src else None,
        "sit_dst": str(sit_dst) if sit_src else None,
        "drift_dst": str(drift_dst) if drift_src else None,
        "sit_vars": sit_vars,
        "drift_vars": drift_vars,
        "sit_present": bool(sit_src),
        "drift_present": bool(drift_src),
        "dry_run": bool(args.dry_run),
        "ok": ok,
        "reason": reason,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
