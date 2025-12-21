from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh CMEMS cache and export NetCDF")
    ap.add_argument("--enable-sit", action="store_true", help="Enable SIT download/copy")
    ap.add_argument("--enable-drift", action="store_true", help="Enable ice drift download/copy")
    ap.add_argument(
        "--resolved",
        default="reports/cmems_resolved.json",
        help="Resolved CMEMS metadata JSON",
    )
    ap.add_argument(
        "--output",
        default="reports/cmems_refresh_last.json",
        help="Refresh metadata output JSON",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cache_dir = root / "data" / "cmems_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    resolved = _load_json(Path(args.resolved)) or {}
    sit_vars = resolved.get("sit", {}).get("variables", [])
    drift_vars = resolved.get("drift", {}).get("variables", [])

    sit_nc = None
    drift_nc = None
    sit_status = "skipped"
    drift_status = "skipped"
    reason = ""

    if args.enable_sit:
        latest = _latest_file(cache_dir, "sit_*.nc")
        if latest is not None:
            sit_nc = str(latest)
            sit_status = "ok_cached"
        else:
            sit_status = "skipped"
            reason = "sit_cache_missing"
    else:
        sit_status = "skipped"
        reason = "sit_disabled"

    if args.enable_drift:
        latest = _latest_file(cache_dir, "drift_*.nc")
        if latest is not None:
            drift_nc = str(latest)
            drift_status = "ok_cached"
        else:
            drift_status = "skipped"
            reason = "drift_cache_missing"
    else:
        drift_status = "skipped"
        if not reason:
            reason = "drift_disabled"

    out = {
        "timestamp": _now_iso(),
        "cache_dir": str(cache_dir),
        "sit_nc": sit_nc,
        "drift_nc": drift_nc,
        "sit_vars": sit_vars,
        "drift_vars": drift_vars,
        "sit_status": sit_status,
        "drift_status": drift_status,
        "reason": reason,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
