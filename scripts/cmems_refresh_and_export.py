from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime, timedelta


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


def _build_time_window(days: int) -> tuple[str, str]:
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    return start.strftime("%Y-%m-%dT%H:%M:%S"), end.strftime("%Y-%m-%dT%H:%M:%S")


def _normalize_iso(value: str) -> str | None:
    if not value:
        return None
    value = value.strip()
    if len(value) == 10 and value.count("-") == 2:
        return f"{value}T00:00:00"
    return value


def _to_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _format_dt(value: datetime | None) -> str | None:
    return value.strftime("%Y-%m-%dT%H:%M:%S") if value else None


def _get_coverage_bounds(dataset_id: str) -> tuple[str | None, str | None]:
    cmd = [
        "copernicusmarine",
        "describe",
        "-i",
        dataset_id,
        "--return-fields",
        "datasets,temporal_coverage_start,temporal_coverage_end",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception:
        return None, None
    if result.returncode != 0:
        return None, None
    times = [
        _normalize_iso(t)
        for t in result.stdout.split()
        if _normalize_iso(t)
    ]
    times = [t for t in times if t and "T" in t]
    if not times:
        return None, None
    return min(times), max(times)


def _clamp_time_window(
    start: str,
    end: str,
    coverage_start: str | None,
    coverage_end: str | None,
) -> tuple[str, str, bool]:
    start_dt = _to_dt(_normalize_iso(start))
    end_dt = _to_dt(_normalize_iso(end))
    cov_start_dt = _to_dt(_normalize_iso(coverage_start or ""))
    cov_end_dt = _to_dt(_normalize_iso(coverage_end or ""))
    if start_dt is None or end_dt is None:
        return start, end, False

    clamped = False
    if cov_start_dt and start_dt < cov_start_dt:
        start_dt = cov_start_dt
        clamped = True
    if cov_end_dt and end_dt > cov_end_dt:
        end_dt = cov_end_dt
        clamped = True
    return _format_dt(start_dt), _format_dt(end_dt), clamped


def _run_subset(
    dataset_id: str,
    variables: list[str],
    bbox: list[float],
    time_start: str,
    time_end: str,
    output_dir: Path,
    output_file: str,
) -> tuple[bool, str]:
    if not dataset_id:
        return False, "dataset_id_missing"
    if not variables:
        return False, "variables_missing"

    start, end = time_start, time_end
    west, east, south, north = bbox
    cmd = [
        "copernicusmarine",
        "subset",
        "-i",
        dataset_id,
        "-t",
        start,
        "-T",
        end,
        "-x",
        str(west),
        "-X",
        str(east),
        "-y",
        str(south),
        "-Y",
        str(north),
        "-o",
        str(output_dir),
        "-f",
        output_file,
    ]
    for var in variables:
        cmd.extend(["-v", var])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    except Exception as exc:
        return False, f"subset_failed: {exc}"
    if result.returncode != 0:
        reason = result.stderr.strip() or result.stdout.strip() or "subset_failed"
        return False, reason[:500]
    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh CMEMS cache and export NetCDF")
    ap.add_argument("--days", type=int, default=2, help="Number of days to fetch")
    ap.add_argument("--start", default=None, help="Start time (YYYY-MM-DD or ISO)")
    ap.add_argument("--end", default=None, help="End time (YYYY-MM-DD or ISO)")
    ap.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("WEST", "EAST", "SOUTH", "NORTH"),
        default=[-40.0, 60.0, 65.0, 85.0],
        help="Bounding box W E S N",
    )
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
    sit_res = resolved.get("sit", {})
    drift_res = resolved.get("drift", {})
    sit_vars = sit_res.get("variables", [])
    drift_vars = drift_res.get("variables", [])
    sit_dataset = sit_res.get("dataset_id")
    drift_dataset = drift_res.get("dataset_id")

    sit_nc = None
    drift_nc = None
    sit_status = "skipped"
    drift_status = "skipped"
    reason = ""

    ts = time.strftime("%Y%m%d%H", time.localtime())
    requested_start, requested_end = _build_time_window(args.days)
    if args.start:
        requested_start = args.start
    if args.end:
        requested_end = args.end
    requested_start = _normalize_iso(requested_start) or requested_start
    requested_end = _normalize_iso(requested_end) or requested_end

    sit_time_used = None
    drift_time_used = None
    sit_time_requested = (requested_start, requested_end)
    drift_time_requested = (requested_start, requested_end)
    sit_time_clamped = False
    drift_time_clamped = False
    sit_time_reason = ""
    drift_time_reason = ""
    if args.enable_sit:
        cov_start, cov_end = _get_coverage_bounds(str(sit_dataset or ""))
        sit_start, sit_end, clamped = _clamp_time_window(
            requested_start, requested_end, cov_start, cov_end
        )
        if sit_start and sit_end and sit_start >= sit_end:
            sit_status = "skipped"
            sit_time_reason = "sit_time_window_empty_after_clamp"
            sit_time_used = None
        else:
            sit_time_used = (sit_start, sit_end)
            sit_time_clamped = clamped
            if sit_start and sit_end:
                out_name = f"sit_{ts}.nc"
                ok, err = _run_subset(
                    dataset_id=str(sit_dataset or ""),
                    variables=list(sit_vars),
                    bbox=list(args.bbox),
                    time_start=sit_start,
                    time_end=sit_end,
                    output_dir=cache_dir,
                    output_file=out_name,
                )
                if ok:
                    sit_nc = str(cache_dir / out_name)
                    sit_status = "ok_downloaded"
                else:
                    latest = _latest_file(cache_dir, "sit_*.nc")
                    if latest is not None:
                        sit_nc = str(latest)
                        sit_status = "ok_cached"
                        reason = f"sit_download_failed: {err}"
                    else:
                        sit_status = "skipped"
                        reason = f"sit_download_failed: {err}"
    else:
        sit_status = "skipped"
        reason = "sit_disabled"

    if args.enable_drift:
        cov_start, cov_end = _get_coverage_bounds(str(drift_dataset or ""))
        drift_start, drift_end, clamped = _clamp_time_window(
            requested_start, requested_end, cov_start, cov_end
        )
        if drift_start and drift_end and drift_start >= drift_end:
            drift_status = "skipped"
            drift_time_reason = "drift_time_window_empty_after_clamp"
            drift_time_used = None
        else:
            drift_time_used = (drift_start, drift_end)
            drift_time_clamped = clamped
            if drift_start and drift_end:
                out_name = f"drift_{ts}.nc"
                ok, err = _run_subset(
                    dataset_id=str(drift_dataset or ""),
                    variables=list(drift_vars),
                    bbox=list(args.bbox),
                    time_start=drift_start,
                    time_end=drift_end,
                    output_dir=cache_dir,
                    output_file=out_name,
                )
                if ok:
                    drift_nc = str(cache_dir / out_name)
                    drift_status = "ok_downloaded"
                else:
                    latest = _latest_file(cache_dir, "drift_*.nc")
                    if latest is not None:
                        drift_nc = str(latest)
                        drift_status = "ok_cached"
                        if not reason:
                            reason = f"drift_download_failed: {err}"
                    else:
                        drift_status = "skipped"
                        if not reason:
                            reason = f"drift_download_failed: {err}"
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
        "sit_dataset": sit_dataset,
        "drift_dataset": drift_dataset,
        "days": args.days,
        "time_requested": {
            "sit": sit_time_requested,
            "drift": drift_time_requested,
        },
        "time_used": {
            "sit": sit_time_used,
            "drift": drift_time_used,
        },
        "time_clamped": {
            "sit": sit_time_clamped,
            "drift": drift_time_clamped,
        },
        "time_reason": {
            "sit": sit_time_reason,
            "drift": drift_time_reason,
        },
        "bbox": list(args.bbox),
        "sit_status": sit_status,
        "drift_status": drift_status,
        "reason": reason,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    strategy_path = root / "reports" / "cmems_strategy.json"
    strategy = _load_json(strategy_path) or {}
    strategy["sit_time"] = {
        "requested": sit_time_requested,
        "used": sit_time_used,
        "clamped": sit_time_clamped,
        "reason": sit_time_reason,
    }
    strategy["drift_time"] = {
        "requested": drift_time_requested,
        "used": drift_time_used,
        "clamped": drift_time_clamped,
        "reason": drift_time_reason,
    }
    strategy["timestamp"] = _now_iso()
    strategy_path.write_text(json.dumps(strategy, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
