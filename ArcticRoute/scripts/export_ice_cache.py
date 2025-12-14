#!/usr/bin/env python3
"""Export cached computer-vision ice probability layers for downstream use.

@role: pipeline
"""

"""Export ice probability cache for a specific date/mission using cv_sat."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

# --- bootstrap imports ---
try:
    from ArcticRoute.scripts._modpath import ensure_path, get_cli_mod
except ModuleNotFoundError:
    HERE = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(HERE, "..", "..")),
        os.path.abspath(os.path.join(HERE, "..")),
        HERE,
    ]
    for candidate in candidates:
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    from scripts._modpath import ensure_path, get_cli_mod  # type: ignore[import]

ensure_path()
CLI_MODULE = get_cli_mod()
if CLI_MODULE == "ArcticRoute.api.cli":
    from ArcticRoute.api.cli import load_yaml_file
else:  # pragma: no cover - fallback path
    from api.cli import load_yaml_file
from ArcticRoute.config.schema import (
    ValidationError as SchemaValidationError,
    model_dump,
    validate_runtime_config,
)
from ArcticRoute.core.predictors.cv_sat import SatCVPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = PROJECT_ROOT / "config" / "runtime.yaml"

EXIT_INPUT_MISSING = 1
EXIT_DATA_FAILURE = 2
EXIT_COMPUTE_FAILURE = 3


def _parse_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"invalid date '{value}': {err}") from err
    return parsed.strftime("%Y-%m-%d")


def _infer_date_from_env(env_path: Path, tidx: int) -> str:
    with xr.open_dataset(env_path) as ds:
        if "time" not in ds.coords:
            raise ValueError("environment dataset lacks 'time' coordinate")
        times = ds["time"].values
        if times.size == 0:
            raise ValueError("environment dataset has empty time coordinate")
        idx = max(0, min(int(tidx), times.size - 1))
        target = np.datetime_as_string(times[idx], unit="D")
    return str(target)


def _format_bbox(bbox: Optional[Sequence[float]]) -> Optional[str]:
    if bbox is None:
        return None
    if len(bbox) != 4:
        raise ValueError("bbox must contain north, west, south, east")
    north, west, south, east = map(float, bbox)
    return f"{north},{west},{south},{east}"


def _collect_log_set(log_dir: Path, mission: str) -> set[Path]:
    return {path.resolve() for path in log_dir.glob(f"stac_results_{mission.lower()}_*.json")}


def _find_newest_log(log_dir: Path, mission: str, before: set[Path]) -> Optional[Path]:
    pattern = f"stac_results_{mission.lower()}_*.json"
    candidates = {path.resolve() for path in log_dir.glob(pattern)}
    new_paths = sorted(candidates - before, key=lambda p: p.stat().st_mtime, reverse=True)
    if new_paths:
        return new_paths[0]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def _ensure_mosaic(
    predictor: SatCVPredictor,
    *,
    date_str: str,
    mission: str,
    bbox_arg: Optional[str],
    env_path: Path,
    max_items: int,
    diag,
) -> Path:
    processed_dir = predictor.processed_dir
    date_token = date_str.replace("-", "")
    candidates = [
        processed_dir / f"sat_mosaic_{date_token}_{mission}.tif",
        processed_dir / f"sat_mosaic_{date_str}_{mission}.tif",
    ]
    for candidate in candidates:
        if candidate.exists():
            diag(f"Using existing mosaic {candidate}")
            return candidate

    logs_dir = predictor.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    bbox_value = bbox_arg
    if bbox_value is None:
        try:
            bbox_value = predictor._compute_bbox()  # type: ignore[attr-defined]
        except Exception as err:  # pragma: no cover - defensive
            raise RuntimeError(f"failed to derive bbox: {err}") from err

    fetch_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "stac_fetch.py"),
        "--bbox",
        bbox_value,
        "--date",
        date_str,
        "--mission",
        mission,
        "--source",
        "MPC",
        "--limit",
        str(max_items),
        "--lazy",
    ]
    diag(f"Fetching STAC items via: {' '.join(fetch_cmd)}")
    before_logs = _collect_log_set(logs_dir, mission)
    subprocess.run(fetch_cmd, cwd=PROJECT_ROOT, check=True)
    log_path = _find_newest_log(logs_dir, mission, before_logs)
    if log_path is None:
        raise RuntimeError("STAC fetch completed but no log file was created.")
    diag(f"STAC log resolved: {log_path}")

    mosaic_name = f"sat_mosaic_{date_token}_{mission}.tif"
    mosaic_path = processed_dir / mosaic_name
    mosaic_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "cog_mosaic_to_grid.py"),
        "--cogs",
        str(log_path),
        "--env",
        str(env_path),
        "--mission",
        mission,
        "--out",
        str(mosaic_path),
    ]
    diag(f"Generating mosaic via: {' '.join(mosaic_cmd)}")
    subprocess.run(mosaic_cmd, cwd=PROJECT_ROOT, check=True)
    if not mosaic_path.exists():
        raise RuntimeError(f"mosaic generation failed; missing {mosaic_path}")
    return mosaic_path


def _calc_nan_ratio(array: np.ndarray) -> float:
    if array.size == 0:
        return float("nan")
    return float(np.isnan(array).mean() * 100.0)


def _flatten_stats(values: np.ndarray) -> Dict[str, float]:
    finite = np.isfinite(values)
    if not np.count_nonzero(finite):
        raise ValueError("ice_prob contains no finite values")
    valid = values[finite]
    return {
        "mean": float(np.mean(valid)),
        "max": float(np.max(valid)),
        "coverage_pct": float(np.mean(valid > 0.5) * 100.0),
        "valid_ratio_pct": float(finite.sum() / values.size * 100.0),
    }


def _band_validity(sat_da: Optional[xr.DataArray], mission: str) -> Dict[str, Optional[float]]:
    if sat_da is None or "band" not in sat_da.dims:
        return {}
    bands = [str(b).upper() for b in sat_da.coords["band"].values]
    required = ("B03", "B08") if mission.upper() == "S2" else ("VV", "VH")
    stats: Dict[str, Optional[float]] = {}
    for band in required:
        if band in bands:
            arr = sat_da.sel(band=band)
            values = np.asarray(arr.values, dtype=np.float32)
            ratio = float(np.isfinite(values).sum() / values.size * 100.0) if values.size else 0.0
            stats[band] = ratio
        else:
            stats[band] = None
    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export ice probability cache via cv_sat predictor.")
    parser.add_argument("--cfg", default=str(DEFAULT_CFG), help="Runtime config file (default: config/runtime.yaml)")
    parser.add_argument("--tidx", type=int, default=0, help="Time index to process (default: 0)")
    parser.add_argument("--date", type=_parse_date, help="Target acquisition date (YYYY-MM-DD)")
    parser.add_argument("--mission", help="Satellite mission override (S2/S1)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose diagnostics")
    args = parser.parse_args(argv)

    verbose = args.verbose or os.environ.get("LOG_LEVEL", "").strip().lower() in {"debug", "verbose"}

    def diag(message: str) -> None:
        if verbose:
            print(f"[DIAG] {message}")

    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        print(f"[ERROR] Config file not found: {cfg_path}")
        return EXIT_INPUT_MISSING

    raw_config = load_yaml_file(cfg_path)
    try:
        runtime_model = validate_runtime_config(raw_config)
    except SchemaValidationError as err:
        print(f"[ERROR] Invalid runtime config: {err}")
        return EXIT_INPUT_MISSING
    config = model_dump(runtime_model)

    data_block = config.get("data") or {}
    env_nc_value = data_block.get("env_nc")
    if not env_nc_value:
        print("[ERROR] data.env_nc missing in config")
        return EXIT_INPUT_MISSING

    env_path = Path(env_nc_value)
    if not env_path.is_absolute():
        env_path = PROJECT_ROOT / env_path
    if not env_path.exists():
        print(f"[ERROR] Environment NetCDF not found: {env_path}")
        return EXIT_INPUT_MISSING

    mission = args.mission or (config.get("predictor_params") or {}).get("cv_sat", {}).get("mission") or "S2"
    mission = str(mission).upper()
    max_items = int((config.get("predictor_params") or {}).get("cv_sat", {}).get("max_items", 4) or 4)
    var_name = (config.get("run") or {}).get("var", "risk_env")

    try:
        date_str = args.date or _infer_date_from_env(env_path, args.tidx)
    except Exception as err:
        print(f"[ERROR] Failed to determine target date: {err}")
        return EXIT_INPUT_MISSING

    predictor = SatCVPredictor(
        env_path=env_path,
        var_name=var_name,
        mission=mission,
        max_items=max_items,
    )

    print(f"[EXPORT] Target date={date_str}, mission={mission}")
    crop_block = config.get("crop")
    bbox_arg = None
    if isinstance(crop_block, dict) and crop_block.get("bbox") is not None:
        bbox_arg = _format_bbox(crop_block["bbox"])

    try:
        mosaic_path = _ensure_mosaic(
            predictor,
            date_str=date_str,
            mission=mission,
            bbox_arg=bbox_arg,
            env_path=env_path,
            max_items=max_items,
            diag=diag,
        )
    except subprocess.CalledProcessError as err:
        print(f"[ERROR] External command failed (step={' '.join(err.cmd)}): {err}")
        return EXIT_DATA_FAILURE
    except Exception as err:
        print(f"[ERROR] Unable to prepare mosaic: {err}")
        return EXIT_DATA_FAILURE

    diag(f"Mosaic target: {mosaic_path}")
    if mosaic_path.exists():
        size_mb = mosaic_path.stat().st_size / (1024 * 1024)
        print(f"[EXPORT] Mosaic resolved: {mosaic_path} ({size_mb:.2f} MiB)")
    else:
        print(f"[ERROR] Mosaic file missing after generation: {mosaic_path}")
        return EXIT_DATA_FAILURE

    try:
        dataset = predictor.prepare(args.tidx)
    except Exception as err:
        message = str(err)
        print(f"[ERROR] SatCVPredictor failed: {message}")
        code = EXIT_COMPUTE_FAILURE if "otsu" in message.lower() or "threshold" in message.lower() else EXIT_DATA_FAILURE
        return code

    sat_da = dataset.data_vars.get("sat_bands") if isinstance(dataset, xr.Dataset) else None
    band_stats = _band_validity(sat_da, mission)
    if band_stats:
        summary = ", ".join(
            f"{band}={ratio:.2f}%" if ratio is not None else f"{band}=missing" for band, ratio in band_stats.items()
        )
        print(f"[EXPORT] Band validity: {summary}")
    else:
        diag("Band validity: unavailable")
    if sat_da is not None:
        sat_shape = tuple(int(dim) for dim in sat_da.shape)
        sat_nan_ratio = _calc_nan_ratio(np.asarray(sat_da.values, dtype=np.float32))
        diag(f"sat_bands shape={sat_shape}, NaN%={sat_nan_ratio:.2f}")

    ice_info = {}
    if isinstance(dataset, xr.Dataset):
        ice_info = dataset.attrs.get("ice_prob_info") or {}

    ice_da: Optional[xr.DataArray]
    if isinstance(dataset, xr.Dataset) and "ice_prob" in dataset:
        ice_da = dataset["ice_prob"]
    elif isinstance(dataset, xr.DataArray) and dataset.name == "ice_prob":
        ice_da = dataset
    else:
        ice_da = None

    diag(
        f"Diagnostics: date={date_str}, mission={mission}, "
        f"ice_prob_available={'yes' if ice_da is not None else 'no'}, info={ice_info}"
    )

    cache_dir = predictor.cv_cache_dir
    cache_nc = cache_dir / "ice_prob_latest.nc"
    cache_json = cache_dir / "ice_prob_latest.json"

    if ice_da is None:
        reason = ice_info.get("reason", "unknown")
        print(
            "[ERROR] Mosaic present but ice_prob missing "
            f"(reason={reason} - missing bands / all NaN / threshold failure / dependency issue)."
        )
        return EXIT_COMPUTE_FAILURE

    ice_values = np.asarray(ice_da.values, dtype=np.float32)
    try:
        stats = _flatten_stats(ice_values)
    except ValueError as err:
        print(f"[ERROR] {err}")
        return EXIT_COMPUTE_FAILURE

    threshold = ice_da.attrs.get("threshold")
    ndwi_mean = ice_da.attrs.get("ndwi_mean")
    ndwi_std = ice_da.attrs.get("ndwi_std")
    valid_count = ice_da.attrs.get("valid_count")
    print(
        "[EXPORT] Otsu stats: "
        f"mean={ndwi_mean if ndwi_mean is not None else 'nan'}, "
        f"std={ndwi_std if ndwi_std is not None else 'nan'}, "
        f"threshold={threshold if threshold is not None else 'nan'}, "
        f"valid_pixels={valid_count if valid_count is not None else 'unknown'}"
    )
    print(
        "[EXPORT] ice_prob stats: "
        f"mean={stats['mean']:.4f}, max={stats['max']:.4f}, "
        f"coverage={stats['coverage_pct']:.2f}%, valid={stats['valid_ratio_pct']:.2f}%"
    )

    if not cache_nc.exists() or not cache_json.exists():
        print("[ERROR] Cache files missing after save operation.")
        return EXIT_DATA_FAILURE

    summary = json.loads(cache_json.read_text(encoding="utf-8"))
    print(
        "[EXPORT] ice_prob cached -> nc={nc}, json={json}, mean={mean:.4f}, max={max:.4f}, "
        "coverage={coverage:.2f}%, valid={valid:.2f}%, threshold={threshold}".format(
            nc=cache_nc,
            json=cache_json,
            mean=stats["mean"],
            max=stats["max"],
            coverage=stats["coverage_pct"],
            valid=stats["valid_ratio_pct"],
            threshold=f"{threshold:.4f}" if isinstance(threshold, (int, float)) and np.isfinite(threshold) else "nan",
        )
    )
    print(f"[EXPORT] Cache summary: {json.dumps(summary, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

