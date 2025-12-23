# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import xarray as xr

from arcticroute.core.cost import compute_grid_signature


def _safe_mtime(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts).isoformat(timespec="seconds")
    except Exception:
        return ""


def _as_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        return Path(text)
    except Exception:
        return None


def _add_dir(dirs: list[Path], path: Path | None) -> None:
    if path is None:
        return
    try:
        p = path.resolve()
    except Exception:
        p = path
    if p not in dirs:
        dirs.append(p)


def build_search_dirs(
    data_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> list[Path]:
    """
    Build a list of search directories for multi-source data discovery.
    """
    try:
        project_root = Path(__file__).resolve().parents[2]
    except Exception:
        project_root = Path.cwd()

    dirs: list[Path] = []

    _add_dir(dirs, project_root / "data_real")
    _add_dir(dirs, project_root / "data")

    root_override = _as_path(data_root)
    if root_override is not None:
        _add_dir(dirs, root_override)
        _add_dir(dirs, root_override / "data_real")
        _add_dir(dirs, root_override / "data")
        _add_dir(dirs, root_override / "data_processed")
        _add_dir(dirs, root_override / "data_processed" / "newenv")
        _add_dir(dirs, root_override / "newenv")

    env_root = _as_path(os.getenv("ARCTICROUTE_DATA_ROOT"))
    if env_root is not None:
        _add_dir(dirs, env_root)
        _add_dir(dirs, env_root / "data_real")
        _add_dir(dirs, env_root / "data")
        _add_dir(dirs, env_root / "data_processed")
        _add_dir(dirs, env_root / "data_processed" / "newenv")
        _add_dir(dirs, env_root / "newenv")

    manifest = _as_path(manifest_path)
    if manifest is not None:
        _add_dir(dirs, manifest.parent)

    xin_env = _as_path(
        os.getenv("ARCTICROUTE_XINSHUJU_DIR")
        or os.getenv("XINSHUJU_DIR")
        or os.getenv("ARCTICROUTE_DESKTOP_DATA_DIR")
    )
    if xin_env is not None:
        _add_dir(dirs, xin_env)
    else:
        try:
            desktop = Path(os.environ.get("USERPROFILE", "")) / "Desktop" / "xinshuju"
        except Exception:
            desktop = None
        if desktop and desktop.exists():
            _add_dir(dirs, desktop)

    existing: list[Path] = []
    for d in dirs:
        try:
            if d.exists() and d not in existing:
                existing.append(d)
        except Exception:
            continue

    return existing


def _expand_ais_dirs(search_dirs: Iterable[Path]) -> list[Path]:
    expanded: list[Path] = []
    for base in search_dirs:
        _add_dir(expanded, base)
        _add_dir(expanded, base / "ais")
        _add_dir(expanded, base / "ais" / "density")
        _add_dir(expanded, base / "ais" / "derived")
    return expanded


def _extract_var_and_shape(ds: xr.Dataset, preferred: list[str]) -> Tuple[str | None, tuple[int, ...] | None]:
    for key in preferred:
        if key in ds:
            try:
                return key, tuple(int(s) for s in ds[key].shape)
            except Exception:
                return key, None
    if ds.data_vars:
        key = list(ds.data_vars.keys())[0]
        try:
            return key, tuple(int(s) for s in ds[key].shape)
        except Exception:
            return key, None
    return None, None


def discover_ais_density(search_dirs: list[Path], grid) -> tuple[pd.DataFrame, dict]:
    """
    Discover AIS density files under search_dirs.
    Returns (dataframe, meta). Failures return empty dataframe + reason in meta.
    """
    try:
        grid_signature = None
        grid_shape = None
        if isinstance(grid, str):
            grid_signature = grid
        elif grid is not None:
            try:
                grid_signature = compute_grid_signature(grid)
                grid_shape = getattr(grid, "lat2d", None)
                if grid_shape is not None:
                    grid_shape = tuple(int(s) for s in grid_shape.shape)
            except Exception:
                grid_signature = None

        rows = []
        expanded_dirs = _expand_ais_dirs(search_dirs)
        for folder in expanded_dirs:
            if not folder.exists():
                continue
            for path in sorted(folder.glob("*.nc")):
                name_lower = path.name.lower()
                if "density" not in name_lower and "ais_density" not in name_lower:
                    continue
                try:
                    with xr.open_dataset(path) as ds:
                        var_name, shape = _extract_var_and_shape(ds, ["ais_density", "density", "ais"])
                        file_sig = ds.attrs.get("grid_signature")
                except Exception as e:
                    rows.append(
                        {
                            "path": str(path),
                            "grid_signature": "",
                            "shape": "",
                            "mtime": _safe_mtime(path),
                            "match": f"read failed: {e}",
                        }
                    )
                    continue

                if grid_signature and file_sig:
                    match_reason = "signature match" if file_sig == grid_signature else "signature mismatch"
                elif grid_shape and shape:
                    match_reason = "shape match" if tuple(shape) == tuple(grid_shape) else "shape mismatch"
                elif file_sig:
                    match_reason = "missing current grid signature"
                else:
                    match_reason = "missing signature"

                rows.append(
                    {
                        "path": str(path),
                        "grid_signature": file_sig or "",
                        "shape": "x".join(str(s) for s in shape) if shape else "",
                        "mtime": _safe_mtime(path),
                        "match": match_reason,
                    }
                )

        df = pd.DataFrame(rows)
        latest_path = ""
        latest_mtime = ""
        if not df.empty and "mtime" in df:
            try:
                df_sorted = df.sort_values("mtime", ascending=False)
                latest_path = df_sorted.iloc[0]["path"]
                latest_mtime = df_sorted.iloc[0]["mtime"]
            except Exception:
                pass

        meta = {
            "count": int(len(df)),
            "latest_path": latest_path,
            "latest_mtime": latest_mtime,
            "grid_signature": grid_signature or "",
            "scan_dirs": [str(d) for d in expanded_dirs],
        }
        return df, meta
    except Exception as e:
        empty = pd.DataFrame(columns=["path", "grid_signature", "shape", "mtime", "match"])
        return empty, {"reason": f"discover_ais_density failed: {e}", "count": 0}


def discover_newenv_cmems(newenv_dir: str | Path | None) -> dict:
    """
    Discover CMEMS newenv files (SIC/SIT/SWH/Drift). Failures return empty meta + reason.
    """
    try:
        newenv_path = _as_path(newenv_dir)
        if newenv_path is None:
            return {"reason": "newenv_dir not set", "files": {}}

        reference_shape = None
        for ref_name in ["env_clean.nc", "grid_spec.nc"]:
            ref_path = newenv_path / ref_name
            if ref_path.exists():
                try:
                    with xr.open_dataset(ref_path) as ds:
                        _, reference_shape = _extract_var_and_shape(ds, list(ds.data_vars.keys()))
                        break
                except Exception:
                    continue

        file_specs = {
            "sic": ("ice_copernicus_sic.nc", ["sic", "ice_concentration", "sea_ice_concentration"]),
            "sit": ("ice_thickness.nc", ["sit", "ice_thickness", "sea_ice_thickness"]),
            "swh": ("wave_swh.nc", ["swh", "wave_swh", "significant_wave_height"]),
            "drift": ("ice_drift.nc", ["uice", "vice", "drift_u", "drift_v", "ice_drift"]),
        }

        files_meta: dict[str, dict] = {}
        for key, (fname, candidates) in file_specs.items():
            path = newenv_path / fname
            if not path.exists():
                files_meta[key] = {
                    "exists": False,
                    "path": str(path),
                    "mtime": "",
                    "vars": [],
                    "shape": "",
                    "shape_match": None,
                    "reason": "file not found",
                }
                continue
            try:
                with xr.open_dataset(path) as ds:
                    var_name, shape = _extract_var_and_shape(ds, candidates)
                    files_meta[key] = {
                        "exists": True,
                        "path": str(path),
                        "mtime": _safe_mtime(path),
                        "vars": [var_name] if var_name else list(ds.data_vars.keys()),
                        "shape": "x".join(str(s) for s in shape) if shape else "",
                        "shape_match": (tuple(shape) == tuple(reference_shape)) if (shape and reference_shape) else None,
                    }
            except Exception as e:
                files_meta[key] = {
                    "exists": False,
                    "path": str(path),
                    "mtime": _safe_mtime(path),
                    "vars": [],
                    "shape": "",
                    "shape_match": None,
                    "reason": f"read failed: {e}",
                }

        return {
            "newenv_dir": str(newenv_path),
            "reference_shape": "x".join(str(s) for s in reference_shape) if reference_shape else "",
            "files": files_meta,
        }
    except Exception as e:
        return {"reason": f"discover_newenv_cmems failed: {e}", "files": {}}


def discover_static_assets(manifest_path: str | Path | None) -> dict:
    """
    Discover static assets from a manifest JSON. Failures return empty meta + reason.
    """
    try:
        manifest = _as_path(manifest_path)
        if manifest is None:
            return {"exists": False, "reason": "manifest path not set", "entries_count": 0}
        if not manifest.exists():
            return {"exists": False, "path": str(manifest), "reason": "manifest file missing", "entries_count": 0}

        raw = json.loads(manifest.read_text(encoding="utf-8"))
        count = 0
        if isinstance(raw, list):
            count = len(raw)
        elif isinstance(raw, dict):
            if isinstance(raw.get("assets"), list):
                count = len(raw["assets"])
            elif isinstance(raw.get("files"), list):
                count = len(raw["files"])
            else:
                count = len(raw.keys())
        else:
            count = 0

        return {
            "exists": True,
            "path": str(manifest),
            "entries_count": int(count),
            "mtime": _safe_mtime(manifest),
        }
    except Exception as e:
        return {"exists": False, "reason": f"discover_static_assets failed: {e}", "entries_count": 0}
