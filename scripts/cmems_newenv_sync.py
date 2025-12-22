#!/usr/bin/env python3
"""
Sync latest CMEMS cache files into newenv fixed filenames and write an index.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


LAYER_CONFIG = {
    "sic": {
        "patterns": ["*sic*.nc", "*siconc*.nc"],
        "target": "ice_copernicus_sic.nc",
    },
    "swh": {
        "patterns": ["*swh*.nc", "*wave*.nc", "*wav*.nc"],
        "target": "wave_swh.nc",
    },
    "sit": {
        "patterns": ["*thickness*.nc", "*sit*.nc"],
        "target": "ice_thickness.nc",
    },
    "drift": {
        "patterns": ["*drift*.nc", "*uice*.nc", "*vice*.nc", "*ice_velocity*.nc"],
        "target": "ice_drift.nc",
    },
}


def _find_latest_match(cache_dir: Path, patterns: List[str]) -> Optional[Path]:
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(cache_dir.rglob(pattern))
    matches = [p for p in matches if p.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _describe_nc(path: Path) -> Tuple[List[str], Optional[Tuple[int, int]], List[str]]:
    notes: List[str] = []
    vars_list: List[str] = []
    shape: Optional[Tuple[int, int]] = None
    try:
        import xarray as xr
    except Exception:
        notes.append("xarray not available; vars/shape skipped")
        return vars_list, shape, notes

    try:
        with xr.open_dataset(path) as ds:
            vars_list = list(ds.data_vars.keys())
            if ds.data_vars:
                first_var = list(ds.data_vars.values())[0]
                if hasattr(first_var, "shape"):
                    shape = tuple(int(s) for s in first_var.shape[:2])
    except Exception as exc:
        notes.append(f"failed to inspect dataset: {exc}")

    return vars_list, shape, notes


def sync_cmems_newenv(cache_dir: Path, newenv_dir: Path) -> Dict:
    newenv_dir.mkdir(parents=True, exist_ok=True)

    index: Dict[str, Dict] = {
        "generated_at": datetime.now().isoformat(),
        "cache_dir": str(cache_dir),
        "newenv_dir": str(newenv_dir),
        "layers": {},
    }

    for layer, cfg in LAYER_CONFIG.items():
        target_path = newenv_dir / cfg["target"]
        entry: Dict[str, object] = {
            "found": False,
            "source_path": None,
            "target_path": str(target_path),
            "mtime": None,
            "vars": [],
            "shape": None,
            "notes": "",
            "reason": "",
        }

        if not cache_dir.exists():
            entry["reason"] = f"cache dir missing: {cache_dir}"
            index["layers"][layer] = entry
            continue

        latest = _find_latest_match(cache_dir, cfg["patterns"])
        if latest is None:
            entry["reason"] = f"no matches in cache for patterns: {cfg['patterns']}"
            index["layers"][layer] = entry
            continue

        try:
            shutil.copy2(latest, target_path)
            entry["found"] = True
            entry["source_path"] = str(latest)
            entry["mtime"] = datetime.fromtimestamp(latest.stat().st_mtime).isoformat()
            vars_list, shape, notes = _describe_nc(target_path)
            entry["vars"] = vars_list
            entry["shape"] = list(shape) if shape else None
            entry["notes"] = "; ".join(notes) if notes else "synced from cache"
        except Exception as exc:
            entry["reason"] = f"copy failed: {exc}"

        index["layers"][layer] = entry

    index_path = newenv_dir / "cmems_newenv_index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=True), encoding="utf-8")

    return index


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync CMEMS cache into newenv and write index.")
    parser.add_argument("--cache-dir", default="data/cmems_cache")
    parser.add_argument("--newenv-dir", default="data_processed/newenv")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    newenv_dir = Path(args.newenv_dir)
    sync_cmems_newenv(cache_dir, newenv_dir)
    print(f"[CMEMS] synced cache {cache_dir} -> {newenv_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
