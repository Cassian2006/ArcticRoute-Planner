"""
AIS density candidate selection and alignment helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

from .grid import Grid2D


@dataclass
class AISCandidate:
    path: Path
    size_mb: float
    mtime: datetime
    shape: Optional[Tuple[int, int]]
    grid_signature: Optional[str]
    signature_matched: bool = False


def _read_candidate_info(path: Path) -> AISCandidate:
    size_mb = path.stat().st_size / (1024 * 1024)
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    shape: Optional[Tuple[int, int]] = None
    grid_signature: Optional[str] = None

    try:
        import xarray as xr

        with xr.open_dataset(path, decode_times=False) as ds:
            if ds.data_vars:
                first_var = list(ds.data_vars.values())[0]
                if hasattr(first_var, "shape"):
                    shape = tuple(int(s) for s in first_var.shape[:2])
            grid_signature = ds.attrs.get("grid_signature")
    except Exception:
        pass

    return AISCandidate(
        path=path,
        size_mb=size_mb,
        mtime=mtime,
        shape=shape,
        grid_signature=grid_signature,
    )


def scan_candidates(search_dirs: Iterable[Path]) -> List[AISCandidate]:
    candidates: List[AISCandidate] = []
    for base in search_dirs:
        if not base.exists():
            continue
        for path in base.rglob("*.nc"):
            if not path.is_file():
                continue
            name_lower = path.name.lower()
            if "density" not in name_lower and "ais" not in name_lower:
                continue
            if "train" in name_lower or "training" in name_lower:
                continue
            candidates.append(_read_candidate_info(path))
    return candidates


def select_best_candidate(
    candidates: List[AISCandidate],
    grid_signature: Optional[str],
) -> Tuple[Optional[AISCandidate], dict]:
    if not candidates:
        return None, {"signature_matched": False, "chosen_reason": "no candidates"}

    for cand in candidates:
        cand.signature_matched = bool(grid_signature and cand.grid_signature == grid_signature)

    matched = [c for c in candidates if c.signature_matched]
    if matched:
        best = max(matched, key=lambda c: (c.shape[0] * c.shape[1] if c.shape else 0, c.mtime))
        return best, {"signature_matched": True, "chosen_reason": "grid_signature matched"}

    def _fallback_score(cand: AISCandidate) -> Tuple[int, datetime]:
        area = cand.shape[0] * cand.shape[1] if cand.shape else 0
        return area, cand.mtime

    best = max(candidates, key=_fallback_score)
    return best, {
        "signature_matched": False,
        "chosen_reason": "no signature match; selected largest/newest candidate",
    }


def load_and_align_density(
    grid: Grid2D,
    candidate: Optional[AISCandidate],
) -> Tuple[np.ndarray, dict]:
    meta = {
        "path": None,
        "success": False,
        "signature_matched": False,
        "reason": "",
        "resampled": False,
    }
    if candidate is None:
        meta["reason"] = "no candidate selected"
        return np.zeros(grid.shape(), dtype=float), meta

    meta["path"] = str(candidate.path)
    meta["signature_matched"] = bool(candidate.signature_matched)

    try:
        import xarray as xr

        with xr.open_dataset(candidate.path, decode_times=False) as ds:
            da = None
            for key in ["ais_density", "density", "ais"]:
                if key in ds:
                    da = ds[key]
                    break
            if da is None and ds.data_vars:
                da = list(ds.data_vars.values())[0]
            if da is None:
                meta["reason"] = "no data variable found"
                return np.zeros(grid.shape(), dtype=float), meta

            if da.shape == grid.shape():
                meta["success"] = True
                meta["reason"] = "shape matched"
                return np.asarray(da.values, dtype=float), meta

            lat_name = "lat" if "lat" in da.coords else ("latitude" if "latitude" in da.coords else None)
            lon_name = "lon" if "lon" in da.coords else ("longitude" if "longitude" in da.coords else None)
            if lat_name and lon_name:
                target = da.interp(
                    **{
                        lat_name: (("y", "x"), grid.lat2d),
                        lon_name: (("y", "x"), grid.lon2d),
                    },
                    method="nearest",
                )
                meta["success"] = True
                meta["resampled"] = True
                meta["reason"] = "resampled to grid"
                return np.asarray(target.values, dtype=float), meta

            meta["reason"] = "shape mismatch and missing coords; cannot resample"
    except Exception as exc:
        meta["reason"] = f"load failed: {exc}"

    return np.zeros(grid.shape(), dtype=float), meta
