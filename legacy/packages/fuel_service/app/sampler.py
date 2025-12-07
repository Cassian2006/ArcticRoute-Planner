from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np

# 占位：若 xarray/rioxarray 不可用，提供平滑回退
try:
    import xarray as xr
    import rioxarray as rxr
except Exception:
    xr = None
    rxr = None


class RouteSampler:
    def __init__(self, repo_root: Path) -> None:
        self._repo = repo_root
        self._cache: Dict[str, Optional[xr.DataArray]] = {}

    def _load(self, ym: str, kind: str) -> Optional[xr.DataArray]:
        key = f"{ym}_{kind}"
        if key in self._cache:
            return self._cache[key]
        p: Optional[Path] = None
        var: Optional[str] = None
        if kind == "ice":
            p = self._repo / "ArcticRoute" / "data_processed" / "risk" / f"R_ice_eff_{ym}.nc"
            var = "risk"
        elif kind == "wave":
            p = self._repo / "ArcticRoute" / "data_processed" / "risk" / f"R_wave_{ym}.nc"
            var = "risk"
        elif kind == "risk":
            p = self._repo / "ArcticRoute" / "data_processed" / "risk" / f"risk_fused_{ym}.nc"
            var = "risk"
        if p is None or not p.exists() or xr is None:
            self._cache[key] = None
            return None
        try:
            ds = xr.open_dataset(p)
            with ds:
                da = ds[var] if (var and var in ds) else ds[list(ds.data_vars)[0]]
                # 强制 WGS84
                if da.rio.crs is None:
                    da = da.rio.write_crs("EPSG:4326")
                self._cache[key] = da.load()
                return self._cache[key]
        except Exception:
            self._cache[key] = None
            return None

    def sample(self, coords: List[Tuple[float, float]], ym: str, kind: str) -> Optional[float]:
        da = self._load(ym, kind)
        if da is None:
            return None
        try:
            lons = xr.DataArray([c[0] for c in coords], dims="points")
            lats = xr.DataArray([c[1] for c in coords], dims="points")
            vals = da.interp(lon=lons, lat=lats, method="nearest")
            # 简单平均
            avg = float(np.nanmean(vals.values))
            return avg if np.isfinite(avg) else None
        except Exception:
            return None



