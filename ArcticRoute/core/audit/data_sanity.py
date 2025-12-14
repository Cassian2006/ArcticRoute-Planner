# REUSE: audit data sanity checks; import-light and fail-soft
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import xarray as xr  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore
    np = None  # type: ignore


def size_checks(path: str | Path, min_bytes_by_type: Dict[str, int], kind: Optional[str]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    try:
        p = Path(path)
        size = p.stat().st_size
    except Exception:
        return False, ["io_error:size"]
    if kind and kind in min_bytes_by_type:
        if size < int(min_bytes_by_type[kind]):
            reasons.append(f"size_below_min:{size}<{min_bytes_by_type[kind]}")
            return False, reasons
    return True, reasons


def _open_nc(path: Path) -> Optional["xr.Dataset"]:
    if xr is None:
        return None
    try:
        return xr.open_dataset(path)
    except Exception:
        return None


def risk_stats(risk_nc: str | Path) -> Optional[Dict[str, float]]:
    p = Path(risk_nc)
    ds = _open_nc(p)
    if ds is None:
        return None
    try:
        var = "risk" if "risk" in ds.variables else list(ds.data_vars)[0]
        da = ds[var]
        vals = da.values
        import numpy as np  # type: ignore
        total = np.isfinite(vals).sum()
        nonzero = np.isfinite(vals) & (vals > 0)
        nz = nonzero.sum()
        nz_pct = float(nz) / float(total) if total else 0.0
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals))
        return {"nonzero_pct": nz_pct, "mean": mean, "std": std}
    except Exception:
        return None


def prior_stats(prior_nc: str | Path) -> Optional[Dict[str, float]]:
    p = Path(prior_nc)
    ds = _open_nc(p)
    if ds is None:
        return None
    try:
        var = None
        for candidate in ("prior_penalty", "penalty", "prior"):
            if candidate in ds.variables:
                var = candidate
                break
        if var is None:
            var = list(ds.data_vars)[0]
        import numpy as np  # type: ignore
        vals = ds[var].values
        return {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    except Exception:
        return None


def interact_corr(ais_density_nc: str | Path, interact_nc: str | Path) -> Optional[float]:
    if xr is None:
        return None
    try:
        ds_d = xr.open_dataset(str(ais_density_nc))
        var_d = list(ds_d.data_vars)[0]
        ds_i = xr.open_dataset(str(interact_nc))
        var_i = list(ds_i.data_vars)[0]
        da_d = ds_d[var_d]
        da_i = ds_i[var_i]
        # align grids
        try:
            da_i2 = da_i.interp_like(da_d, method="nearest")
        except Exception:
            da_i2 = da_i
        import numpy as np  # type: ignore
        a = da_d.values.reshape(-1)
        b = da_i2.values.reshape(-1)
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 100:
            return None
        a = a[mask]
        b = b[mask]
        # pearson
        am = a - a.mean()
        bm = b - b.mean()
        denom = (np.sqrt((am**2).sum()) * np.sqrt((bm**2).sum()))
        if denom == 0:
            return 0.0
        return float((am * bm).sum() / denom)
    except Exception:
        return None








