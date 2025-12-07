from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

# REUSE quantile normalization from risk fusion
try:
    from ArcticRoute.core.risk.fusion import _quantile_norm as _qnorm  # type: ignore
except Exception:  # pragma: no cover
    _qnorm = None  # type: ignore

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RISK_DIR = os.path.join(REPO_ROOT, "ArcticRoute", "data_processed", "risk")
ECO_DIR = os.path.join(REPO_ROOT, "ArcticRoute", "data_processed", "eco")
CFG_ECO = os.path.join(REPO_ROOT, "ArcticRoute", "config", "eco.yaml")

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


def _load_cfg() -> Dict[str, Any]:
    if yaml is None or not os.path.exists(CFG_ECO):
        return {}
    try:
        with open(CFG_ECO, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _open_da(path: str, var_hint: Optional[str] = None) -> Optional["xr.DataArray"]:
    if xr is None or not os.path.exists(path):
        return None
    try:
        ds = xr.open_dataset(path)
        if var_hint and var_hint in ds:
            da = ds[var_hint]
        else:
            # try common variable names
            for k in ("risk", "Risk", "R_ice", "R_wave"):
                if k in ds:
                    da = ds[k]
                    break
            else:
                da = ds[list(ds.data_vars)[0]] if ds.data_vars else None
        if da is None:
            ds.close()
            return None
        # Ensure time dimension exists for consistency
        if "time" not in da.dims:
            da = da.expand_dims({"time": [0]})
        return da
    except Exception:
        return None


def fuel_per_nm_map(
    ym: str,
    vessel_class: str,
    alpha_ice: Optional[float] = None,
    alpha_wave: Optional[float] = None,
    *,
    ice_da: Optional["xr.DataArray"] = None,
    wave_da: Optional["xr.DataArray"] = None,
) -> Tuple["xr.DataArray", Dict[str, Any]]:
    """
    生成每海里燃油消耗栅格（t per nm），返回 (eco_cost_nm_t, meta)。
    phi_ice = 1 + alpha_ice * R_ice_eff; phi_wave = 1 + alpha_wave * R_wave (可选)
    基线 fuel_per_nm_base 与 ice_class_factor 来自 config/eco.yaml。
    若提供 ice_da/wave_da 则优先使用（用于单元测试）。
    """
    if xr is None:
        raise RuntimeError("xarray required")
    cfg = _load_cfg()
    eco_cfg = cfg.get("eco", {}) if isinstance(cfg, dict) else {}
    vc = (eco_cfg.get("vessel_classes", {}) or {}).get(vessel_class, {})
    base = float(vc.get("fuel_per_nm_base", 0.012) or 0.012)
    phi_class = float(vc.get("ice_class_factor", 1.0) or 1.0)
    a_ice = float(alpha_ice if alpha_ice is not None else eco_cfg.get("alpha_ice", 0.8) or 0.8)
    a_wave = float(alpha_wave if alpha_wave is not None else eco_cfg.get("alpha_wave", 0.0) or 0.0)

    # Load layers
    ice = ice_da
    if ice is None:
        cand = [
            os.path.join(RISK_DIR, f"R_ice_eff_{ym}.nc"),
            os.path.join(RISK_DIR, f"risk_ice_{ym}.nc"),
        ]
        for p in cand:
            ice = _open_da(p, None)
            if ice is not None:
                break
    wave = wave_da
    if wave is None:
        p_wave = os.path.join(RISK_DIR, f"R_wave_{ym}.nc")
        wave = _open_da(p_wave, None)

    if ice is None:
        raise FileNotFoundError("R_ice_eff missing; required for eco map")

    # Broadcast wave to ice dims
    if wave is not None:
        try:
            wave = wave.broadcast_like(ice)
        except Exception:
            # coarse fallback: expand time only
            if "time" not in wave.dims and "time" in ice.dims:
                wave = wave.expand_dims({"time": ice.coords["time"]})

    # Compute multipliers
    ice_arr = ice.astype("float32").clip(0.0, 1.0)
    if wave is not None:
        wave_arr = wave.astype("float32").clip(0.0, 1.0)
        phi_wave = 1.0 + float(a_wave) * wave_arr
    else:
        phi_wave = 1.0
    phi_ice = 1.0 + float(a_ice) * ice_arr

    fuel_nm = float(base) * float(phi_class) * phi_ice * (phi_wave if isinstance(phi_wave, (int, float)) else phi_wave)
    # name and attrs
    da = ice.copy(data=np.asarray(fuel_nm, dtype="float32"))
    da.name = "eco_cost_nm_t"
    da.attrs.update({
        "long_name": "Fuel per nautical mile [t/nm]",
        "vessel_class": vessel_class,
        "fuel_per_nm_base": float(base),
        "ice_class_factor": float(phi_class),
        "alpha_ice": float(a_ice),
        "alpha_wave": float(a_wave),
    })
    meta = {
        "ym": ym,
        "vessel_class": vessel_class,
        "fuel_per_nm_base": float(base),
        "ice_class_factor": float(phi_class),
        "alpha_ice": float(a_ice),
        "alpha_wave": float(a_wave),
        "has_wave": bool(wave is not None),
    }
    return da, meta


def eco_cost_norm(eco_cost_nm_t: "xr.DataArray") -> "xr.DataArray":
    """将每海里燃油消耗映射量化归一到 [0,1]。
    REUSE: 使用 risk.fusion._quantile_norm。
    """
    if _qnorm is None:
        # simple percentile normalization as fallback
        arr = eco_cost_nm_t
        if "time" in arr.dims and int(arr.sizes.get("time", 0)) > 0:
            outs = []
            for t in range(int(arr.sizes["time"])):
                a = arr.isel(time=t)
                v = np.asarray(a.values, dtype=float)
                finite = v[np.isfinite(v)]
                if finite.size == 0:
                    norm = np.zeros_like(v)
                else:
                    p1 = float(np.nanpercentile(finite, 1))
                    p99 = float(np.nanpercentile(finite, 99))
                    if not np.isfinite(p99) or p99 <= p1:
                        norm = np.zeros_like(v)
                    else:
                        norm = np.clip((v - p1) / (p99 - p1), 0.0, 1.0)
                outs.append(xr.DataArray(norm, dims=a.dims, coords=a.coords, attrs=a.attrs))
            nda = xr.concat(outs, dim="time")
        else:
            v = np.asarray(arr.values, dtype=float)
            finite = v[np.isfinite(v)]
            if finite.size == 0:
                norm = np.zeros_like(v)
            else:
                p1 = float(np.nanpercentile(finite, 1))
                p99 = float(np.nanpercentile(finite, 99))
                if not np.isfinite(p99) or p99 <= p1:
                    norm = np.zeros_like(v)
                else:
                    norm = np.clip((v - p1) / (p99 - p1), 0.0, 1.0)
            nda = xr.DataArray(norm, dims=arr.dims, coords=arr.coords, attrs=arr.attrs)
    else:
        nda = _qnorm(eco_cost_nm_t)  # type: ignore
    nda = nda.rename("eco_cost_norm")
    nda.attrs.update({"long_name": "Normalized eco penalty [0,1]"})
    return nda


__all__ = ["fuel_per_nm_map", "eco_cost_norm"]

