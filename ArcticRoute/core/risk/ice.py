"""冰风险层（R_ice）：基于 P1 ice_cost 复用并可选叠加厚度/边缘梯度

- build_risk_ice(ym, t0, t1, gamma, use_thickness=True, use_edge=True, dry_run=True)
  读取：sic_fcst_<ym>.nc（必需）与 ice_cost_<ym>.nc（若存在直接映射为 R_ice；否则用 1-sic 兜底）
  可选：叠加边缘项 edge_strength = |∇SIC|（Sobel/梯度近似），厚度项（当前占位，若 env_clean.nc 含 thickness 则归一后参与）
  输出：risk_ice_<ym>.nc（变量 R_ice ∈[0,1]），并注册 artifact。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

try:
    from scipy.ndimage import sobel  # type: ignore
except Exception:  # pragma: no cover
    sobel = None  # type: ignore

from ArcticRoute.cache.index_util import register_artifact


def _pick_sic(ym: str) -> xr.DataArray:
    p = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "ice_forecast", "merged", f"sic_fcst_{ym}.nc")
    if not os.path.exists(p):
        raise FileNotFoundError(f"缺少 sic_fcst 文件: {p}")
    ds = xr.open_dataset(p)
    try:
        v = None
        for name in ("sic_pred","sic","siconc","ci","ice_conc","sea_ice_concentration"):
            if name in ds:
                v = ds[name]
                break
        if v is None:
            raise ValueError("未找到 SIC 变量")
        v = v.clip(0.0, 1.0)
        if "time" in v.dims:
            v0 = v.isel(time=0)
        else:
            v0 = v
        return v0
    finally:
        try:
            ds.close()
        except Exception:
            pass


def _pick_cost(ym: str) -> Optional[xr.DataArray]:
    p = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "ice_forecast", "merged", f"ice_cost_{ym}.nc")
    if not os.path.exists(p):
        return None
    ds = xr.open_dataset(p)
    try:
        if "ice_cost" in ds:
            v = ds["ice_cost"]
            if "time" in v.dims:
                v0 = v.isel(time=0)
            else:
                v0 = v
            return v0.clip(0.0, 1.0)
        return None
    finally:
        try:
            ds.close()
        except Exception:
            pass


def _edge_strength(arr: np.ndarray) -> np.ndarray:
    if sobel is None:
        gy, gx = np.gradient(arr)
    else:
        gy = sobel(arr, axis=0, mode="nearest")
        gx = sobel(arr, axis=1, mode="nearest")
    g = np.hypot(gx, gy)
    # 分位归一
    lo, hi = np.nanpercentile(g, [1, 99])
    if not np.isfinite(hi - lo) or (hi - lo) <= 1e-6:
        return np.zeros_like(g, dtype=np.float32)
    g = (g - lo) / (hi - lo)
    return np.clip(g, 0.0, 1.0).astype(np.float32)


def build_risk_ice(ym: str, t0: float, t1: float, gamma: float, use_thickness: bool = True, use_edge: bool = True, dry_run: bool = True) -> Dict[str, Any]:
    if xr is None:
        raise RuntimeError("xarray required for risk_ice")

    sic0 = _pick_sic(ym)
    cost0 = _pick_cost(ym)
    base = None
    if cost0 is not None:
        base = cost0
    else:
        base = (1.0 - sic0).clip(0.0, 1.0)
    arr = np.asarray(base.values, dtype=np.float32)

    # 可选边缘项：与 base 做加权融合（轻量实现）
    if use_edge:
        edge = _edge_strength(np.asarray(sic0.values, dtype=float))
        arr = np.clip(0.8 * arr + 0.2 * edge, 0.0, 1.0)
        # 若存在 cv_cache/edge_dist_YYYYMM.nc 则进一步融合（归一后弱权重叠加）
        try:
            p_edge = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "cv_cache", f"edge_dist_{ym}.nc")
            if os.path.exists(p_edge):
                ds_e = xr.open_dataset(p_edge)
                try:
                    vname = list(ds_e.data_vars)[0]
                    ed = ds_e[vname]
                    try:
                        ed = ed.interp_like(sic0, method="nearest")
                    except Exception:
                        pass
                    ev = np.asarray(ed.values, dtype=float)
                    # 距离越小风险越大：做反向并分位归一
                    ev = 1.0 - (ev - np.nanmin(ev)) / (np.nanmax(ev) - np.nanmin(ev) + 1e-6)
                    lo, hi = np.nanpercentile(ev, [1, 99])
                    if hi > lo:
                        evn = np.clip((ev - lo) / (hi - lo), 0.0, 1.0)
                        arr = np.clip(0.85 * arr + 0.15 * evn, 0.0, 1.0)
                finally:
                    try:
                        ds_e.close()
                    except Exception:
                        pass
        except Exception:
            pass

    # 可选厚度项：若 env_clean.nc 含 thickness 则轻量融合（占位）
    if use_thickness:
        try:
            envp = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "env_clean.nc")
            if os.path.exists(envp):
                ds_env = xr.open_dataset(envp)
                try:
                    th = None
                    for name in ("thickness","ice_thickness","sit"):
                        if name in ds_env:
                            th = ds_env[name]
                            break
                    if th is not None:
                        th2 = th
                        try:
                            th2 = th2.interp_like(sic0, method="nearest")
                        except Exception:
                            pass
                        ta = np.asarray(th2.values, dtype=float)
                        # 分位归一后与 base 融合（弱权重）
                        lo, hi = np.nanpercentile(ta, [1, 99])
                        if hi > lo:
                            tn = np.clip((ta - lo) / (hi - lo), 0.0, 1.0)
                            arr = np.clip(0.9 * arr + 0.1 * tn, 0.0, 1.0)
                finally:
                    try:
                        ds_env.close()
                    except Exception:
                        pass
        except Exception:
            pass

    da = xr.DataArray(arr, dims=sic0.dims, coords=sic0.coords, name="R_ice")
    ds_out = xr.Dataset({"R_ice": da})
    ds_out["R_ice"].attrs.update({"long_name": "Ice risk (derived)", "units": "1"})
    ds_out = ds_out.assign_attrs({"layer": "risk_ice", "ym": ym, "use_edge": int(use_edge), "use_thickness": int(use_thickness)})

    out_dir = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "risk")
    os.makedirs(out_dir, exist_ok=True)
    out_nc = os.path.join(out_dir, f"risk_ice_{ym}.nc")

    if not dry_run:
        comp = {"zlib": True, "complevel": 4}
        ds_out.to_netcdf(out_nc, encoding={"R_ice": comp})
        try:
            run_id = os.environ.get("RUN_ID", "") or __import__("time").strftime("%Y%m%dT%H%M%S")
        except Exception:
            run_id = ""
        try:
            register_artifact(run_id=run_id, kind="risk_ice", path=out_nc, attrs={"ym": ym})
        except Exception:
            pass

    return {"out": out_nc, "dry_run": bool(dry_run), "shape": {k:int(ds_out.sizes[k]) for k in ds_out.sizes}}


__all__ = ["build_risk_ice"]

