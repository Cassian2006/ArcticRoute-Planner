from __future__ import annotations

"""
Product-of-Experts (PoE) 融合头（最小可用版）

- 输入：组件专家概率层（ice/wave/acc 等），先按分位数归一化到 [0,1]
- 融合：在 log 空间累加（带温度 T 与每专家权重 w_k），p_poe ∝ Π p_k^{w_k/T}
- 输出：Risk ∈ [0,1]（均值），保存到 risk_fused_<ym>.nc（不提供方差）

数值稳健：对 p 设最小 ε（如 1e-6），防止 log(0)。
"""

import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

from ArcticRoute.core.risk.fuse_prep import find_layer_paths  # REUSE

RISK_DIR = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "risk")
EPS = 1e-6


def _open_var(ds: "xr.Dataset", pref: Tuple[str, ...]) -> Optional[str]:
    for k in pref:
        if k in ds:
            return k
    for name in list(ds.data_vars.keys()):
        low = name.lower()
        for k in pref:
            if k.lower() in low:
                return name
    return None


def _quantile_norm_arr(arr: np.ndarray) -> np.ndarray:
    v = arr.astype(float)
    finite = v[np.isfinite(v)]
    if finite.size:
        q01 = float(np.nanpercentile(finite, 1))
        q99 = float(np.nanpercentile(finite, 99))
    else:
        q01 = 0.0
        q99 = 1.0
    if not np.isfinite(q01) or not np.isfinite(q99) or q99 <= q01:
        out = np.zeros_like(v, dtype=float)
    else:
        out = (v - q01) / (q99 - q01)
        out = np.clip(out, 0.0, 1.0)
        out[~np.isfinite(v)] = np.nan
    return out


def _norm_da(da: "xr.DataArray") -> "xr.DataArray":
    if "time" in da.dims and int(da.sizes.get("time", 0)) > 0:
        outs = []
        for t in range(int(da.sizes["time"])):
            a = da.isel(time=t)
            outs.append(xr.DataArray(_quantile_norm_arr(np.asarray(a.values)), dims=a.dims, coords=a.coords, attrs=a.attrs))
        return xr.concat(outs, dim="time")
    else:
        return xr.DataArray(_quantile_norm_arr(np.asarray(da.values)), dims=da.dims, coords=da.coords, attrs=da.attrs)


def fuse_poe(ym: str, *, temperature: float = 1.0, expert_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    if xr is None:
        raise RuntimeError("xarray required")
    T = max(float(temperature), EPS)
    w_map = {"ice": 1.0, "wave": 1.0, "acc": 1.0}
    if isinstance(expert_weights, dict):
        for k in w_map:
            try:
                if k in expert_weights:
                    w_map[k] = float(expert_weights[k])
            except Exception:
                pass

    paths = find_layer_paths(ym)
    used: Dict[str, Any] = {}
    layers: Dict[str, xr.DataArray] = {}

    for kind, pref in ("ice", ("R_ice", "risk", "ice", "Risk")), ("wave", ("R_wave", "wave", "risk")), ("acc", ("R_acc", "acc", "risk")):
        p = paths.get(kind)
        if not (isinstance(p, str) and os.path.exists(p)):
            continue
        ds = xr.open_dataset(p)
        var = _open_var(ds, pref) or (list(ds.data_vars)[0] if ds.data_vars else None)
        if var is None:
            try: ds.close()
            except Exception: pass
            continue
        da = ds[var]
        layers[kind] = da
        used[kind] = {"path": p, "var": var}

    if not layers:
        raise RuntimeError("no layers for PoE fusion")

    ref_da = next(iter(layers.values()))

    stack_list = []
    w_list = []
    for kind in ("ice", "wave", "acc"):
        if kind not in layers:
            continue
        da = layers[kind]
        if "time" in ref_da.dims and "time" not in da.dims:
            da = da.expand_dims({"time": ref_da.coords.get("time", [0])})
        if "time" in da.dims and "time" in ref_da.dims:
            t_ref = np.asarray(ref_da.coords["time"].values)
            t_da = np.asarray(da.coords["time"].values)
            inter = np.intersect1d(t_ref, t_da)
            if inter.size > 0:
                da = da.sel(time=inter)
                ref_da = ref_da.sel(time=inter)
        nda = _norm_da(da)
        stack_list.append(nda)
        w_list.append(float(w_map.get(kind, 1.0)))

    base = stack_list[0]
    arrs = [np.asarray(a.values, dtype=float) for a in stack_list]
    shape = base.shape
    arrs = [a if a.shape == shape else np.broadcast_to(a, shape) for a in arrs]
    W = np.array(w_list, dtype=float)
    W = np.where(np.isfinite(W), W, 0.0)

    # PoE：log 概率相加
    stack = np.stack(arrs, axis=-1)
    P = np.clip(stack, EPS, 1.0 - EPS)
    logp = np.log(P)
    # 权重并温度缩放
    logp_w = logp * (W.reshape((1,) * (logp.ndim - 1) + (logp.shape[-1],)) / T)
    s = np.nansum(logp_w, axis=-1)
    # 归一化：exp 后仍在 [0,1]，不强制再缩放（PoE 本身产生已归一的概率近似）
    poe = np.exp(s)
    poe = np.clip(poe, 0.0, 1.0).astype("float32")

    risk_da = base.copy(data=poe)
    risk_da.name = "Risk"
    out = xr.Dataset({"Risk": risk_da})

    run_id = time.strftime("%Y%m%dT%H%M%S")
    out.attrs.update({"ym": ym, "method": "poe", "run_id": run_id, "sources": str(used), "temperature": T, "weights": str(w_map)})

    os.makedirs(RISK_DIR, exist_ok=True)
    out_path = os.path.join(RISK_DIR, f"risk_fused_{ym}.nc")
    out.to_netcdf(out_path)
    try:
        out.close()  # type: ignore
    except Exception:
        pass
    return {"ym": ym, "out": out_path, "used": used, "vars": ["Risk"], "temperature": T, "weights": w_map}


__all__ = ["fuse_poe"]



