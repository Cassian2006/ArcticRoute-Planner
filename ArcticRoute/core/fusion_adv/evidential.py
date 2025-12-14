from __future__ import annotations

"""
Evidential 融合头（最小可用版）

- 输入：与 Phase F 融合相同的数据源（ice/wave/acc + 可选 interact/prior）
- 输出：Risk ∈ [0,1]（均值）、RiskVar ≥ 0（方差近似），保存到 risk_fused_<ym>.nc
- 训练相关留空；此实现为弱监督/无监督下的即插即用头：
  1) 各层按分位数 1%-99% 归一化到 [0,1]
  2) 先得到加权均值（与 Phase F 一致，REUSE 思路）作为 m
  3) 使用启发式“证据强度”e：由各层一致性与稀疏度得到；将 (m,e) 转为 Beta(a,b)，导出方差

备注：若未来有真实 evidential 训练，保留相同接口，可直接替换内部实现。
"""

import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

from ArcticRoute.core.risk.fuse_prep import find_layer_paths  # REUSE 层发现

RISK_DIR = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "risk")


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


def _beta_from_mean_strength(m: np.ndarray, e: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    # Dirichlet evidence: S = e0+e1+K; Beta(a=m*S, b=(1-m)*S)
    S = np.maximum(e, eps)
    a = np.clip(m, eps, 1.0 - eps) * S
    b = (1.0 - np.clip(m, eps, 1.0 - eps)) * S
    return a, b


def _heuristic_evidence(stack: np.ndarray) -> np.ndarray:
    """根据多层一致性（方差越小 -> 证据越强）和总体强度估计 e。
    stack: shape (..., C) in [0,1]
    返回与前面维度一致的 e（标量强度）。
    """
    # 层间方差（小→一致→强）
    var = np.nanvar(stack, axis=-1)
    # 稀疏度：越接近 0 或 1，越“确定”
    mean = np.nanmean(stack, axis=-1)
    sharp = np.maximum(0.0, np.abs(mean - 0.5) * 2.0)  # 0..1
    # 组合：映射到 [S_min, S_max]
    cons = 1.0 / (1.0 + var * 10.0)
    score = 0.5 * cons + 0.5 * sharp  # 0..1
    S = 2.0 + 18.0 * score  # 最小2，最大20（可调）
    return S


def fuse_evidential(ym: str, *, weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    if xr is None:
        raise RuntimeError("xarray required")
    paths = find_layer_paths(ym)
    used: Dict[str, Any] = {}
    layers: Dict[str, xr.DataArray] = {}

    # 读取候选层（ice/wave/acc）
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
        raise RuntimeError("no layers for evidential fusion")

    # 对齐参考坐标
    ref_da = next(iter(layers.values()))
    # 归一化并堆叠
    stack_list = []
    for kind in ("ice", "wave", "acc"):
        if kind not in layers:
            continue
        da = layers[kind]
        if "time" in ref_da.dims and "time" not in da.dims:
            da = da.expand_dims({"time": ref_da.coords.get("time", [0])})
        if "time" in da.dims and "time" in ref_da.dims:
            # 对齐时间交集
            t_ref = np.asarray(ref_da.coords["time"].values)
            t_da = np.asarray(da.coords["time"].values)
            inter = np.intersect1d(t_ref, t_da)
            if inter.size > 0:
                da = da.sel(time=inter)
                ref_da = ref_da.sel(time=inter)
        nda = _norm_da(da)
        stack_list.append(nda)

    if not stack_list:
        raise RuntimeError("all layers invalid for evidential fusion")

    # shape 同步
    base = stack_list[0]
    arrs = [np.asarray(a.values, dtype=float) for a in stack_list]
    # 简化：广播到相同形状
    shape = base.shape
    arrs = [a if a.shape == shape else np.broadcast_to(a, shape) for a in arrs]
    stack = np.stack(arrs, axis=-1)

    # 均值（Phase F REUSE）
    m = np.nanmean(stack, axis=-1)
    m = np.clip(m, 0.0, 1.0).astype("float32")

    # 证据强度与方差（Beta）
    S = _heuristic_evidence(stack)
    a, b = _beta_from_mean_strength(m, S)
    var = (a * b) / (((a + b) ** 2) * (a + b + 1.0))
    var = np.maximum(var, 0.0).astype("float32")

    # 写盘
    risk_da = base.copy(data=m)
    risk_da.name = "Risk"
    var_da = base.copy(data=var)
    var_da.name = "RiskVar"
    out = xr.Dataset({"Risk": risk_da, "RiskVar": var_da})
    run_id = time.strftime("%Y%m%dT%H%M%S")
    out.attrs.update({"ym": ym, "method": "evidential", "run_id": run_id, "sources": str(used)})

    os.makedirs(RISK_DIR, exist_ok=True)
    out_path = os.path.join(RISK_DIR, f"risk_fused_{ym}.nc")
    out.to_netcdf(out_path)
    try:
        out.close()  # type: ignore
    except Exception:
        pass
    return {"ym": ym, "out": out_path, "used": used, "vars": ["Risk", "RiskVar"]}


__all__ = ["fuse_evidential"]



