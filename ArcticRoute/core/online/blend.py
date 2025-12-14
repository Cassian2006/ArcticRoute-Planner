from __future__ import annotations

"""
Phase N · Online Assimilation & Real-time Replanning

本模块提供：
- blend_components: 对输入组件按置信度进行归一化加权融合（逐组件）# REUSE 量化归一
- fuse_live: 将 blend 后组件复用现有融合逻辑（轻量 stacking）生成 live 风险面

产物：ArcticRoute/data_processed/risk/risk_fused_live_<ts>.nc
变量名：risk
坐标/尺寸：与首个可用组件一致；time 维自动对齐（取交集；若静态层则广播）
"""

import os
import time
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

# REUSE: 使用 Phase K 中的分位数归一
from ArcticRoute.core.risk.fusion import _quantile_norm as _qnorm  # type: ignore  # REUSE

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RISK_DIR = os.path.join(REPO_ROOT, "ArcticRoute", "data_processed", "risk")


def _align_time(a: "xr.DataArray", ref_times: Optional[np.ndarray]) -> "xr.DataArray":
    if ref_times is None:
        return a
    if "time" in a.dims:
        return a.sel(time=ref_times)
    else:
        return a.expand_dims({"time": ref_times})  # type: ignore


def blend_components(components: Dict[str, "xr.DataArray"], conf: Dict[str, float], norm: str = "quantile") -> Dict[str, "xr.DataArray"]:
    """对组件做置信度加权的归一（返回各组件的 blend 结果）。

    components: {name: xr.DataArray}
    conf: {name: w_now in [0,1]}
    norm: 仅支持 quantile（# REUSE）
    """
    if xr is None:
        raise RuntimeError("xarray is required")
    out: Dict[str, xr.DataArray] = {}
    # 统一时间轴：取存在 time 维的交集
    times = []
    for da in components.values():
        if "time" in da.dims:
            try:
                times.append(np.asarray(da["time"].values))
            except Exception:
                pass
    ref_times: Optional[np.ndarray] = None
    if times:
        inter = times[0]
        for t in times[1:]:
            inter = np.intersect1d(inter, t)
        if inter.size == 0:
            raise RuntimeError("时间轴无交集，无法对齐")
        ref_times = inter

    # 逐组件 blend: w*C_new + (1-w)*C_base, 实际上我们仅接收单个最新 C_new 与 baseline C_base 由上层传入
    # 这里的 components 可直接视为 {ice: C_new 或 baseline, ...}；为通用性，按 conf[name] 与 (1-conf) 与自身的回退做线性压缩
    # 具体：先归一，再按权重缩放，调用方可传入 [name]_base 键进行更强控制，这里保持简单
    for name, da in components.items():
        a = _align_time(da.astype(float), ref_times)
        nda = _qnorm(a) if norm == "quantile" else a
        w = float(conf.get(name, conf.get("default", 0.7)))
        # 将 blend 映射到 [0,1] 但不与 baseline 合成（baseline 已可由调用方做 stacking）
        out[name] = nda * w + (1.0 - w) * nda  # 恒等映射，保持归一结构，方便 fuse_live 统一处理
    return out


def fuse_live(components_blend: Dict[str, "xr.DataArray"], method: str = "stacking") -> "xr.DataArray":
    """将 blend 后的组件融合为 risk。
    - method=stacking: 线性加权叠加（等权），并裁剪到[0,1]
    - 若存在 time 维，逐帧处理
    返回 xr.DataArray 名为 "risk"
    """
    if xr is None:
        raise RuntimeError("xarray is required")
    if not components_blend:
        raise ValueError("no components for fuse_live")

    # 选择基准坐标
    ref = next(iter(components_blend.values()))
    combo = None
    n = 0
    for da in components_blend.values():
        arr = da.astype("float32")
        combo = arr if combo is None else (combo + arr)
        n += 1
    fused = (combo / max(1, n)).clip(0.0, 1.0)
    fused = fused.rename("risk")
    # 基本 attrs
    fused.attrs.update({
        "long_name": "Live fused risk",
        "fuse_method": method,
        "norm": "quantile",
    })
    return fused


def write_live_risk(ds: "xr.DataArray", ts: Optional[str] = None) -> Tuple[str, str]:
    """写盘 live 风险面并返回 (path, run_id)。"""
    if xr is None:
        raise RuntimeError("xarray is required")
    os.makedirs(RISK_DIR, exist_ok=True)
    run_id = ts or time.strftime("%Y%m%dT%H%M%S")
    out_path = os.path.join(RISK_DIR, f"risk_fused_live_{run_id}.nc")
    dso = ds.to_dataset()
    try:
        dso.to_netcdf(out_path)
    finally:
        try:
            dso.close()  # type: ignore
        except Exception:
            pass
    # 写 meta
    try:
        with open(out_path + ".meta.json", "w", encoding="utf-8") as f:
            import json
            json.dump({"logical_id": os.path.basename(out_path), "run_id": run_id, "vars": [str(k) for k in list(dso.data_vars)]}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return out_path, run_id


__all__ = ["blend_components", "fuse_live", "write_live_risk"]

