from __future__ import annotations
"""
UNet-Former 的 MoE 推理包装：
- 先调用基模型 infer_month 得到全局概率图
- 如存在 bucket_{ym}.nc 与 reports/d_stage/phaseL/calibration_<bucket>_<ym>.json，
  按桶对对应区域做后校准（缺失回退全局或原样）
- 预留适配器挂点（experts）：若检测到 outputs/phaseL/moe/*/expert_<bucket>.ckpt，可在未来接入

REUSE: 只做后处理，不改变原始模型代码；保持输出路径与变量名不变。
"""
from typing import Any, Dict, List, Optional
import os
import glob
import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

from .unetformer import infer_month as _infer_base  # REUSE
from .calibrate import apply_calibration  # REUSE

ROOT = os.path.join(os.getcwd(), "ArcticRoute")
RISK_DIR = os.path.join(ROOT, "data_processed", "risk")
PHASEL_DIR = os.path.join(ROOT, "reports", "d_stage", "phaseL")


def _find_bucket_mapping(dsb: "xr.Dataset") -> Dict[int, str]:
    # 读取 attrs.mapping 或旁路映射（未知则以整数自身字符串化）
    var = list(dsb.data_vars)[0]
    mapping = dsb[var].attrs.get("mapping") if hasattr(dsb[var], "attrs") else None
    out: Dict[int, str] = {}
    if isinstance(mapping, dict):
        for k, v in mapping.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                try:
                    out[int(v)] = str(k)
                except Exception:
                    continue
    # 兜底：范围内转字符串
    if not out:
        arr = np.asarray(dsb[var].values)
        for i in np.unique(arr):
            try:
                out[int(i)] = str(int(i))
            except Exception:
                continue
    return out


def _calib_path_for_bucket(bucket: str, ym: str) -> Optional[str]:
    cand = os.path.join(PHASEL_DIR, f"calibration_{bucket}_{ym}.json")
    return cand if os.path.exists(cand) else None


def infer_month_moe(ym: str, inputs: List[str], ckpt: str, calibrated: bool = False, calib_path: Optional[str] = None) -> Dict[str, Any]:
    if xr is None:
        raise RuntimeError("xarray required")
    # 1) 先获得全局推理
    payload = _infer_base(ym, inputs, ckpt=ckpt, calibrated=calibrated, calib_path=calib_path)
    # 2) 尝试按桶后处理
    risk_path = os.path.join(RISK_DIR, f"risk_fused_{ym}.nc")
    bucket_path = os.path.join(RISK_DIR, f"bucket_{ym}.nc")
    if (not os.path.exists(risk_path)) or (not os.path.exists(bucket_path)):
        payload.update({"moe": True, "by_bucket": True, "fallback": "missing_bucket"})
        return payload
    ds_r = xr.open_dataset(risk_path)
    var = "risk" if "risk" in ds_r.variables else list(ds_r.data_vars)[0]
    risk = np.asarray(ds_r[var].values, dtype=np.float32)
    ds_b = xr.open_dataset(bucket_path)
    bvar = list(ds_b.data_vars)[0]
    buckets = np.asarray(ds_b[bvar].values)
    mapping = _find_bucket_mapping(ds_b)

    # 3) 每个桶若有 calibration_<bucket>_<ym>.json 则应用
    risk2 = risk.copy()
    uniq = np.unique(buckets)
    applied = []
    for bid in uniq:
        bname = mapping.get(int(bid), str(int(bid)))
        cpath = _calib_path_for_bucket(bname, ym)
        if not cpath:
            continue
        mask = (buckets == int(bid))
        try:
            risk2[mask] = apply_calibration(risk[mask], cpath)
            applied.append(bname)
        except Exception:
            continue
    # 4) 覆写文件（保持变量名/坐标/attrs）
    ds_r[var].values[:] = risk2
    ds_r.to_netcdf(risk_path)
    try:
        ds_r.close(); ds_b.close()
    except Exception:
        pass
    payload.update({"moe": True, "by_bucket": True, "fallback": ("none" if applied else "no_calib"), "buckets_calibrated": applied})
    return payload


__all__ = ["infer_month_moe"]

