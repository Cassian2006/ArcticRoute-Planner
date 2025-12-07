from __future__ import annotations

"""D-08: 弱监督权重校准（可选开关）

提供 tune_weights(ym, seeds=20, threshold=0.02, dry_run=False, write_config=False)
- 基于“主航线走廊”或“可通航掩膜”（ArcticRoute/data_processed/corridor/corridor_prob.nc，如有）做轻量 Pairwise/LTR 风格的启发式：
  目标函数 = 掩膜区域内 Risk 的均值（越低越好）
- 过程：
  1) 读取风险分量层（复用 find_layer_paths 与变量选择 + 分位数归一 at time=0）
  2) 以 baseline 权重（来自 config/risk_fuse_<ym>.yaml 或 0.6/0.2/0.2）作为对照
  3) 随机采样 seeds 组候选权重（Dirichlet 或近邻扰动），评价目标函数
  4) 若最优候选相对改进 >= 阈值 threshold，则接受新权重，否则保持基线
- 输出：payload（含 baseline/best/improve），若 write_config=True 且被接受，则写入 ArcticRoute/config/risk_fuse_<ym>.yaml，并登记 artifact
"""

import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from ArcticRoute.core.risk.fuse_prep import find_layer_paths  # 复用发现策略
from ArcticRoute.core.risk.fusion import _quantile_norm, _pick_var  # 复用变量选择与归一
from ArcticRoute.cache.index_util import register_artifact

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CONFIG_DIR = os.path.join(REPO_ROOT, "ArcticRoute", "config")
RISK_DIR = os.path.join(REPO_ROOT, "ArcticRoute", "data_processed", "risk")
CORRIDOR_PATH = os.path.join(REPO_ROOT, "ArcticRoute", "data_processed", "corridor", "corridor_prob.nc")
REPORT_DIR = os.path.join(REPO_ROOT, "reports", "d_stage")


def _load_corridor_mask(path: str) -> Optional[np.ndarray]:
    if xr is None:
        return None
    if not os.path.exists(path):
        return None
    try:
        ds = xr.open_dataset(path)
        # 变量优先级：corridor_prob/prob/mask
        var = None
        for name in ("corridor_prob", "prob", "mask"):
            if name in ds:
                var = ds[name]
                break
        if var is None:
            names = list(ds.data_vars.keys())
            if names:
                var = ds[names[0]]
        if var is None:
            ds.close()
            return None
        if "time" in var.dims:
            var = var.isel(time=0)
        arr = np.asarray(var.values, dtype=float)
        ds.close()
        # 转为 0/1 掩膜：阈值 0.5
        mask = (arr >= 0.5)
        if mask.ndim == 3:
            mask = mask[0]
        return mask.astype(bool)
    except Exception:
        return None


def _load_normed_components(ym: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """返回归一化到 [0,1] 的各分量 2D 数组（time=0 切片），以及 used/sources 等信息。"""
    if xr is None:
        raise RuntimeError("xarray required")
    paths = find_layer_paths(ym)
    used: Dict[str, Any] = {}
    arrays: Dict[str, np.ndarray] = {}
    for kind in ("ice", "wave", "acc"):
        p = paths.get(kind)
        if not (isinstance(p, str) and os.path.exists(p)):
            continue
        try:
            ds = xr.open_dataset(p)
            var, _issues, src = _pick_var(ds, kind)
            if var is None or var not in ds:
                ds.close(); continue
            da = ds[var]
            if "time" in da.dims:
                da = da.isel(time=0)
            nda = _quantile_norm(da)
            arr = np.asarray(nda.values, dtype=float)
            if arr.ndim == 3:
                arr = arr[0]
            arrays[kind] = arr
            used[kind] = {"path": p, "var": ("R_ice(ice_cost)" if (kind=="ice" and src=="ice_cost") else var)}
            ds.close()
        except Exception:
            continue
    # 对齐形状：取共同最小 H,W 并裁剪（避免重投影）
    if not arrays:
        raise RuntimeError("无可用分量用于权重校准")
    hs = [a.shape[-2] for a in arrays.values()]
    ws = [a.shape[-1] for a in arrays.values()]
    H, W = int(min(hs)), int(min(ws))
    for k in list(arrays.keys()):
        a = arrays[k]
        arrays[k] = a[-H:, -W:]
    return arrays, used


def _load_baseline_weights(ym: str) -> Dict[str, float]:
    cfg_path = os.path.join(CONFIG_DIR, f"risk_fuse_{ym}.yaml")
    if os.path.exists(cfg_path) and yaml is not None:
        try:
            cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8").read())
            if isinstance(cfg, dict) and isinstance(cfg.get("weights"), dict):
                w = cfg["weights"]
                return {
                    "alpha": float(w.get("alpha", 0.6)),
                    "beta": float(w.get("beta", 0.2)),
                    "gamma": float(w.get("gamma", 0.2)),
                }
        except Exception:
            pass
    return {"alpha": 0.6, "beta": 0.2, "gamma": 0.2}


def _risk_from_weights(arrays: Dict[str, np.ndarray], w: Dict[str, float]) -> np.ndarray:
    present = [k for k in ("ice","wave","acc") if k in arrays]
    if not present:
        raise RuntimeError("无在场分量")
    # 对在场层重归一
    total = sum(w.get({"ice":"alpha","wave":"beta","acc":"gamma"}[k], 0.0) for k in present)
    if total <= 0:
        eff = {k: 1.0/len(present) for k in present}
    else:
        eff = {k: float(w.get({"ice":"alpha","wave":"beta","acc":"gamma"}[k], 0.0))/total for k in present}
    combo = None
    for k in present:
        term = arrays[k] * eff[k]
        combo = term if combo is None else (combo + term)
    return combo  # type: ignore


def _score_with_mask(risk: np.ndarray, mask: Optional[np.ndarray]) -> float:
    a = np.asarray(risk, dtype=float)
    if mask is not None and mask.shape == a.shape[-2:]:
        m = mask
        if a.ndim == 3:
            a = a[0]
        vals = a[m]
    else:
        vals = a[np.isfinite(a)]
    if vals.size == 0:
        return float("nan")
    return float(np.nanmean(vals))


def _rand_weights_dirichlet(rng: np.random.RandomState) -> Dict[str,float]:
    v = rng.dirichlet([1.0, 1.0, 1.0])
    return {"alpha": float(v[0]), "beta": float(v[1]), "gamma": float(v[2])}


def tune_weights(ym: str, *, seeds: int = 20, threshold: float = 0.02, dry_run: bool = False, write_config: bool = False) -> Dict[str, Any]:
    if xr is None:
        raise RuntimeError("xarray required")
    arrays, used = _load_normed_components(ym)
    mask = _load_corridor_mask(CORRIDOR_PATH)

    baseline = _load_baseline_weights(ym)
    base_risk = _risk_from_weights(arrays, baseline)
    base_score = _score_with_mask(base_risk, mask)

    rng = np.random.RandomState(42)
    best_w = dict(baseline)
    best_score = base_score

    for i in range(int(max(1, seeds))):
        cand = _rand_weights_dirichlet(rng)
        risk = _risk_from_weights(arrays, cand)
        score = _score_with_mask(risk, mask)
        # 目标更低更好
        if np.isfinite(score) and (not np.isfinite(best_score) or score < best_score):
            best_score = score
            best_w = cand

    improve = (base_score - best_score) / base_score if (np.isfinite(base_score) and base_score != 0) else 0.0
    accept = bool(np.isfinite(best_score)) and (improve >= float(threshold))

    run_id = time.strftime("%Y%m%dT%H%M%S")
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, f"tune_weights_{ym}.json")

    payload = {
        "ym": ym,
        "used": used,
        "baseline": {"weights": baseline, "score": base_score},
        "best": {"weights": best_w, "score": best_score},
        "improve": float(improve),
        "threshold": float(threshold),
        "accept": int(accept),
        "report": report_path,
    }

    if dry_run:
        return payload

    # 写报告
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # 若接受且要求写配置，则更新 config/risk_fuse_<ym>.yaml
    if accept and write_config:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        cfg_path = os.path.join(CONFIG_DIR, f"risk_fuse_{ym}.yaml")
        cfg_obj = {
            "ym": ym,
            "norm": "quantile",
            "missing": "skip_and_warn",
            "weights": {"alpha": float(best_w["alpha"]), "beta": float(best_w["beta"]), "gamma": float(best_w["gamma"])},
        }
        try:
            text = yaml.safe_dump(cfg_obj, sort_keys=False, allow_unicode=True) if yaml is not None else json.dumps(cfg_obj, ensure_ascii=False, indent=2)
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass

    # 登记 artifact
    try:
        register_artifact(run_id=run_id, kind="risk_fuse_tune", path=report_path, attrs={"ym": ym, "accept": int(accept)})
    except Exception:
        pass

    return payload


__all__ = ["tune_weights"]

