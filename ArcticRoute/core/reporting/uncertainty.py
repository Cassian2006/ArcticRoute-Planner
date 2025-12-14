from __future__ import annotations

"""
Phase I | 不确定性报告（最小可用版）

提供：
- build_month(ym): 读取 fused 风险（Risk[必需], RiskVar[可选]），构建弱标签，计算：
  - Metrics: NLL, Brier, ECE, Var-Calibration（方差 vs 误差的相关性/曲线）, Sharpness（avg std）
  - Plots: 可靠性曲线（REUSE Phase H）、方差直方 & 方差-误差散点/分箱
  - 产出：reports/d_stage/phaseI/uncertainty_<ym>.json / .png

若无 RiskVar，则统计中剔除 variance 相关项，但流程可跑通。
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore

from .calibration import build_weak_labels, reliability_curve, metrics as calib_metrics  # REUSE

REPO_ROOT = Path(__file__).resolve().parents[3]
RISK_DIR = REPO_ROOT / "ArcticRoute" / "data_processed" / "risk"
FEAT_DIR = REPO_ROOT / "ArcticRoute" / "data_processed" / "features"
REPORT_DIR = REPO_ROOT / "ArcticRoute" / "reports" / "d_stage" / "phaseI"


def _arr2d(da: "xr.DataArray") -> np.ndarray:
    arr = da
    if "time" in arr.dims and int(arr.sizes.get("time", 0)) > 0:
        arr = arr.isel(time=0)
    A = np.asarray(arr.values, dtype=float)
    if A.ndim == 2:
        return A
    A2 = np.squeeze(A)
    if A2.ndim > 2:
        axes = tuple(range(0, A2.ndim - 2))
        A2 = A2.mean(axis=axes)
    return A2


def _ece_brier_auc(P: "xr.DataArray", Y: "xr.DataArray") -> Dict[str, float]:
    return calib_metrics(P, Y, n_bins=15)


def _nll(P: np.ndarray, Y: np.ndarray) -> float:
    eps = 1e-6
    p = np.clip(P, eps, 1.0 - eps)
    y = (Y > 0.5).astype(float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _var_calibration(P: np.ndarray, Y: np.ndarray, V: Optional[np.ndarray]) -> Dict[str, Any]:
    # 返回：方差分箱的均值-误差关系与相关系数
    if V is None:
        return {"bins": [], "var_centers": [], "mse_by_bin": [], "corr": float("nan")}
    m = np.isfinite(P) & np.isfinite(Y) & np.isfinite(V)
    if not np.any(m):
        return {"bins": [], "var_centers": [], "mse_by_bin": [], "corr": float("nan")}
    p = np.clip(P[m], 0.0, 1.0)
    y = (Y[m] > 0.5).astype(float)
    v = np.maximum(V[m], 0.0)
    err2 = (p - y) ** 2
    # 皮尔森相关（方差大 -> 误差应大）
    try:
        corr = float(np.corrcoef(v, err2)[0, 1])
    except Exception:
        corr = float("nan")
    # 按方差分箱
    n_bins = 10
    bins = np.quantile(v, np.linspace(0.0, 1.0, n_bins + 1))
    idx = np.digitize(v, bins, right=True) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    mse = []
    for b in range(n_bins):
        sel = idx == b
        if np.any(sel):
            mse.append(float(np.mean(err2[sel])))
        else:
            mse.append(float("nan"))
    return {"bins": bins.tolist(), "var_centers": centers.tolist(), "mse_by_bin": mse, "corr": corr}


def _sharpness(V: Optional[np.ndarray]) -> float:
    if V is None:
        return float("nan")
    v = np.asarray(V, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.mean(np.sqrt(np.maximum(v, 0.0))))


def build_month(ym: str) -> Dict[str, str]:
    if xr is None:
        raise RuntimeError("xarray required")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%dT%H%M%S")

    risk_path = RISK_DIR / f"risk_fused_{ym}.nc"
    if not risk_path.exists():
        raise FileNotFoundError(str(risk_path))
    ds = xr.open_dataset(risk_path)
    # 预测均值 P
    da = ds["Risk"] if "Risk" in ds else ds[list(ds.data_vars)[0]]
    P = xr.where(xr.ufuncs.isfinite(da), xr.apply_ufunc(np.clip, da, 0.0, 1.0), np.nan)
    # 方差 V（可选）
    v_da = None
    for cand in ("RiskVar", "risk_var", "Var", "variance", "risk_variance"):
        if cand in ds:
            v_da = ds[cand]
            break
    V = xr.where(xr.ufuncs.isfinite(v_da), v_da, np.nan) if v_da is not None else None

    # 弱标签（REUSE 与 Phase H 相同策略）
    dens_path = FEAT_DIR / f"ais_density_{ym}.nc"
    dens = xr.open_dataset(dens_path)[list(xr.open_dataset(dens_path).data_vars)[0]] if dens_path.exists() else None
    labels = build_weak_labels(dens, None, None, cfg={}) if dens is not None else xr.DataArray((P.values >= np.nanmedian(P.values)).astype(float), dims=P.dims, coords=P.coords)

    # 指标与曲线
    met = _ece_brier_auc(P, labels)
    P2 = _arr2d(P); Y2 = _arr2d(labels)
    nll = _nll(P2[np.isfinite(P2) & np.isfinite(Y2)], Y2[np.isfinite(P2) & np.isfinite(Y2)]) if np.isfinite(P2).any() else float("nan")
    rel = reliability_curve(P, labels, n_bins=15)

    V2 = _arr2d(V) if V is not None else None
    varcal = _var_calibration(P2, Y2, V2)
    sharp = _sharpness(V2)

    js = {
        "ym": ym,
        "metrics": {"ece": met.get("ece"), "brier": met.get("brier"), "auc": met.get("auc"), "nll": nll, "sharpness": sharp},
        "reliability": rel,
        "var_calibration": varcal,
        "has_variance": bool(V is not None),
    }

    js_path = REPORT_DIR / f"uncertainty_{ym}.json"
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(js, f, ensure_ascii=False, indent=2)
    with open(str(js_path) + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"logical_id": js_path.name, "inputs": [str(risk_path)], "run_id": run_id}, f, ensure_ascii=False, indent=2)

    png_path = REPORT_DIR / f"uncertainty_{ym}.png"
    if plt is not None:
        try:
            fig, axes = plt.subplots(1, 3 if V is not None else 2, figsize=(11, 3.6))
            ax0 = axes[0]
            centers = np.asarray(rel.get("bin_centers") or [], dtype=float)
            pos_rate = np.asarray(rel.get("pos_rate") or [], dtype=float)
            ax0.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
            ax0.plot(centers, pos_rate, "o-", label="empirical")
            ax0.set_title(f"Reliability (ECE={met.get('ece', float('nan')):.3f})")
            ax0.set_xlabel("predicted risk"); ax0.set_ylabel("positive rate"); ax0.legend()

            ax1 = axes[1]
            if V is not None:
                vflat = np.asarray(V2).ravel()
                vflat = vflat[np.isfinite(vflat)]
                ax1.hist(np.sqrt(np.maximum(vflat, 0.0)), bins=20, color="#4c72b0")
                ax1.set_title(f"Sharpness (avg std={sharp:.3f})")
                ax1.set_xlabel("std"); ax1.set_ylabel("count")

                ax2 = axes[2]
                vc = varcal
                x = np.asarray(vc.get("var_centers") or [], dtype=float)
                y = np.asarray(vc.get("mse_by_bin") or [], dtype=float)
                ax2.plot(x, y, "o-")
                ax2.set_title(f"Var-Calibration (corr={vc.get('corr', float('nan')):.3f})")
                ax2.set_xlabel("variance bin center"); ax2.set_ylabel("MSE")
            else:
                # 没有方差时，第二个面板显示预测分布
                pflat = np.asarray(P2).ravel()
                pflat = pflat[np.isfinite(pflat)]
                ax1.hist(pflat, bins=20, color="#4c72b0")
                ax1.set_title("Pred Dist (no variance)")
                ax1.set_xlabel("p"); ax1.set_ylabel("count")

            fig.suptitle(f"Uncertainty Report {ym}")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            with open(str(png_path) + ".meta.json", "w", encoding="utf-8") as f:
                json.dump({"logical_id": png_path.name, "inputs": [str(js_path)], "run_id": run_id}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    try:
        ds.close()
    except Exception:
        pass

    return {"json": str(js_path), "png": str(png_path)}


__all__ = ["build_month"]



