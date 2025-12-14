from __future__ import annotations

# Phase H | Calibration & reliability
# 提供：
# - build_weak_labels(ais_density, incidents, ice_mask, cfg) -> xr.DataArray{0/1/NaN}
# - reliability_curve(pred: xr.DataArray, labels: xr.DataArray, n_bins=15) -> dict
# - metrics(pred, labels) -> {"ece":..., "brier":..., "auc":...}
# - build_month(ym) 高阶入口：读取 fused 风险 + 构造弱标签 → 输出 JSON/PNG 与 .meta.json

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

REPO_ROOT = Path(__file__).resolve().parents[3]
RISK_DIR = REPO_ROOT / "ArcticRoute" / "data_processed" / "risk"
FEAT_DIR = REPO_ROOT / "ArcticRoute" / "data_processed" / "features"
REPORT_DIR = REPO_ROOT / "ArcticRoute" / "reports" / "d_stage" / "phaseH"


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


def build_weak_labels(
    ais_density: Optional["xr.DataArray"], incidents: Optional["xr.DataArray"], ice_mask: Optional["xr.DataArray"], cfg: Optional[Dict[str, Any]] = None
) -> "xr.DataArray":
    if xr is None:
        raise RuntimeError("xarray required")
    cfg = cfg or {}
    # 目标：正样本=走廊高密度；负样本=事故/禁航(冰)；其余 NaN
    like = None
    lat = None; lon = None
    if ais_density is not None:
        a = _arr2d(ais_density)
        thr = float(cfg.get("density_threshold", np.nanpercentile(a[np.isfinite(a)], 75) if np.isfinite(a).any() else 0.0))
        pos = (a >= thr).astype(float)
        like = pos if like is None else np.maximum(like, pos)
        lat = ais_density.coords.get("lat") or ais_density.coords.get("latitude")
        lon = ais_density.coords.get("lon") or ais_density.coords.get("longitude")
    if incidents is not None:
        b = _arr2d(incidents)
        neg = (b > 0).astype(float)
        like = (1 - neg) if like is None else np.where(neg > 0, 0.0, like)
        if lat is None:
            lat = incidents.coords.get("lat") or incidents.coords.get("latitude")
            lon = incidents.coords.get("lon") or incidents.coords.get("longitude")
    if ice_mask is not None:
        c = _arr2d(ice_mask)
        # 冰区视为负样本（不可通行）
        neg = (c > 0.5).astype(float)
        like = (1 - neg) if like is None else np.where(neg > 0, 0.0, like)
        if lat is None:
            lat = ice_mask.coords.get("lat") or ice_mask.coords.get("latitude")
            lon = ice_mask.coords.get("lon") or ice_mask.coords.get("longitude")
    if like is None:
        # 无弱标签来源，则返回全 NaN 的 DataArray（让上游优雅降级）
        shape = (ais_density.sizes.get("y", 64), ais_density.sizes.get("x", 64)) if ais_density is not None else (64, 64)
        return xr.DataArray(np.full(shape, np.nan, dtype=float), dims=("y", "x"))
    return xr.DataArray(like, coords={"lat": lat, "lon": lon}, dims=(like.shape[-2] and "y" or "y", like.shape[-1] and "x" or "x"))


def reliability_curve(pred: "xr.DataArray", labels: "xr.DataArray", n_bins: int = 15) -> Dict[str, Any]:
    # 对齐尺寸：以标签为准
    try:
        if (tuple(pred.dims) != tuple(labels.dims)) or any(int(pred.sizes.get(d, -1)) != int(labels.sizes.get(d, -2)) for d in labels.dims):
            pred = pred.interp_like(labels, method="nearest")
    except Exception:
        pass # Fallback to numpy-level alignment
    P = _arr2d(pred)
    L = _arr2d(labels)
    if P.shape != L.shape:
        try:
            from skimage.transform import resize as _resize  # type: ignore
            P = _resize(P, L.shape, order=1, mode='edge', anti_aliasing=True, preserve_range=True)
        except Exception:
            H = min(P.shape[0], L.shape[0]); W = min(P.shape[1], L.shape[1])
            L = L[:H,:W]
            P = P[:H,:W]
    m = np.isfinite(P) & np.isfinite(L)
    p = np.clip(P[m], 0.0, 1.0)
    y = (L[m] > 0.5).astype(float)
    if p.size == 0:
        return {"bin_centers": [], "pos_rate": [], "counts": []}
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins, right=True) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    pos_rate = []
    counts = []
    for b in range(n_bins):
        sel = (idx == b)
        counts.append(int(sel.sum()))
        if sel.any():
            pos_rate.append(float(y[sel].mean()))
        else:
            pos_rate.append(float("nan"))
    return {"bin_centers": centers.tolist(), "pos_rate": pos_rate, "counts": counts}


def _ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins, right=True) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = (idx == b)
        if not sel.any():
            continue
        conf = float(p[sel].mean())
        acc = float(y[sel].mean())
        ece += (sel.mean()) * abs(conf - acc)
    return float(ece)


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _auc(p: np.ndarray, y: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore
        return float(roc_auc_score(y, p))
    except Exception:
        # 简易 AUC 估计（Mann–Whitney U）
        pos = p[y > 0.5]
        neg = p[y <= 0.5]
        if pos.size == 0 or neg.size == 0:
            return float("nan")
        # 近似：P(pos > neg)
        return float(np.mean(pos.reshape(-1, 1) > neg.reshape(1, -1)))


def metrics(pred: "xr.DataArray", labels: "xr.DataArray", n_bins: int = 15) -> Dict[str, float]:
    # 对齐预测 P 到标签 L 的网格
    if not (pred.dims == labels.dims and pred.shape == labels.shape):
        try:
            pred = pred.interp_like(labels, method="nearest")
        except Exception:
            try:
                from skimage.transform import resize as _resize
                p_arr = _arr2d(pred)
                l_arr = _arr2d(labels)
                p_resized = _resize(p_arr, l_arr.shape, order=1, mode='edge', anti_aliasing=True, preserve_range=True)
                pred = xr.DataArray(p_resized, dims=labels.dims, coords=labels.coords)
            except Exception:
                pass  # Fallback to original arrays, might fail but worth a try

    P = _arr2d(pred)
    L = _arr2d(labels)

    # Final check: crop to smallest common shape if still not aligned
    if P.shape != L.shape:
        H = min(P.shape[0], L.shape[0])
        W = min(P.shape[1], L.shape[1])
        P = P[:H, :W]
        L = L[:H, :W]

    m = np.isfinite(P) & np.isfinite(L)
    p = np.clip(P[m], 0.0, 1.0)
    y = (L[m] > 0.5).astype(float)
    if p.size == 0:
        return {"ece": float("nan"), "brier": float("nan"), "auc": float("nan")}
    return {"ece": _ece(p, y, n_bins=n_bins), "brier": _brier(p, y), "auc": _auc(p, y)}


def _git_sha(repo: Path) -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo), stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _hash_obj(obj: object) -> str:
    import hashlib
    m = hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    return m.hexdigest()[:16]


def build_month(ym: str) -> Dict[str, str]:
    if xr is None:
        raise RuntimeError("xarray required")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%dT%H%M%S")
    # 读取总体预测：fused 风险视为概率/风险（0-1 裁剪）
    risk_path = RISK_DIR / f"risk_fused_{ym}.nc"
    if not risk_path.exists():
        raise FileNotFoundError(str(risk_path))
    ds = xr.open_dataset(risk_path)
    da = ds["Risk"] if "Risk" in ds else ds[list(ds.data_vars)[0]]
    P = xr.where(xr.ufuncs.isfinite(da), xr.apply_ufunc(np.clip, da, 0.0, 1.0), np.nan)

    # 弱标签：若缺失则用简单阈值自引导（使流程可跑通）
    dens_path = FEAT_DIR / f"ais_density_{ym}.nc"
    dens = xr.open_dataset(dens_path)[list(xr.open_dataset(dens_path).data_vars)[0]] if dens_path.exists() else None
    labels = build_weak_labels(dens, None, None, cfg={}) if dens is not None else xr.DataArray((P.values >= np.nanmedian(P.values)).astype(float), dims=P.dims, coords=P.coords)

    # 尺寸/坐标对齐（以预测 P 为准）
    try:
        labels = labels.interp_like(P, method="nearest")
    except Exception:
        pass
    # 对齐预测到标签网格（优先），避免形状不一致
    try:
        P = P.interp_like(labels, method="nearest")
    except Exception:
        pass

    # 对齐预测到标签网格（优先使用 xarray 的 interp_like，失败则 resize 回退）
    P_aligned = P
    try:
        P_aligned = P.interp_like(labels, method="nearest")
    except Exception:
        try:
            import numpy as _np
            from skimage.transform import resize as _resize  # type: ignore
            A = _np.asarray(P.values)
            H, W = int(labels.sizes.get("y", A.shape[-2])), int(labels.sizes.get("x", A.shape[-1]))
            Ar = _resize(A, (H, W), order=0, mode='edge', anti_aliasing=False, preserve_range=True).astype(float)
            P_aligned = xr.DataArray(Ar, dims=labels.dims, coords=labels.coords)
        except Exception:
            pass

    # overall 指标与可靠性曲线
    met = metrics(P_aligned, labels, n_bins=15)
    rel = reliability_curve(P_aligned, labels, n_bins=15)

    # per-mode（若存在专用模态预测层则计算；否则跳过）
    per_mode = {}
    mode_files = {
        "sea": risk_path,  # 复用 fused 作为 sea
        "rail": RISK_DIR / f"risk_rail_{ym}.nc",
        "road": RISK_DIR / f"risk_road_{ym}.nc",
    }
    for m, p in mode_files.items():
        if m != "sea" and (not Path(p).exists()):
            continue
        try:
            if m == "sea":
                Pm = P
            else:
                dsm = xr.open_dataset(p)
                dam = dsm["risk"] if "risk" in dsm else dsm[list(dsm.data_vars)[0]]
                Pm = xr.where(xr.ufuncs.isfinite(dam), xr.apply_ufunc(np.clip, dam, 0.0, 1.0), np.nan)
            per_mode[m] = {"metrics": metrics(Pm, labels, n_bins=15), "reliability": reliability_curve(Pm, labels, n_bins=15)}
            if m != "sea":
                try:
                    dsm.close()
                except Exception:
                    pass
        except Exception:
            continue

    # 写 JSON（包含 overall + per_mode）
    js_path = REPORT_DIR / f"calibration_{ym}.json"
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump({"ym": ym, "metrics": met, "reliability": rel, "per_mode": per_mode}, f, ensure_ascii=False, indent=2)
    with open(str(js_path) + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"logical_id": js_path.name, "inputs": [str(risk_path)], "run_id": run_id, "git_sha": _git_sha(REPO_ROOT), "config_hash": _hash_obj({})}, f, ensure_ascii=False, indent=2)

    # 画图：左侧 overall，右侧（如有）叠加 per-mode 曲线
    png_path = REPORT_DIR / f"calibration_{ym}.png"
    if plt is not None:
        try:
            centers = np.asarray(rel.get("bin_centers") or [], dtype=float)
            pos_rate = np.asarray(rel.get("pos_rate") or [], dtype=float)
            fig, ax = plt.subplots(figsize=(6.0, 4.2))
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
            ax.plot(centers, pos_rate, "o-", label=f"overall (ECE={met.get('ece', float('nan')):.3f})")
            # per-mode 叠加
            for m, pr in per_mode.items():
                cm = np.asarray(pr["reliability"].get("bin_centers") or [], dtype=float)
                rm = np.asarray(pr["reliability"].get("pos_rate") or [], dtype=float)
                ax.plot(cm, rm, "o-", label=f"{m}")
            ax.set_xlabel("predicted risk")
            ax.set_ylabel("positive rate")
            ax.set_title(f"Reliability {ym}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            with open(str(png_path) + ".meta.json", "w", encoding="utf-8") as f:
                json.dump({"logical_id": png_path.name, "inputs": [str(js_path)], "run_id": run_id, "git_sha": _git_sha(REPO_ROOT), "config_hash": _hash_obj({})}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    try:
        ds.close()
    except Exception:
        pass
    return {"json": str(js_path), "png": str(png_path)}


__all__ = ["build_weak_labels", "reliability_curve", "metrics", "build_month"]



