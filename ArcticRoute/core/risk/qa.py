from __future__ import annotations

"""D-05: 融合 QA 与贡献度可视化

summarize_fused(ym, contour=True, out_dir=outputs):
- 读取 ArcticRoute/data_processed/risk/risk_fused_<ym>.nc
- 产出：
  - outputs/risk_fused_<ym>.png （热力图，time 取第 0 帧；可选叠加等值线）
  - outputs/risk_fused_contrib_<ym>.png （α/β/γ 条形图，优先 weights_effective 回退 weights）
  - outputs/risk_fused_<ym>.json （统计摘要：nan_pct、非零率、q01/05/50/95/99）
- 写盘并 register_artifact
"""

import os
import json
from typing import Any, Dict, Optional

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

from ArcticRoute.cache.index_util import register_artifact

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RISK_DIR = os.path.join(REPO_ROOT, "ArcticRoute", "data_processed", "risk")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "outputs")


def _parse_weights(attrs: Dict[str, Any]) -> Dict[str, float]:
    def _to_dict_str(s: Any) -> Dict[str, Any]:
        if isinstance(s, dict):
            return s
        if isinstance(s, str):
            try:
                return json.loads(s.replace("'", '"'))  # 宽松解析
            except Exception:
                pass
        return {}
    we = _to_dict_str(attrs.get("weights_effective"))
    if not we:
        we = _to_dict_str(attrs.get("weights"))
    out = {k: float(v) for k, v in we.items() if isinstance(v, (int, float, str))}
    # 仅保留 alpha/beta/gamma
    return {k: float(out.get(k, 0.0)) for k in ("alpha", "beta", "gamma")}


def _stats(da: "xr.DataArray") -> Dict[str, Any]:
    if "time" in da.dims and int(da.sizes.get("time", 0)) > 0:
        sample = da.isel(time=0).values
    else:
        sample = da.values
    arr = np.asarray(sample, dtype=float)
    nan_pct = float(np.isnan(arr).mean() * 100.0)
    finite = arr[np.isfinite(arr)]
    nnz = float((finite > 0).mean() * 100.0) if finite.size else 0.0

    def q(p: float) -> float:
        try:
            return float(np.nanpercentile(finite, p)) if finite.size else float("nan")
        except Exception:
            return float("nan")

    return {
        "nan_pct": round(nan_pct, 3),
        "nonzero_pct": round(nnz, 3),
        "q01": q(1),
        "q05": q(5),
        "q50": q(50),
        "q95": q(95),
        "q99": q(99),
    }


def summarize_fused(ym: str, *, contour: bool = True, out_dir: Optional[str] = None, dry_run: bool = False) -> Dict[str, Any]:
    if xr is None:
        raise RuntimeError("xarray required")
    nc_path = os.path.join(RISK_DIR, f"risk_fused_{ym}.nc")
    if not os.path.exists(nc_path):
        raise FileNotFoundError(nc_path)

    out_dir = out_dir or DEFAULT_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    ds = xr.open_dataset(nc_path)
    if "Risk" not in ds:
        try:
            ds.close()
        except Exception:
            pass
        raise RuntimeError("Risk 变量不存在于 fused 文件中")

    da = ds["Risk"]
    # 选择展示帧
    view = da.isel(time=0) if ("time" in da.dims and int(da.sizes.get("time", 0)) > 0) else da

    png_map = os.path.join(out_dir, f"risk_fused_{ym}.png")
    png_bar = os.path.join(out_dir, f"risk_fused_contrib_{ym}.png")
    js_path = os.path.join(out_dir, f"risk_fused_{ym}.json")

    payload: Dict[str, Any] = {"ym": ym, "map": png_map, "contrib": png_bar, "json": js_path, "nc": nc_path}

    # 1) 热力图（可选等值线）
    if plt is not None:
        try:
            arr = np.asarray(view.values, dtype=float)
            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_subplot(111)
            im = ax.imshow(arr, origin="upper", cmap="inferno")
            ax.set_title(f"Fused Risk {ym}")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Risk")
            if contour:
                try:
                    levels = np.linspace(np.nanpercentile(arr, 10), np.nanpercentile(arr, 90), 5)
                    ax.contour(arr, levels=levels, colors="white", linewidths=0.6, alpha=0.7)
                except Exception:
                    pass
            fig.tight_layout()
            if not dry_run:
                fig.savefig(png_map, dpi=150)
            plt.close(fig)
        except Exception:
            pass

    # 2) 贡献度条形图（α/β/γ）
    weights = _parse_weights(ds.attrs)
    payload["weights"] = weights
    if plt is not None:
        try:
            labels = ["alpha(R_ice)", "beta(R_wave)", "gamma(R_acc)"]
            values = [weights.get("alpha", 0.0), weights.get("beta", 0.0), weights.get("gamma", 0.0)]
            fig = plt.figure(figsize=(5.2, 3.2))
            ax = fig.add_subplot(111)
            ax.bar(labels, values, color=["#2a9d8f", "#264653", "#e76f51"])
            ax.set_ylim(0, 1)
            ax.set_ylabel("weight")
            ax.set_title("Fused Weights")
            fig.tight_layout()
            if not dry_run:
                fig.savefig(png_bar, dpi=150)
            plt.close(fig)
        except Exception:
            pass

    # 3) 统计 JSON（time=0 或静态）
    st = _stats(da)
    payload["stats"] = st
    if not dry_run:
        try:
            with open(js_path, "w", encoding="utf-8") as f:
                json.dump({"ym": ym, "stats": st, "weights": weights, "nc": nc_path}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # 注册索引（若写盘）
    if not dry_run:
        try:
            run_id = os.environ.get("RUN_ID", "") or __import__("time").strftime("%Y%m%dT%H%M%S")
        except Exception:
            run_id = ""
        try:
            register_artifact(run_id=run_id, kind="risk_fused_map", path=png_map, attrs={"ym": ym})
            register_artifact(run_id=run_id, kind="risk_fused_contrib", path=png_bar, attrs={"ym": ym})
            register_artifact(run_id=run_id, kind="risk_fused_stats", path=js_path, attrs={"ym": ym})
        except Exception:
            pass

    try:
        ds.close()
    except Exception:
        pass

    return payload


__all__ = ["summarize_fused"]
