"""Features QA 摘要与可视化快照

- summarize_features(month, features_nc_path, out_json, out_png, recon_summary_path=None, dry_run=True)
  - 统计：非零格点数、最大/均值、时间覆盖率、清洗掉的比例（来自 B-04/05 摘要）
  - 可视：导出 ais_density 的 PNG（时间均值，若存在 ais_density_cls 则对合计层可视）
  - 非 dry-run 写盘并登记工件

约束：
- 写盘采用 register_artifact()
- Windows 路径 os.path.join
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import xarray as xr

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore

from ArcticRoute.cache.index_util import register_artifact


def _default_paths(month: str) -> Dict[str, str]:
    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    return {
        "json": os.path.join(out_dir, f"features_summary_{month}.json"),
        "png": os.path.join(out_dir, f"features_{month}.png"),
    }


def _load_features_summary() -> Dict[str, Any]:
    # 可选：从 reports/recon/features_summary.json 聚合 B-04/05 的清洗统计
    path = os.path.join(os.getcwd(), "reports", "recon", "features_summary.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def summarize_features(
    month: str,
    features_nc_path: str,
    out_json: Optional[str] = None,
    out_png: Optional[str] = None,
    recon_summary_path: Optional[str] = None,
    dry_run: bool = True,
) -> Dict[str, Any]:
    paths = _default_paths(month)
    out_json = out_json or paths["json"]
    out_png = out_png or paths["png"]

    ds = xr.open_dataset(features_nc_path)
    try:
        # 选择变量：优先 ais_density；其次 ais_density_cls 合计层
        if "ais_density" in ds:
            da = ds["ais_density"]
        elif "ais_density_cls" in ds:
            da = ds["ais_density_cls"].sum(dim="vclass")
        else:
            raise ValueError("features 数据集中不含 ais_density/ais_density_cls")
        # 基础统计
        arr = da.values  # (time,y,x)
        nonzero = int(np.count_nonzero(arr))
        maxv = float(np.nanmax(arr)) if arr.size else 0.0
        meanv = float(np.nanmean(arr)) if arr.size else 0.0
        # 覆盖率：非零比率
        cover_pct = 100.0 * (nonzero / float(arr.size)) if arr.size else 0.0
        # 聚合清洗统计
        clean_stats = _load_features_summary()
        # 组装摘要
        summary = {
            "month": month,
            "dims": {k: int(v) for k, v in ds.sizes.items()},
            "nonzero": nonzero,
            "max": round(maxv, 3),
            "mean": round(meanv, 3),
            "coverage_pct": round(cover_pct, 3),
            "cleaning": {
                "raw_cnt": clean_stats.get("raw_cnt"),
                "kept_cnt": clean_stats.get("kept_cnt"),
                "drop_cnt": clean_stats.get("drop_cnt"),
                "drop_reasons": clean_stats.get("drop_reasons"),
            },
        }
        # PNG 可视：时间均值
        if plt is not None:
            mean_map = np.nanmean(arr, axis=0) if arr.ndim == 3 else arr
            fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
            im = ax.imshow(mean_map, origin="lower")
            ax.set_title(f"AIS density mean {month}")
            fig.colorbar(im, ax=ax, shrink=0.8)
            if not dry_run:
                os.makedirs(os.path.dirname(out_png), exist_ok=True)
                plt.savefig(out_png, bbox_inches="tight")
            plt.close(fig)
        # 写 JSON
        if not dry_run:
            os.makedirs(os.path.dirname(out_json), exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as fw:
                json.dump(summary, fw, ensure_ascii=False, indent=2)
            try:
                register_artifact(run_id=os.environ.get("RUN_ID", ""), kind="features_summary", path=out_json, attrs={"month": month})
                if plt is not None:
                    register_artifact(run_id=os.environ.get("RUN_ID", ""), kind="features_png", path=out_png, attrs={"month": month})
            except Exception:
                pass
    finally:
        try:
            ds.close()
        except Exception:
            pass
    return {"json": out_json, "png": out_png}


__all__ = ["summarize_features"]

