from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np  # type: ignore

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

from sklearn.neighbors import NearestNeighbors  # type: ignore

from ArcticRoute.core.prior.geo import to_xy, gc_distance_nm
from ArcticRoute.core.prior.transformer_split import make_splits

ROOT = Path(__file__).resolve().parents[2]
AOUT = ROOT / "ArcticRoute" / "data_processed" / "ais"
AOUT_ALT = ROOT / "data_processed" / "ais"
PROUT = ROOT / "ArcticRoute" / "data_processed" / "prior"
PROUT_ALT = ROOT / "data_processed" / "prior"
REPORTS = ROOT / "reports"


@dataclass
class EvalConfig:
    ym: str
    method: str = "transformer"
    tau: float = 0.5
    seed: int = 42


def _read_parquet_any(p: Path):
    if pl is not None:
        return pl.read_parquet(str(p))  # type: ignore
    return pd.read_parquet(str(p))  # type: ignore


def _to_pd(df_any: Any) -> "pd.DataFrame":  # type: ignore
    if pd is None:
        raise RuntimeError("pandas required")
    if pl is not None and isinstance(df_any, pl.DataFrame):  # type: ignore[attr-defined]
        return df_any.to_pandas()  # type: ignore
    if isinstance(df_any, pd.DataFrame):  # type: ignore[attr-defined]
        return df_any
    raise RuntimeError("unsupported df type")


def _load_centerlines_geojson(ym: str) -> Dict[str, Any]:
    path = REPORTS / "phaseE" / "center" / f"prior_centerlines_{ym}.geojson"
    if not path.exists():
        raise FileNotFoundError(f"missing centerlines: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_prior_nc(ym: str) -> "xr.Dataset":  # type: ignore
    nc = PROUT / f"prior_transformer_{ym}.nc"
    if xr is None:
        raise ImportError("xarray required to open prior raster")
    if not nc.exists():
        raise FileNotFoundError(f"missing prior raster: {nc}")
    return xr.open_dataset(nc)


def _flat_grid_latlon(ds: "xr.Dataset") -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:  # type: ignore
    lat = ds["lat"].values  # [H,W]
    lon = ds["lon"].values
    H, W = lat.shape
    pts = np.column_stack([lat.reshape(-1), lon.reshape(-1)])
    return pts, ds["P_prior"].values.reshape(-1), (H, W)


def _nearest_prior_values(ds: "xr.Dataset", lats: np.ndarray, lons: np.ndarray) -> np.ndarray:  # type: ignore
    pts, Pflat, shape = _flat_grid_latlon(ds)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(pts)
    Q = np.column_stack([lats.astype(float), lons.astype(float)])
    dist, idx = nbrs.kneighbors(Q, return_distance=True)
    idx = idx.reshape(-1)
    vals = Pflat[idx]
    return vals


def _min_distance_to_centerlines(lat: float, lon: float, features: Sequence[Dict[str, Any]]) -> float:
    # 平面近似：EPSG:3413 下点到折线最短距离（米）
    try:
        px, py = to_xy(lat, lon)
    except Exception:
        px, py = None, None
    dmin = float("inf")
    for ft in features:
        coords = ft.get("geometry", {}).get("coordinates", []) or []
        # 投影整条折线
        prev = None
        for LON, LAT in coords:
            try:
                x, y = to_xy(float(LAT), float(LON))
            except Exception:
                prev = None
                continue
            if prev is not None and px is not None and py is not None:
                x1, y1 = prev
                x2, y2 = x, y
                # 点到线段距离
                vx, vy = x2 - x1, y2 - y1
                wx, wy = px - x1, py - y1
                vv = vx * vx + vy * vy
                t = 0.0 if vv == 0 else max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
                projx = x1 + t * vx
                projy = y1 + t * vy
                dx = px - projx
                dy = py - projy
                d = (dx * dx + dy * dy) ** 0.5
                if d < dmin:
                    dmin = d
            prev = (x, y)
    if not np.isfinite(dmin):
        dmin = 0.0
    return float(dmin)


def evaluate_prior(cfg: EvalConfig) -> Tuple[Dict[str, Any], Path, Path]:
    # 数据
    seg_path = AOUT / f"segment_index_{cfg.ym}.parquet"
    if not seg_path.exists():
        seg_path = AOUT_ALT / f"segment_index_{cfg.ym}.parquet"
    trk_path = AOUT / f"tracks_{cfg.ym}.parquet"
    if not trk_path.exists():
        trk_path = AOUT_ALT / f"tracks_{cfg.ym}.parquet"
    seg = _to_pd(_read_parquet_any(seg_path))
    trk = _to_pd(_read_parquet_any(trk_path)).sort_values(["segment_id", "ts"])  # 只需要 lat/lon/ts/mmsi

    # 切分（若无外部 split，按 seed 重现）
    splits = make_splits(seg, seed=int(cfg.seed), train_ratio=0.8, stratify="mmsi")
    val_mmsi = set(splits.get("val_mmsi", []))
    # 选取 Val 点（可抽样加速）
    val_trk = trk[trk["mmsi"].isin(val_mmsi)][["mmsi", "ts", "lat", "lon"]].copy()
    if len(val_trk) > 500_000:
        val_trk = val_trk.sample(n=500_000, random_state=int(cfg.seed))

    lats = val_trk["lat"].to_numpy(dtype=float)
    lons = val_trk["lon"].to_numpy(dtype=float)

    # 若无验证样本，输出零值指标并返回（避免 NearestNeighbors 报错）
    if lats.size == 0 or lons.size == 0:
        metrics = {
            "ym": cfg.ym,
            "method": cfg.method,
            "tau": float(cfg.tau),
            "coverage": 0.0,
            "deviation_mean_m": 0.0,
            "deviation_p95_m": 0.0,
            "stability_day_var": 0.0,
            "stability_week_var": 0.0,
            "n_val_points": 0,
            "n_val_sample_for_deviation": 0,
        }
        rep_dir = REPORTS / "phaseE"
        rep_dir.mkdir(parents=True, exist_ok=True)
        json_path = rep_dir / f"prior_metrics_{cfg.ym}.json"
        html_path = rep_dir / f"prior_summary_{cfg.ym}.html"
        json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = [
            f"<html><head><meta charset='utf-8'><title>Prior Summary {cfg.ym}</title></head><body>",
            f"<h1>Prior Summary ({cfg.ym})</h1>",
            f"<p>method={cfg.method} tau={cfg.tau}</p>",
            f"<ul>",
            f"  <li>coverage: 0.0000</li>",
            f"  <li>deviation_mean_m: 0.00 m</li>",
            f"  <li>deviation_p95_m: 0.00 m</li>",
            f"  <li>stability_day_var: 0.000000</li>",
            f"  <li>stability_week_var: 0.000000</li>",
            f"  <li>n_val_points: 0</li>",
            f"</ul>",
            f"</body></html>",
        ]
        html_path.write_text("\n".join(lines), encoding="utf-8")
        return metrics, json_path, html_path

    # Prior 栅格
    ds = _load_prior_nc(cfg.ym)
    Pvals = _nearest_prior_values(ds, lats, lons)

    # 覆盖率（P_prior >= tau）
    covered = (Pvals >= float(cfg.tau)).astype(np.float32)
    coverage = float(covered.mean() if len(covered) else 0.0)

    # 横向偏差：到最近中心线
    cjl = _load_centerlines_geojson(cfg.ym)
    feats = [ft for ft in cjl.get("features", [])]
    dlist: List[float] = []
    # 取样避免 O(N*M) 爆炸
    sample_idx = np.linspace(0, len(val_trk) - 1, num=min(len(val_trk), 100_000), dtype=int)
    for i in sample_idx:
        d_m = _min_distance_to_centerlines(float(lats[i]), float(lons[i]), feats)
        dlist.append(d_m)
    if dlist:
        d_arr = np.array(dlist, dtype=float)
        dev_mean_m = float(np.mean(d_arr))
        dev_p95_m = float(np.quantile(d_arr, 0.95))
    else:
        dev_mean_m = 0.0
        dev_p95_m = 0.0

    # 稳健性：按日/周覆盖率方差
    # 计算每点是否覆盖
    val_trk = val_trk.reset_index(drop=True)
    val_trk["covered"] = (Pvals >= float(cfg.tau))
    val_trk["date"] = pd.to_datetime(val_trk["ts"], unit="s", utc=True).dt.date  # type: ignore
    by_day = val_trk.groupby("date")["covered"].mean().to_numpy(dtype=float)
    day_var = float(np.var(by_day)) if len(by_day) else 0.0
    # ISO 周
    val_trk["week"] = pd.to_datetime(val_trk["ts"], unit="s", utc=True).dt.isocalendar().week.astype(int)  # type: ignore
    by_week = val_trk.groupby("week")["covered"].mean().to_numpy(dtype=float)
    week_var = float(np.var(by_week)) if len(by_week) else 0.0

    metrics = {
        "ym": cfg.ym,
        "method": cfg.method,
        "tau": float(cfg.tau),
        "coverage": coverage,
        "deviation_mean_m": dev_mean_m,
        "deviation_p95_m": dev_p95_m,
        "stability_day_var": day_var,
        "stability_week_var": week_var,
        "n_val_points": int(len(Pvals)),
        "n_val_sample_for_deviation": int(len(dlist)),
    }

    # 写报告
    rep_dir = REPORTS / "phaseE"
    rep_dir.mkdir(parents=True, exist_ok=True)
    json_path = rep_dir / f"prior_metrics_{cfg.ym}.json"
    html_path = rep_dir / f"prior_summary_{cfg.ym}.html"
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    # 简单 HTML
    lines = [
        f"<html><head><meta charset='utf-8'><title>Prior Summary {cfg.ym}</title></head><body>",
        f"<h1>Prior Summary ({cfg.ym})</h1>",
        f"<p>method={cfg.method} tau={cfg.tau}</p>",
        f"<ul>",
        f"  <li>coverage: {coverage:.4f}</li>",
        f"  <li>deviation_mean_m: {dev_mean_m:.2f} m</li>",
        f"  <li>deviation_p95_m: {dev_p95_m:.2f} m</li>",
        f"  <li>stability_day_var: {day_var:.6f}</li>",
        f"  <li>stability_week_var: {week_var:.6f}</li>",
        f"  <li>n_val_points: {int(len(Pvals))}</li>",
        f"</ul>",
        f"</body></html>",
    ]
    html_path.write_text("\n".join(lines), encoding="utf-8")

    return metrics, json_path, html_path


__all__ = ["EvalConfig", "evaluate_prior"]








