from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np  # type: ignore

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from ArcticRoute.core.prior.geo import gc_distance_nm
from ArcticRoute.cache.index_util import register_artifact

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data_processed" / "ais"
REPORT_DIR = ROOT / "reports" / "phaseE"


@dataclass
class CenterlineConfig:
    ym: str
    band_quantile: float = 0.75
    min_cluster_size: int = 30
    resample_points: int = 100
    out_dir: Optional[str] = None
    dry_run: bool = False


def _read_parquet_any(p: Path):
    if pl is not None:
        return pl.read_parquet(str(p))  # type: ignore
    return pd.read_parquet(str(p))  # type: ignore


def _to_pandas(df_any: Any) -> "pd.DataFrame":  # type: ignore
    if pd is None:
        raise RuntimeError("pandas required")
    if pl is not None and isinstance(df_any, pl.DataFrame):  # type: ignore[attr-defined]
        return df_any.to_pandas()  # type: ignore
    if isinstance(df_any, pd.DataFrame):  # type: ignore[attr-defined]
        return df_any
    raise RuntimeError("Unsupported DF type")


def _cumu_dist_m(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    n = len(lat)
    d = np.zeros(n, dtype=float)
    if n <= 1:
        return d
    for i in range(1, n):
        d[i] = d[i-1] + gc_distance_nm(float(lat[i-1]), float(lon[i-1]), float(lat[i]), float(lon[i])) * 1852.0
    return d


def _resample_by_arclen(lat: np.ndarray, lon: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(lat) == 0:
        return np.zeros(m, dtype=float), np.zeros(m, dtype=float)
    s = _cumu_dist_m(lat, lon)
    if s[-1] <= 0:
        # 全部相同点，重复
        return np.full(m, float(lat[0])), np.full(m, float(lon[0]))
    targets = np.linspace(0.0, float(s[-1]), num=m)
    lat_out = np.interp(targets, s, lat)
    lon_out = np.interp(targets, s, lon)
    return lat_out, lon_out


def _cluster_centerline_latlon(trajs: List[Tuple[np.ndarray, np.ndarray]], m: int) -> Tuple[np.ndarray, np.ndarray]:
    # 逐轨 resample 到 m 点，然后按点位做中位数（DBA 近似）
    if not trajs:
        return np.zeros(m), np.zeros(m)
    L = []
    G = []
    for (lat, lon) in trajs:
        if len(lat) < 2:
            lat_rs = np.full(m, lat[0] if len(lat) else 0.0)
            lon_rs = np.full(m, lon[0] if len(lon) else 0.0)
        else:
            lat_rs, lon_rs = _resample_by_arclen(lat, lon, m)
        L.append(lat_rs)
        G.append(lon_rs)
    Lm = np.median(np.stack(L, axis=0), axis=0)
    Gm = np.median(np.stack(G, axis=0), axis=0)
    return Lm.astype(float), Gm.astype(float)


def _bandwidth_profile(trajs: List[Tuple[np.ndarray, np.ndarray]], cl_lat: np.ndarray, cl_lon: np.ndarray, q: float) -> Tuple[np.ndarray, Dict[str, float]]:
    # 计算每个索引位置的横向距离分位数（近似：按索引对齐），以及分位曲线的聚合统计
    m = len(cl_lat)
    if not trajs or m == 0:
        return np.zeros(m, dtype=float), {"bw_mean": 0.0, "bw_p75": 0.0}
    dists = []  # [n_traj, m]
    for (lat, lon) in trajs:
        if len(lat) < 2:
            lat_rs = np.full(m, lat[0] if len(lat) else 0.0)
            lon_rs = np.full(m, lon[0] if len(lon) else 0.0)
        else:
            lat_rs, lon_rs = _resample_by_arclen(lat, lon, m)
        dd = np.array([gc_distance_nm(float(lat_rs[k]), float(lon_rs[k]), float(cl_lat[k]), float(cl_lon[k])) * 1852.0 for k in range(m)], dtype=float)
        dists.append(dd)
    M = np.stack(dists, axis=0)  # [N,m]
    bw = np.quantile(M, q=float(q), axis=0)  # [m]
    stats = {
        "bw_mean": float(np.mean(bw)),
        "bw_p75": float(np.quantile(bw, 0.75)),
    }
    return bw.astype(float), stats


def build_centerlines(ym: str, band_quantile: float = 0.75, out_dir: Optional[str] = None, dry_run: bool = False, min_cluster_size: int = 30) -> str:
    """从聚类结果生成每个主簇的中心线与带宽。

    - 方法：DBA-approx（按弧长重采样后的逐点中位数）；带宽为逐点到中心线的距离分位（默认 p{band_quantile}）。
    - 输入：data_processed/ais/embeddings_<YM>.parquet、cluster_assign_<YM>.parquet、tracks_<YM>.parquet
    - 输出：reports/phaseE/center/prior_centerlines_<YM>.geojson（可通过 out_dir 覆盖目录）
    返回：输出 GeoJSON 的绝对路径（str）
    """
    emb_path = OUT_DIR / f"embeddings_{ym}.parquet"
    clus_path = OUT_DIR / f"cluster_assign_{ym}.parquet"
    trk_path = OUT_DIR / f"tracks_{ym}.parquet"
    if not (emb_path.exists() and clus_path.exists() and trk_path.exists()):
        raise FileNotFoundError("缺少 embeddings/cluster_assign/tracks 任一输入")

    df_emb = _to_pandas(_read_parquet_any(emb_path))
    df_clus = _to_pandas(_read_parquet_any(clus_path))
    df_trk = _to_pandas(_read_parquet_any(trk_path)).sort_values(["segment_id", "ts"]).copy()

    # 只保留在 cluster_assign 中出现的段
    df = df_emb.merge(df_clus, on="segment_id", how="inner")
    # 聚合轨迹
    seg_groups: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for seg_id, g in df_trk.groupby("segment_id"):
        lat = g["lat"].to_numpy(dtype=np.float64, copy=True)
        lon = g["lon"].to_numpy(dtype=np.float64, copy=True)
        seg_groups[str(seg_id)] = (lat, lon)

    # 收集簇→轨迹
    clusters: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for _, r in df.iterrows():
        cid = int(r.get("cluster_id", -1))
        if cid < 0:
            continue
        sid = str(r["segment_id"])
        if sid in seg_groups:
            clusters.setdefault(cid, []).append(seg_groups[sid])

    features: List[Dict[str, Any]] = []
    # 每簇日志
    for cid, trajs in clusters.items():
        n_tracks = len(trajs)
        if n_tracks < int(min_cluster_size):
            continue
        # 计算中心线（DBA-approx）
        cl_lat, cl_lon = _cluster_centerline_latlon(trajs, m=100)
        # 带宽曲线与统计
        bw, bw_stats = _bandwidth_profile(trajs, cl_lat, cl_lon, q=float(band_quantile))
        coords = [[float(lon), float(lat)] for lat, lon in zip(cl_lat.tolist(), cl_lon.tolist())]
        feat = {
            "type": "Feature",
            "properties": {
                "cluster_id": int(cid),
                "ym": ym,
                "n_tracks": int(n_tracks),
                "method": "DBA-approx",
                "bw_q": float(band_quantile),
                "bw_mean": float(bw_stats.get("bw_mean", 0.0)),
                "bw_p75": float(bw_stats.get("bw_p75", 0.0)),
            },
            "geometry": {"type": "LineString", "coordinates": coords},
        }
        features.append(feat)
        # 日志：每簇覆盖与带宽
        print(json.dumps({
            "ym": ym,
            "cluster_id": int(cid),
            "n_tracks": int(n_tracks),
            "bw_q": float(band_quantile),
            "bw_mean": float(bw_stats.get("bw_mean", 0.0)),
            "bw_p75": float(bw_stats.get("bw_p75", 0.0)),
        }, ensure_ascii=False))

    # 输出目录
    out_base = Path(out_dir) if out_dir else (REPORT_DIR / "center")
    out_base.mkdir(parents=True, exist_ok=True)
    out_path = out_base / f"prior_centerlines_{ym}.geojson"
    fc = {"type": "FeatureCollection", "features": features}
    if not dry_run:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(fc, f, ensure_ascii=False)
        try:
            register_artifact(run_id=ym, kind="prior_centerlines", path=str(out_path), attrs={"ym": ym, "clusters": len(features)})
        except Exception:
            pass

    # 总结日志
    print(json.dumps({"ym": ym, "clusters": len(features), "out": str(out_path)}, ensure_ascii=False))

    return str(out_path)


__all__ = ["CenterlineConfig", "build_centerlines"]

