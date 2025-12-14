from __future__ import annotations

"""
Phase F｜编队识别（完整版最小实现）

- detect_convoy_episodes(ym) -> List[Episode-like dict]
- 数据：ArcticRoute/data_processed/ais/tracks_<ym>.parquet（必要）
- 方法：将 lon/lat 投影为 EPSG:3413 米坐标；构造 [x, y, t] 特征（时间做尺度归一）
  用 HDBSCAN 聚类，按簇内成员与时间顺序抽取连续轨迹形成 episodes。
- 输出：每个 episode 含 mmsi_a/mmsi_b/times/lats/lons，供 KDE 走廊使用。

注意：
- 列名自适应：time/timestamp/datetime, mmsi/shipid/id, lon/longitude/x, lat/latitude/y, sog/speed 可选。
- 若依赖缺失或文件/列不可用，将返回空集。
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import os
import math

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    from pyproj import Transformer  # type: ignore
except Exception:  # pragma: no cover
    Transformer = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover
    hdbscan = None  # type: ignore

AIS_DIR = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "ais")


@dataclass
class Episode:
    mmsi_a: int
    mmsi_b: int
    times: List[pd.Timestamp]
    lats: List[float]
    lons: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mmsi_a": int(self.mmsi_a),
            "mmsi_b": int(self.mmsi_b),
            "times": [str(t) for t in self.times],
            "lats": list(self.lats),
            "lons": list(self.lons),
        }


def _normalize_cols(df: "pd.DataFrame") -> "pd.DataFrame":
    cols = {c.lower(): c for c in df.columns}
    lon = cols.get("lon") or cols.get("longitude") or cols.get("x")
    lat = cols.get("lat") or cols.get("latitude") or cols.get("y")
    time = cols.get("time") or cols.get("timestamp") or cols.get("datetime")
    mmsi = cols.get("mmsi") or cols.get("shipid") or cols.get("id")
    sog = cols.get("sog") or cols.get("speed")
    if not (lon and lat and time and mmsi):
        return df.iloc[0:0].copy()
    out = df[[lon, lat, time, mmsi] + ([sog] if sog else [])].copy()
    out.columns = ["lon", "lat", "time", "mmsi"] + (["sog"] if sog else [])
    try:
        out["time"] = pd.to_datetime(out["time"])  # type: ignore
    except Exception:
        return df.iloc[0:0].copy()
    # 基本清洗
    out = out[np.isfinite(out["lon"]) & np.isfinite(out["lat"])].copy()
    return out


def _load_tracks(ym: str) -> "pd.DataFrame":
    if pd is None:
        return None  # type: ignore
    path = os.path.join(AIS_DIR, f"tracks_{ym}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
    return _normalize_cols(df)


def _project_ll_to_xy(df: "pd.DataFrame") -> Tuple[np.ndarray, np.ndarray]:
    if Transformer is None:
        raise RuntimeError("pyproj is required for convoy detection")
    transformer = Transformer.from_crs(4326, 3413, always_xy=True)
    x, y = transformer.transform(df["lon"].values, df["lat"].values)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _hdbscan_convoy(df: "pd.DataFrame") -> List[Episode]:
    # 时间缩放：将秒尺度拉平到空间尺度（让 10min ~ 5km 的权重近似）
    # 估算：300s 对应 3km => scale_t = 10 m/s
    tsec = (df["time"].astype("int64") // 10**9).values.astype(float)
    if np.std(tsec) < 1e-6:
        t_scaled = np.zeros_like(tsec)
    else:
        t_scaled = tsec * 10.0  # 10 m/s
    x, y = _project_ll_to_xy(df)
    feats = np.stack([x, y, t_scaled], axis=1)

    if hdbscan is None:
        return []
    # 经验参数：海上稀疏 → 最小簇 5，core 距离 5km
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3, cluster_selection_epsilon=5000.0, metric="euclidean")
    labels = clusterer.fit_predict(feats)

    episodes: List[Episode] = []
    for lab in sorted(set(labels)):
        if lab < 0:
            continue
        mask = labels == lab
        sub = df.loc[mask].sort_values("time")
        # 提取该簇内的配对（近邻配对聚合）
        # 简化：按时间分片，每个片内两两近邻，聚合为一个 episode
        rows = sub[["mmsi", "time", "lat", "lon"]].values.tolist()
        if len(rows) < 2:
            continue
        # 使用第一个与最后一个观测构造 episode 路径
        mmsi_vals = list({int(r[0]) for r in rows})
        m1 = mmsi_vals[0]
        m2 = mmsi_vals[1] if len(mmsi_vals) > 1 else mmsi_vals[0]
        times = [r[1] for r in rows]
        lats = [float(r[2]) for r in rows]
        lons = [float(r[3]) for r in rows]
        ep = Episode(mmsi_a=int(m1), mmsi_b=int(m2), times=list(times), lats=lats, lons=lons)
        episodes.append(ep)
    return episodes


def detect_convoy_episodes(ym: str) -> List[Dict[str, Any]]:
    if pd is None or np is None:
        return []
    df = _load_tracks(ym)
    if df is None or df.empty:
        return []
    # 过滤低速噪声
    if "sog" in df.columns:
        df = df[df["sog"].fillna(0.0) > 2.0]
    if df.empty:
        return []
    episodes = _hdbscan_convoy(df)
    return [e.to_dict() for e in episodes]


__all__ = ["detect_convoy_episodes"]
