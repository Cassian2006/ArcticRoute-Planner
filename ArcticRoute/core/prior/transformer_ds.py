from __future__ import annotations

# 附加：真实数据集与 collate（E-T07.5）

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np  # type: ignore

from ArcticRoute.cache.index_util import register_artifact

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

import torch
from torch.utils.data import Dataset

from ArcticRoute.core.prior.transformer_split import ContrastivePairSampler, SamplerConfig
from ArcticRoute.core.prior.transformer_feats import FeatureConfig, make_features

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data_processed" / "ais"


@dataclass
class PairDatasetConfig:
    ym: str
    split: str = "train"  # train|val
    max_len: int = 512
    sampler: SamplerConfig = field(default_factory=SamplerConfig)


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


class TrajPairDataset(Dataset):
    """基于段索引与 tracks 的正样本双视图对数据集。

    - 读取 segment_index_<ym>.parquet 与 tracks_<ym>.parquet
    - 使用 ContrastivePairSampler 生成两个裁剪窗口（按秒）
    - 返回两视图的原始时序字段（lat,lon,sog,cog,ts），不做投影/特征，交由 collate 完成
    """

    def __init__(self, split_index_json: Optional[Path], segment_index_path: Path, tracks_path: Path, cfg: PairDatasetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # 段索引
        seg_df_any = _read_parquet_any(segment_index_path)
        self.seg_df = _to_pandas(seg_df_any)
        # 读取切分（若提供），按 mmsi 过滤段
        if split_index_json and split_index_json.exists():
            import json as _json
            data = _json.loads(split_index_json.read_text(encoding="utf-8"))
            mset = set(data.get("train_mmsi") if cfg.split == "train" else data.get("val_mmsi"))
            if mset:
                self.seg_df = self.seg_df[self.seg_df["mmsi"].isin(mset)].copy()
        self.seg_df = self.seg_df.reset_index(drop=True)
        # tracks 按段聚合
        traj_any = _read_parquet_any(tracks_path)
        traj = _to_pandas(traj_any)
        # 确保排序
        traj = traj.sort_values(["segment_id", "ts"]).copy()
        groups = {}
        for seg_id, g in traj.groupby("segment_id"):
            groups[str(seg_id)] = {
                "ts": g["ts"].to_numpy(dtype=np.int64),
                "lat": g["lat"].to_numpy(dtype=np.float64, copy=True),
                "lon": g["lon"].to_numpy(dtype=np.float64, copy=True),
                "sog": g.get("sog", pd.Series([np.nan]*len(g))).to_numpy(dtype=np.float64, copy=True),
                "cog": g.get("cog", pd.Series([np.nan]*len(g))).to_numpy(dtype=np.float64, copy=True),
            }
        self.traj_by_seg: Dict[str, Dict[str, np.ndarray]] = groups
        # 采样器基于 seg_df（聚合粒度）
        self.sampler = ContrastivePairSampler(self.seg_df, cfg.sampler)
        self.N = len(self.seg_df)

    def __len__(self) -> int:
        return self.N

    def _crop_view(self, seg_id: str, ts0: int, offset: int, length: int) -> Dict[str, np.ndarray]:
        arrs = self.traj_by_seg.get(seg_id)
        if not arrs:
            return {"lat": np.array([]), "lon": np.array([]), "sog": np.array([]), "cog": np.array([]), "ts": np.array([])}
        ts = arrs["ts"]
        t_start = ts0 + int(offset)
        t_end = t_start + int(length)
        sel = (ts >= t_start) & (ts <= t_end)
        view = {k: arrs[k][sel] for k in ("lat","lon","sog","cog")}
        view["ts"] = ts[sel]
        return view

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.seg_df.iloc[idx]
        seg_id: str = str(r["segment_id"])
        ts_start = int(r["ts_start"])  # 段起点时间
        # 采样两个裁剪
        (sid1, (o1, L1)), (sid2, (o2, L2)) = self.sampler.sample_positive(idx)
        assert sid1 == seg_id and sid2 == seg_id
        v1 = self._crop_view(seg_id, ts_start, o1, L1)
        v2 = self._crop_view(seg_id, ts_start, o2, L2)
        return {"seg_id": seg_id, "view_i": v1, "view_j": v2, "mmsi": int(r["mmsi"]) }


def collate_pairs(batch: Sequence[Dict[str, Any]], seq_len: int, feat_cfg: Optional[FeatureConfig] = None) -> Dict[str, Any]:
    B = len(batch)
    T = int(seq_len)
    C = 9  # 通道数
    x_i = np.zeros((B, T, C), dtype=np.float32)
    x_j = np.zeros((B, T, C), dtype=np.float32)
    m_i = np.zeros((B, T), dtype=bool)
    m_j = np.zeros((B, T), dtype=bool)
    seg_ids: List[str] = []
    mmsi_list: List[int] = []

    for b, item in enumerate(batch):
        seg_ids.append(str(item.get("seg_id", "")))
        mmsi_list.append(int(item.get("mmsi", 0)))
        for view_key, X, M in (("view_i", x_i, m_i), ("view_j", x_j, m_j)):
            v = item[view_key]
            lat = v["lat"].astype(np.float64)
            lon = v["lon"].astype(np.float64)
            sog = v["sog"].astype(np.float64) if v.get("sog") is not None else None
            cog = v["cog"].astype(np.float64) if v.get("cog") is not None else None
            ts = v["ts"].astype(np.int64)
            feats = make_features(lat, lon, sog, cog, ts, feat_cfg)
            Xi = feats["x"]  # [t,C]
            Mi = feats["mask"]  # [t]
            # 截断/填充到 T
            tlen = min(len(Mi), T)
            if tlen > 0:
                X[b, :tlen, :] = Xi[:tlen, :]
                M[b, :tlen] = Mi[:tlen]

    out = {
        "x_i": torch.from_numpy(x_i),
        "mask_i": torch.from_numpy(m_i),
        "x_j": torch.from_numpy(x_j),
        "mask_j": torch.from_numpy(m_j),
        "meta": {"seg_ids": seg_ids, "mmsi": mmsi_list},
    }
    return out


def prepare_segments(ym: str, min_len_sec: int | None = None, max_gap_sec: int | None = None, grid: str | None = "1/60", dry_run: bool = False) -> None:
    """从 tracks_<ym>.parquet 生成 segment_index_<ym>.parquet 与 split_<ym>.json。

    - 读取：data_processed/ais/tracks_<ym>.parquet
    - 输出：data_processed/ais/segment_index_<ym>.parquet, data_processed/ais/split_<ym>.json
    - 过滤：若提供 min_len_sec，仅保留持续时间 >= min_len_sec 的段
    - 说明：max_gap_sec 在分段阶段已处理，这里仅透传到日志
    """
    from ArcticRoute.core.prior.transformer_split import make_splits  # 延迟导入以避免循环
    import json as _json

    ym_s = str(ym)
    aout = ROOT / "data_processed" / "ais"
    tracks_path = aout / f"tracks_{ym_s}.parquet"
    if not tracks_path.exists():
        raise FileNotFoundError(f"缺少 tracks 输入: {tracks_path}")

    # 读取 tracks 并按 segment_id 聚合
    df_any = _read_parquet_any(tracks_path)
    df = _to_pandas(df_any)
    # 必要列检查
    need_cols = {"segment_id", "mmsi", "ts", "lat", "lon"}
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise ValueError(f"tracks 缺少必要列: {miss}")

    # 聚合：每段时间范围与 bbox
    grp = df.sort_values(["segment_id", "ts"]).groupby("segment_id")
    import numpy as _np  # type: ignore
    seg_rows: List[Dict[str, Any]] = []
    for seg_id, g in grp:
        ts_s = int(_np.min(g["ts"].to_numpy(dtype=_np.int64)))
        ts_e = int(_np.max(g["ts"].to_numpy(dtype=_np.int64)))
        mmsi = int(g["mmsi"].iloc[0])
        lat_min = float(_np.min(g["lat"].to_numpy(dtype=_np.float64)))
        lat_max = float(_np.max(g["lat"].to_numpy(dtype=_np.float64)))
        lon_min = float(_np.min(g["lon"].to_numpy(dtype=_np.float64)))
        lon_max = float(_np.max(g["lon"].to_numpy(dtype=_np.float64)))
        vclass = g["vclass"].iloc[0] if "vclass" in g.columns else None
        dur = ts_e - ts_s
        if min_len_sec is not None and dur < int(min_len_sec):
            continue
        seg_rows.append({
            "segment_id": str(seg_id),
            "mmsi": int(mmsi),
            "ts_start": int(ts_s),
            "ts_end": int(ts_e),
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
            "bbox": [lon_min, lat_min, lon_max, lat_max],
            **({"vclass": None if vclass is None else str(vclass)}),
            "n_points": int(len(g)),
            "duration_s": int(dur),
        })

    # 写 segment_index
    seg_index_path = aout / f"segment_index_{ym_s}.parquet"
    split_path = aout / f"split_{ym_s}.json"

    plan = {
        "ym": ym_s,
        "tracks": str(tracks_path),
        "segment_index": str(seg_index_path),
        "split_json": str(split_path),
        "segments_after_filter": len(seg_rows),
        "min_len_sec": (None if min_len_sec is None else int(min_len_sec)),
        "max_gap_sec": (None if max_gap_sec is None else int(max_gap_sec)),
        "grid": (None if grid is None else str(grid)),
    }

    if dry_run:
        print(_json.dumps({"plan": "prepare_segments", **plan}, ensure_ascii=False))
        return

    # 写 parquet
    if pd is None:
        raise RuntimeError("pandas required to write segment_index parquet")
    import pandas as _pd  # type: ignore
    seg_df = _pd.DataFrame(seg_rows)
    seg_index_path.parent.mkdir(parents=True, exist_ok=True)
    seg_df.to_parquet(str(seg_index_path), engine="pyarrow")
    try:
        register_artifact(run_id=os.environ.get("RUN_ID", ""), kind="ais_segment_index", path=str(seg_index_path), attrs={"ym": ym_s})
    except Exception:
        pass

    # 生成 split 并写 json
    splits = make_splits(seg_df, seed=42, train_ratio=0.8, stratify="mmsi")
    with open(split_path, "w", encoding="utf-8") as f:
        _json.dump(splits, f, ensure_ascii=False, indent=2)
    try:
        register_artifact(run_id=os.environ.get("RUN_ID", ""), kind="prior_split", path=str(split_path), attrs={"ym": ym_s})
    except Exception:
        pass

    print(_json.dumps({"done": True, **plan}, ensure_ascii=False))


__all__ = ["TrajPairDataset", "PairDatasetConfig", "collate_pairs", "prepare_segments"]
