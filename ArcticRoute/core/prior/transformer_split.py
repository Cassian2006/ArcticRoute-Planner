from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import random

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

import math

from ArcticRoute.core.prior.geo import to_xy


def _ensure_engine() -> str:
    if pl is not None:
        return "polars"
    if pd is not None:
        return "pandas"
    raise RuntimeError("No dataframe engine available (polars/pandas)")


def _read_parquet(path: Path):
    if pl is not None:
        return pl.read_parquet(str(path))  # type: ignore
    return pd.read_parquet(str(path))  # type: ignore


def _to_pandas(df_any: Any) -> "pd.DataFrame":  # type: ignore
    if pd is None:
        raise RuntimeError("pandas required for split utilities")
    if pl is not None and isinstance(df_any, pl.DataFrame):  # type: ignore[attr-defined]
        return df_any.to_pandas()  # type: ignore
    if isinstance(df_any, pd.DataFrame):  # type: ignore[attr-defined]
        return df_any
    raise RuntimeError("Unsupported dataframe for conversion")


def make_splits(segment_index: Any, seed: int = 42, train_ratio: float = 0.8, stratify: str = "mmsi") -> Dict[str, Any]:
    """按 MMSI 分层切分 Train/Val，避免信息泄漏。

    - segment_index: DataFrame 或 parquet 路径
    - 返回：{"train_mmsi": [...], "val_mmsi": [...], "train_rows": N, "val_rows": M}
    """
    # 读取
    if isinstance(segment_index, (str, Path)):
        df_any = _read_parquet(Path(segment_index))
    else:
        df_any = segment_index
    dpf = _to_pandas(df_any)

    if stratify != "mmsi":
        raise NotImplementedError("当前仅支持按 mmsi 分层")

    # MMSI 去重并随机划分
    mmsi_list = sorted(set(int(x) for x in dpf["mmsi"].tolist()))
    rng = random.Random(int(seed))
    rng.shuffle(mmsi_list)
    cut = int(math.floor(len(mmsi_list) * float(train_ratio)))
    train_mmsi = set(mmsi_list[:cut])
    val_mmsi = set(mmsi_list[cut:])

    # 过滤行
    d_train = dpf[dpf["mmsi"].isin(train_mmsi)].copy()
    d_val = dpf[dpf["mmsi"].isin(val_mmsi)].copy()

    # 断言无交集
    assert train_mmsi.isdisjoint(val_mmsi), "Train/Val MMSI 存在交集"

    return {
        "train_mmsi": sorted(list(train_mmsi)),
        "val_mmsi": sorted(list(val_mmsi)),
        "train_rows": int(len(d_train)),
        "val_rows": int(len(d_val)),
        "seed": int(seed),
        "train_ratio": float(train_ratio),
    }


@dataclass
class SamplerConfig:
    time_window_s: int = 6 * 3600  # 近时：6小时
    space_window_m: float = 50_000.0  # 近域：50 km
    seed: int = 42


class ContrastivePairSampler:
    """基于段索引的对比采样器。

    - 正样本：同一段的不同裁剪（返回 (segment_id, crop1), (segment_id, crop2)）
    - 难负：同月、近时近域、不同 MMSI（返回 (anchor_seg, negative_seg)）
      近域在 EPSG:3413 投影下用欧氏距离近邻筛选。
    """

    def __init__(self, seg_index: Any, cfg: Optional[SamplerConfig] = None) -> None:
        if isinstance(seg_index, (str, Path)):
            df_any = _read_parquet(Path(seg_index))
        else:
            df_any = seg_index
        self.df = _to_pandas(df_any)
        self.cfg = cfg or SamplerConfig()
        self.rng = random.Random(int(self.cfg.seed))
        # 预计算中点与投影坐标
        # 需要 lat/lon 列：若 tracks 阶段未保留，可使用 bbox 中心近似
        def _mid_latlon(row) -> Tuple[float, float]:
            if "lat" in row and "lon" in row:
                return float(row["lat"]), float(row["lon"])
            # 使用 bbox: [lon_min, lat_min, lon_max, lat_max]
            b = row.get("bbox")
            if isinstance(b, (list, tuple)) and len(b) == 4:
                lon_c = (float(b[0]) + float(b[2])) / 2.0
                lat_c = (float(b[1]) + float(b[3])) / 2.0
                return lat_c, lon_c
            # 退化：估算
            return float(row.get("lat_min", 0.0)) + 0.0, float(row.get("lon_min", 0.0)) + 0.0
        mids: List[Tuple[float, float]] = []
        for _, r in self.df.iterrows():
            lat_c, lon_c = _mid_latlon(r)
            mids.append((lat_c, lon_c))
        xs: List[float] = []
        ys: List[float] = []
        for lat_c, lon_c in mids:
            x, y = to_xy(lat_c, lon_c)
            xs.append(x)
            ys.append(y)
        self.df["x"] = xs
        self.df["y"] = ys
        self.df["ts_mid"] = (self.df["ts_start"].astype("int64") + self.df["ts_end"].astype("int64")) // 2

        # 简易近邻索引：按网格哈希加速（避免 sklearn 依赖）
        cell = max(1.0, self.cfg.space_window_m / 2.0)
        self.grid: Dict[Tuple[int, int], List[int]] = {}
        for idx, r in self.df.iterrows():
            cx = int(math.floor(r["x"] / cell))
            cy = int(math.floor(r["y"] / cell))
            self.grid.setdefault((cx, cy), []).append(idx)

    def _neighbors(self, r: "pd.Series") -> List[int]:  # type: ignore
        # 在 3x3 邻域网格内搜集候选，再按时间与距离窗口筛选
        cell = max(1.0, self.cfg.space_window_m / 2.0)
        cx = int(math.floor(r["x"] / cell))
        cy = int(math.floor(r["y"] / cell))
        cands: List[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cands.extend(self.grid.get((cx + dx, cy + dy), []))
        out: List[int] = []
        for j in cands:
            if j == r.name:
                continue
            rj = self.df.loc[j]
            if int(rj["mmsi"]) == int(r["mmsi"]):
                continue
            if abs(int(rj["ts_mid"]) - int(r["ts_mid"])) > int(self.cfg.time_window_s):
                continue
            dx = float(rj["x"]) - float(r["x"])
            dy = float(rj["y"]) - float(r["y"])
            if math.hypot(dx, dy) <= float(self.cfg.space_window_m):
                out.append(j)
        return out

    def sample_positive(self, i: int) -> Tuple[Tuple[str, Tuple[int, int]], Tuple[str, Tuple[int, int]]]:
        """从同一段生成两个不同裁剪的正样本。
        返回：((segment_id, (offset1, length1)), (segment_id, (offset2, length2)))
        注意：这里只返回元信息，真正的裁剪在上层按 tracks 实现。
        """
        r = self.df.iloc[i]
        seg_id = str(r["segment_id"]) 
        dur = int(r["ts_end"]) - int(r["ts_start"]) 
        # 简化：裁剪长度为总长的 30%~60%
        L1 = max(1, int(dur * (0.3 + 0.2 * self.rng.random())))
        L2 = max(1, int(dur * (0.3 + 0.2 * self.rng.random())))
        o1 = int(self.rng.randint(0, max(0, dur - L1)))
        o2 = int(self.rng.randint(0, max(0, dur - L2)))
        return (seg_id, (o1, L1)), (seg_id, (o2, L2))

    def sample_hard_negative(self, i: int) -> Optional[Tuple[str, str]]:
        """为段 i 采样一个难负段（不同 MMSI，近时近域）。返回 (anchor_seg_id, neg_seg_id)。无则返回 None。"""
        r = self.df.iloc[i]
        neigh = self._neighbors(r)
        if not neigh:
            return None
        j = self.rng.choice(neigh)
        return (str(r["segment_id"]), str(self.df.loc[j, "segment_id"]))


__all__ = ["make_splits", "ContrastivePairSampler", "SamplerConfig"]














