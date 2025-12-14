"""AIS density quick debug pipeline (JSON -> normalize -> clean -> segment -> timebin -> grid -> aggregate)

用极小样本（100–1000 行）在终端快速自检，定位“全 0”的原因。

打印：
- 每一步 5 行样例（字段：ts, lat, lon, sog, cog, iy, ix, time_bin, time_bin_idx）
- 每一步计数与丢弃比例
- 关键检查：键名映射、经度范围[-180,180)、网格索引越界、时间分桶落在所选月份

运行入口由 CLI: features.debug 调用。
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None  # type: ignore

from ArcticRoute.io.ais_norm import normalize_record, load_keymap
from ArcticRoute.io.timebin import make_time_bins
from ArcticRoute.io.grid_index import _load_rect_grid  # reuse grid
from ArcticRoute.io.ais_ingest import _iter_raw_records  # type: ignore

# ---- helpers ----

def _wrap_lon180(lon: float) -> float:
    # wrap to [-180, 180)
    if lon is None or (isinstance(lon, float) and (np.isnan(lon))):
        return lon
    x = float(lon)
    x = ((x + 180.0) % 360.0) - 180.0
    # 将 +180.0 正好映射到 -180.0（保持半开区间）
    if x >= 180.0:
        x -= 360.0
    return x


def _haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R_nm = 3440.065
    rlat1 = np.radians(lat1)
    rlon1 = np.radians(lon1)
    rlat2 = np.radians(lat2)
    rlon2 = np.radians(lon2)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = np.sin(dlat/2)**2 + np.cos(rlat1)*np.cos(rlat2)*np.sin(dlon/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return float(R_nm * c)


def _print_head(tag: str, rows: List[Dict[str, Any]], bins: Optional[np.ndarray] = None) -> None:
    print(f"\n[{tag}] 样例（最多5行）：")
    for i, r in enumerate(rows[:5]):
        tbi = r.get("time_bin_idx")
        tb = None
        if bins is not None and isinstance(tbi, (int, np.integer)) and tbi is not None and tbi >= 0 and tbi < len(bins) - 1:
            try:
                tb = int(bins[int(tbi)])
            except Exception:
                tb = None
        sample = {
            "ts": r.get("ts"),
            "lat": r.get("lat"),
            "lon": r.get("lon"),
            "sog": r.get("sog"),
            "cog": r.get("cog"),
            "iy": r.get("iy"),
            "ix": r.get("ix"),
            "time_bin_idx": tbi,
            "time_bin": (None if tb is None else str(np.datetime64(tb, 's'))),
        }
        print(json.dumps(sample, ensure_ascii=False))


def _count_and_report(stage: str, before: int, after: int, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    dropped = max(0, before - after)
    pct = (dropped / before * 100.0) if before > 0 else 0.0
    payload = {"stage": stage, "before": before, "after": after, "dropped": dropped, "drop_pct": round(pct, 2)}
    if extra:
        payload.update(extra)
    print(json.dumps({"count": payload}, ensure_ascii=False))
    return payload


# ---- core ----

def run_debug(src_dir: str, month: str, limit: int = 1000, step: str = "6H", speed_max: float = 35.0) -> Dict[str, Any]:
    # 1) 读取少量原始 JSON
    files: List[str] = []
    for dp, _, fns in os.walk(src_dir):
        for fn in fns:
            if fn.lower().endswith(".json"):
                files.append(os.path.join(dp, fn))
    files.sort()
    raw_cnt = 0
    keymap = load_keymap(default_root=os.path.join(os.getcwd(), "reports", "recon"))

    # 2) normalize
    norm_rows: List[Dict[str, Any]] = []
    map_miss = 0
    for fp in files:
        for raw in _iter_raw_records(fp):
            if raw_cnt >= limit:
                break
            raw_cnt += 1
            norm = normalize_record(raw, keymap)
            if norm is None:
                map_miss += 1
                continue
            # 类型收敛
            try:
                norm["ts"] = int(norm["ts"])  # epoch seconds
                norm["lat"] = float(norm["lat"])  # in [-90,90]
                norm["lon"] = float(norm["lon"])  # expected in [-180,180]
                if "sog" in norm:
                    norm["sog"] = float(norm["sog"])
                if "cog" in norm:
                    norm["cog"] = float(norm["cog"])
            except Exception:
                map_miss += 1
                continue
            norm_rows.append(norm)
        if raw_cnt >= limit:
            break

    _print_head("normalize", norm_rows)
    cnt_norm = _count_and_report("normalize", raw_cnt, len(norm_rows), {"map_miss": map_miss})

    if not norm_rows:
        return {
            "stage": "normalize",
            "raw_cnt": raw_cnt,
            "kept": 0,
            "diagnosis": "键名映射未命中或字段类型非法（lat/lon/sog/cog 非数值）",
        }

    # 2.5) 经度 wrap 检查
    lon_changed = 0
    for r in norm_rows:
        lon0 = r.get("lon")
        lon1 = _wrap_lon180(lon0)
        if lon0 != lon1:
            lon_changed += 1
            r["lon"] = lon1
    if lon_changed:
        print(json.dumps({"lon_wrap": {"changed": lon_changed, "total": len(norm_rows)}}, ensure_ascii=False))

    # 3) 清洗：速度阈值 + 去重
    # 去重键：mmsi + ts
    seen = set()
    cleaned: List[Dict[str, Any]] = []
    speed_drop = 0
    dup_drop = 0
    for r in norm_rows:
        k = (int(r.get("mmsi", 0)), int(r.get("ts", 0)))
        if k in seen:
            dup_drop += 1
            continue
        seen.add(k)
        sog = r.get("sog")
        if isinstance(sog, (int, float)):
            try:
                if not (0.0 <= float(sog) <= float(speed_max)):
                    speed_drop += 1
                    continue
            except Exception:
                pass
        cleaned.append(r)

    _print_head("clean", cleaned)
    _ = _count_and_report("clean", len(norm_rows), len(cleaned), {"speed_drop": speed_drop, "dup_drop": dup_drop})

    if not cleaned:
        return {
            "stage": "clean",
            "raw_cnt": raw_cnt,
            "kept": 0,
            "diagnosis": "清洗后为空（速度越界或去重导致）",
        }

    # 4) 分段（按时间缺口）——仅标注，不强依赖
    cleaned.sort(key=lambda x: (int(x.get("mmsi", 0)), int(x.get("ts", 0))))
    GAP = 1800  # 30min
    seg_no_by_mmsi: Dict[int, int] = {}
    last_ts_by_mmsi: Dict[int, int] = {}
    for r in cleaned:
        m = int(r.get("mmsi", 0))
        t = int(r.get("ts", 0))
        last = last_ts_by_mmsi.get(m)
        cut = (last is not None) and (t - int(last) >= GAP)
        if m not in seg_no_by_mmsi:
            seg_no_by_mmsi[m] = 0
        if cut:
            seg_no_by_mmsi[m] += 1
        r["segment_no"] = seg_no_by_mmsi[m]
        last_ts_by_mmsi[m] = t

    _print_head("segment", cleaned)
    _ = _count_and_report("segment", len(cleaned), len(cleaned), None)

    # 5) 时间分桶
    bins = make_time_bins(str(month), step=str(step))
    ts_vals = np.array([int(r["ts"]) for r in cleaned], dtype=np.int64)
    bin_idx = np.searchsorted(bins, ts_vals, side="right") - 1
    oob_mask = (bin_idx < 0) | (bin_idx >= len(bins) - 1)
    time_kept: List[Dict[str, Any]] = []
    oob_time = int(oob_mask.sum())
    for r, bi in zip(cleaned, bin_idx):
        if bi < 0 or bi >= len(bins) - 1:
            continue
        r2 = dict(r)
        r2["time_bin_idx"] = int(bi)
        time_kept.append(r2)

    _print_head("timebin", time_kept, bins)
    _ = _count_and_report("timebin", len(cleaned), len(time_kept), {"oob_time": oob_time, "bins": int(len(bins))})

    if not time_kept:
        return {
            "stage": "timebin",
            "raw_cnt": raw_cnt,
            "kept": 0,
            "diagnosis": "时间不在所选月份或 step 导致全部越界",
        }

    # 6) 网格索引
    try:
        lat1d, lon1d = _load_rect_grid(None)
        Ny, Nx = int(lat1d.shape[0]), int(lon1d.shape[0])
    except Exception as e:
        return {"stage": "grid", "error": f"加载网格失败: {e}"}

    def _nearest_index(axis_vals: np.ndarray, values: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(axis_vals, values, side='left')
        idx_right = np.clip(idx, 0, len(axis_vals) - 1)
        idx_left = np.clip(idx - 1, 0, len(axis_vals) - 1)
        dist_right = np.abs(axis_vals[idx_right] - values)
        dist_left = np.abs(axis_vals[idx_left] - values)
        choose_left = dist_left <= dist_right
        out = np.where(choose_left, idx_left, idx_right)
        return np.clip(out, 0, len(axis_vals) - 1)

    lat_arr = np.array([float(r["lat"]) for r in time_kept], dtype=float)
    lon_arr = np.array([float(r["lon"]) for r in time_kept], dtype=float)
    iy = _nearest_index(lat1d, lat_arr)
    ix = _nearest_index(lon1d, lon_arr)

    grid_kept: List[Dict[str, Any]] = []
    oob_nan = int(np.isnan(lat_arr).sum() + np.isnan(lon_arr).sum())
    for r, iyy, ixx in zip(time_kept, iy, ix):
        r2 = dict(r)
        r2["iy"] = int(iyy)
        r2["ix"] = int(ixx)
        grid_kept.append(r2)

    _print_head("grid", grid_kept, bins)
    _ = _count_and_report("grid", len(time_kept), len(grid_kept), {"iy_range": [int(np.min(iy)), int(np.max(iy))] if len(iy)>0 else None, "ix_range": [int(np.min(ix)), int(np.max(ix))] if len(ix)>0 else None, "Ny": Ny, "Nx": Nx, "oob_nan": oob_nan})

    if not grid_kept:
        return {
            "stage": "grid",
            "raw_cnt": raw_cnt,
            "kept": 0,
            "diagnosis": "网格索引为空（可能越界或坐标为空）",
        }

    # 7) 聚合（计数）
    counts: Dict[Tuple[int,int,int], int] = {}
    for r in grid_kept:
        key = (int(r["time_bin_idx"]), int(r["iy"]), int(r["ix"]))
        counts[key] = counts.get(key, 0) + 1

    nonzero = len(counts)
    total_rows = len(grid_kept)
    # Top-5 cells
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
    top_view = [
        {"time_bin_idx": k[0], "iy": k[1], "ix": k[2], "count": v} for k, v in top
    ]
    print(json.dumps({"aggregate": {"nonzero_cells": nonzero, "rows": total_rows, "top": top_view}}, ensure_ascii=False))

    # 诊断：若仍“全 0”
    diag = None
    if total_rows == 0 or nonzero == 0:
        if cnt_norm.get("after", 0) == 0:
            diag = "键名映射失败或字段非法"
        elif len(cleaned) == 0:
            diag = "清洗阶段全部被丢弃（速度阈值/去重）"
        elif len(time_kept) == 0:
            diag = "时间分桶全部越界（不在目标月份/step 错误）"
        elif len(grid_kept) == 0:
            diag = "网格索引越界或网格载入失败"
        else:
            diag = "聚合阶段异常"

    return {
        "month": str(month),
        "step": str(step),
        "raw_cnt": raw_cnt,
        "normalized": len(norm_rows),
        "cleaned": len(cleaned),
        "time_kept": len(time_kept),
        "grid_kept": len(grid_kept),
        "nonzero_cells": nonzero,
        "top": top_view,
        "diagnosis": diag,
    }


__all__ = ["run_debug"]

