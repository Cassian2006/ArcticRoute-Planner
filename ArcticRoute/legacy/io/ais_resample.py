"""AIS 段内等距重采样与派生特征（可选）

- resample_segment(df_seg, step_sec=60)：对单段（同一 segment_id）的轨迹做时间等距重采样，线性插值 lat/lon/sog/cog
- 派生特征：
  - dtheta（航向/航迹角变化量，采用 cog 作近似，单位度，[-180,180] wrap）
  - accel = dsog/dt（kn/s，按等距步长计算）
  - hour_sin/cos（小时正余弦编码，基于 UTC）
  - doy_sin/cos（一年内日序正余弦编码，基于 UTC）

约束：
- 优先 polars，回退 pandas
- 可用于批处理：resample_segments(parquet_in, parquet_out, step_sec=60, dry_run=True)
- 非 dry-run 写盘时调用 register_artifact(kind="ais_resampled")
- 路径使用 os.path.join
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List
from datetime import datetime, timezone

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from ArcticRoute.cache.index_util import register_artifact


def _angle_diff_deg(a2: float, a1: float) -> float:
    d = (float(a2) - float(a1) + 180.0) % 360.0 - 180.0
    return d


def _time_features(ts: int) -> Dict[str, float]:
    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
    hour = dt.hour + dt.minute/60.0 + dt.second/3600.0
    day_of_year = dt.timetuple().tm_yday
    hour_angle = 2*math.pi*hour/24.0
    doy_angle = 2*math.pi*day_of_year/365.0
    return {
        "hour_sin": math.sin(hour_angle),
        "hour_cos": math.cos(hour_angle),
        "doy_sin": math.sin(doy_angle),
        "doy_cos": math.cos(doy_angle),
    }


def _resample_segment_pandas(df_seg: "pd.DataFrame", step_sec: int) -> "pd.DataFrame":  # type: ignore[name-defined]
    import numpy as np
    if df_seg.empty:
        return df_seg
    df = df_seg.sort_values("ts").copy()
    t0, t1 = int(df["ts"].iloc[0]), int(df["ts"].iloc[-1])
    if t1 <= t0:
        return df
    ts_new = list(range(t0, t1+1, int(step_sec)))
    idx = pd.Index(ts_new, name="ts")
    # 设置 ts 为索引
    df = df.set_index("ts")
    # 插值列
    for col in ("lat", "lon", "sog", "cog"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.reindex(df.index.union(idx))
    df["lat"] = df["lat"].interpolate(method="time") if df.index.inferred_type == "datetime64" else df["lat"].interpolate()
    df["lon"] = df["lon"].interpolate()  # 简化：经纬用线性插值
    df["sog"] = df["sog"].interpolate()
    df["cog"] = df["cog"].interpolate()
    # 仅保留等距索引
    df = df.loc[idx]
    df = df.reset_index()
    # dtheta / accel
    df["dtheta"] = df["cog"].diff().fillna(0.0).apply(lambda x: _angle_diff_deg(df["cog"].iloc[df.index[df.index.get_loc(df.index[0])]] if False else 0.0, 0.0))  # 占位无用，仅避免 mypy
    # 正确计算角度差
    diffs = [0.0]
    for i in range(1, len(df)):
        diffs.append(_angle_diff_deg(df.loc[i, "cog"], df.loc[i-1, "cog"]))
    df["dtheta"] = diffs
    df["accel"] = (df["sog"].diff() / float(step_sec)).fillna(0.0)
    # 时间特征
    feats = [ _time_features(int(ts)) for ts in df["ts"].tolist() ]
    for k in ("hour_sin", "hour_cos", "doy_sin", "doy_cos"):
        df[k] = [f[k] for f in feats]
    return df


def _resample_segment_polars(df_seg: "pl.DataFrame", step_sec: int) -> "pl.DataFrame":  # type: ignore[name-defined]
    # 退化：先转 pandas 处理，再回到 polars（保持实现简洁且正确）
    pdf = df_seg.to_pandas()  # type: ignore[no-untyped-call]
    out = _resample_segment_pandas(pdf, step_sec)
    return pl.from_pandas(out)  # type: ignore[arg-type]


def resample_segment(df_seg: Any, step_sec: int = 60) -> Any:
    """对单段进行等距重采样与特征派生。
    要求 df_seg 至少包含列：ts, lat, lon, sog, cog；若包含 segment_id 将被保留。
    返回与输入同引擎的 DataFrame。
    """
    if pl is not None and isinstance(df_seg, pl.DataFrame):  # type: ignore[attr-defined]
        return _resample_segment_polars(df_seg, step_sec)
    return _resample_segment_pandas(df_seg, step_sec)


def resample_segments(parquet_in: str, parquet_out: str, step_sec: int = 60, dry_run: bool = True) -> Dict[str, Any]:
    """对包含多段的 parquet 数据按段重采样并输出。
    需要存在 segment_id 列（可由 B-06 生成）。
    """
    if pl is not None:
        df = pl.read_parquet(parquet_in)
        if "segment_id" not in df.columns:
            raise ValueError("segment_id 列缺失，请先运行分段处理 (B-06)")
        out_parts: List[pl.DataFrame] = []  # type: ignore[type-arg]
        for seg_id, g in df.group_by("segment_id"):
            out_parts.append(resample_segment(g, step_sec))  # type: ignore[arg-type]
        out_df = pl.concat(out_parts) if out_parts else df.head(0)
    else:
        dpf = pd.read_parquet(parquet_in)  # type: ignore[call-arg]
        if "segment_id" not in dpf.columns:
            raise ValueError("segment_id 列缺失，请先运行分段处理 (B-06)")
        out_parts = []
        for seg_id, g in dpf.groupby("segment_id"):
            out_parts.append(resample_segment(g, step_sec))
        out_df = pd.concat(out_parts, ignore_index=True) if out_parts else dpf.head(0)

    if not dry_run:
        os.makedirs(os.path.dirname(parquet_out), exist_ok=True)
        if pl is not None and isinstance(out_df, pl.DataFrame):  # type: ignore[attr-defined]
            out_df.write_parquet(parquet_out)
        else:
            out_df.to_parquet(parquet_out)  # type: ignore[call-arg]
        try:
            register_artifact(run_id=os.environ.get("RUN_ID", ""), kind="ais_resampled", path=parquet_out, attrs={"step_sec": int(step_sec)})
        except Exception:
            pass
    return {"out": parquet_out, "rows": int(len(out_df))}


__all__ = ["resample_segment", "resample_segments"]

