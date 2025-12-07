"""AIS 轨迹分段（按时间缺口/速度中断）

- segment_tracks(parquet_in, parquet_out, gap_sec=1800, dry_run=True)
- 按 MMSI + 时间升序，遇到：
  - 时间缺口 >= gap_sec，或
  - 连续 teleport_flag（若存在该列）
  则切段，生成 segment_id 与 seg_idx

DOD：输出 tracks_YYYYMM.parquet（段级行），并 register_artifact(kind="ais_tracks")
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from ArcticRoute.cache.index_util import register_artifact


def _ensure_engine() -> str:
    if pl is not None:
        return "polars"
    if pd is not None:
        return "pandas"
    raise RuntimeError("No dataframe engine available (polars/pandas)")


def _read_parquet(path: str):
    if pl is not None:
        return pl.read_parquet(path)
    return pd.read_parquet(path)  # type: ignore


def _write_parquet(df_any: Any, out_path: str) -> None:
    if pl and isinstance(df_any, pl.DataFrame):  # type: ignore[attr-defined]
        df_any.write_parquet(out_path)
        return
    if pd and isinstance(df_any, pd.DataFrame):  # type: ignore[attr-defined]
        df_any.to_parquet(out_path)
        return
    raise RuntimeError("Unsupported dataframe type for writing")


def segment_tracks(parquet_in: str, parquet_out: str, gap_sec: int = 1800, dry_run: bool = True) -> Dict[str, Any]:
    eng = _ensure_engine()
    df = _read_parquet(parquet_in)

    if pl and eng == "polars" and isinstance(df, pl.DataFrame):  # type: ignore[attr-defined]
        df2 = df.sort(["mmsi", "ts"]).with_columns([
            pl.col("ts").shift(1).over("mmsi").alias("ts_prev"),
        ])
        df2 = df2.with_columns([
            (pl.col("ts") - pl.col("ts_prev")).alias("dt_s"),
        ])
        has_tp = "teleport_flag" in df2.columns
        cut = (pl.col("dt_s") >= int(gap_sec))
        if has_tp:
            cut = cut | (pl.col("teleport_flag") == True)  # noqa: E712
        df2 = df2.with_columns([
            cut.alias("cut_here")
        ])
        # 累加段号
        df2 = df2.with_columns([
            pl.when(pl.col("cut_here")).then(1).otherwise(0).cum_sum().over("mmsi").alias("seg_no")
        ])
        # 段内序号
        df2 = df2.with_columns([
            pl.int_range(0, pl.len()).over(["mmsi", "seg_no"]).alias("seg_idx")  # type: ignore[arg-type]
        ])
        # 段ID：mmsi_yyyymmdd_seq（从段首 ts 派生 yyyymmdd）
        dt0 = (pl.col("ts").shift(-1).over(["mmsi", "seg_no"]))  # 近似段首，简化实现
        # 简化：用 ts 格式化为 yyyymmdd
        def _fmt_date(ts: int) -> str:
            return datetime.utcfromtimestamp(int(ts)).strftime("%Y%m%d")
        df2 = df2.with_columns([
            pl.map_elements([pl.col("mmsi"), pl.col("ts")], lambda m, t: f"{int(m)}_{_fmt_date(int(t))}")
            .over(["mmsi", "seg_no"]).alias("seg_prefix")
        ])
        df2 = df2.with_columns([
            (pl.col("seg_prefix") + "_" + (pl.col("seg_no") + 1).cast(pl.Utf8)).alias("segment_id")
        ])
        out_df = df2.drop(["ts_prev", "dt_s", "cut_here", "seg_prefix"])  # 清爽输出

    else:
        # pandas 实现
        import numpy as np
        df2 = df.sort_values(["mmsi", "ts"]).copy()
        df2["ts_prev"] = df2.groupby("mmsi")["ts"].shift(1)
        df2["dt_s"] = (df2["ts"] - df2["ts_prev"]).astype("float64")
        cut = df2["dt_s"] >= int(gap_sec)
        if "teleport_flag" in df2.columns:
            cut = cut | df2["teleport_flag"].astype(bool)
        df2["seg_no"] = cut.astype(int)
        df2["seg_no"] = df2.groupby("mmsi")["seg_no"].cumsum()
        df2["seg_idx"] = df2.groupby(["mmsi", "seg_no"]).cumcount()
        def _fmt_date(ts: int) -> str:
            return datetime.utcfromtimestamp(int(ts)).strftime("%Y%m%d")
        df2["segment_id"] = df2.apply(lambda r: f"{int(r['mmsi'])}_{_fmt_date(int(r['ts']))}_{int(r['seg_no'])+1}", axis=1)
        out_df = df2.drop(columns=["ts_prev", "dt_s"])  # type: ignore

    if not dry_run:
        os.makedirs(os.path.dirname(parquet_out), exist_ok=True)
        _write_parquet(out_df, parquet_out)
        try:
            register_artifact(run_id=os.environ.get("RUN_ID", ""), kind="ais_tracks", path=parquet_out, attrs={"gap_sec": int(gap_sec)})
        except Exception:
            pass
    return {"out": parquet_out, "rows": int(len(out_df))}


__all__ = ["segment_tracks"]

