"""AIS 反瞬移清洗（Anti‑Teleport）

- haversine_nm: 计算两个经纬度点之间的大圆距离（海里）
- clean_teleport: 基于相邻点隐含速度阈值剔除跳点或标记 teleport_flag

特性/约束：
- 支持 polars 优先，回退 pandas
- 输入可为 DataFrame 或 parquet 路径
- 剪裁规则：按 MMSI 分组、时间升序；两点间 distance_nm / (dt_hours) > speed_kn_max 则判定为 teleport
- mark_only=True 仅标记 teleport_flag；否则丢弃“后一点”
- 汇总计数：teleport_removed
- 写盘通过 write_cleaned(..., dry_run=True)，非 dry-run 时使用 register_artifact()
- 路径使用 os.path.join
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional, Tuple

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from ArcticRoute.cache.index_util import register_artifact


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """大圆距离（海里）。"""
    # 地球半径（海里）
    R_nm = 3440.065
    rlat1 = math.radians(float(lat1))
    rlon1 = math.radians(float(lon1))
    rlat2 = math.radians(float(lat2))
    rlon2 = math.radians(float(lon2))
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R_nm * c


def _ensure_engine() -> str:
    if pl is not None:
        return "polars"
    if pd is not None:
        return "pandas"
    raise RuntimeError("No dataframe engine available (polars/pandas)")


def _read_parquet(path: str):
    eng = _ensure_engine()
    if eng == "polars":
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


def _compute_flags_polars(df: "pl.DataFrame", speed_kn_max: float) -> "pl.DataFrame":  # type: ignore[name-defined]
    # 预期列：mmsi, ts, lat, lon
    df2 = df.sort(["mmsi", "ts"]).with_columns([
        pl.col("lat").shift(1).over("mmsi").alias("lat_prev"),
        pl.col("lon").shift(1).over("mmsi").alias("lon_prev"),
        pl.col("ts").shift(1).over("mmsi").alias("ts_prev"),
    ])
    # 计算距离/时间/速度
    def _haversine_expr(lat1, lon1, lat2, lon2):
        # 使用 Python UDF（小批量足够）；大规模可考虑向量化近似
        return pl.map_elements(
            [lat1, lon1, lat2, lon2],
            lambda l1, g1, l2, g2: None if (l1 is None or g1 is None or l2 is None or g2 is None) else haversine_nm(l1, g1, l2, g2),
            return_dtype=pl.Float64,
        )
    df2 = df2.with_columns([
        _haversine_expr(pl.col("lat_prev"), pl.col("lon_prev"), pl.col("lat"), pl.col("lon")).alias("dist_nm"),
        (pl.col("ts") - pl.col("ts_prev")).cast(pl.Float64).alias("dt_s"),
    ]).with_columns([
        (pl.col("dist_nm") / (pl.col("dt_s") / 3600.0)).alias("speed_kn")
    ])
    df2 = df2.with_columns([
        ((pl.col("dt_s") > 0) & (pl.col("speed_kn") > float(speed_kn_max))).alias("teleport_flag")
    ])
    return df2


def _compute_flags_pandas(df: "pd.DataFrame", speed_kn_max: float) -> "pd.DataFrame":  # type: ignore[name-defined]
    import numpy as np
    df2 = df.sort_values(["mmsi", "ts"]).copy()
    df2["lat_prev"] = df2.groupby("mmsi")["lat"].shift(1)
    df2["lon_prev"] = df2.groupby("mmsi")["lon"].shift(1)
    df2["ts_prev"] = df2.groupby("mmsi")["ts"].shift(1)
    # 距离
    def _hv(row):
        if any(pd.isna(row[k]) for k in ("lat_prev", "lon_prev", "lat", "lon")):
            return float("nan")
        return haversine_nm(row["lat_prev"], row["lon_prev"], row["lat"], row["lon"])  # type: ignore[arg-type]
    df2["dist_nm"] = df2.apply(_hv, axis=1)
    df2["dt_s"] = (df2["ts"] - df2["ts_prev"]).astype("float64")
    df2["speed_kn"] = df2["dist_nm"] / (df2["dt_s"] / 3600.0)
    df2["teleport_flag"] = (df2["dt_s"] > 0) & (df2["speed_kn"] > float(speed_kn_max))
    return df2


def clean_teleport(df_or_path: Any, speed_kn_max: float = 45.0, mark_only: bool = False):
    """标记/剔除瞬移点。

    返回 (df, stats)；stats: {"teleport_removed": int}
    - mark_only=True: 不剔除，仅添加 teleport_flag
    - mark_only=False: 丢弃 teleport_flag=True 的点（丢弃的是“后一点”）
    """
    eng = _ensure_engine()
    if isinstance(df_or_path, str):
        df = _read_parquet(df_or_path)
    else:
        df = df_or_path

    if eng == "polars" and isinstance(df, pl.DataFrame):  # type: ignore[attr-defined]
        df2 = _compute_flags_polars(df, speed_kn_max)
        if mark_only:
            return df2, {"teleport_removed": 0}
        removed = int(df2.select(pl.sum(pl.col("teleport_flag").cast(pl.Int64))).item())  # type: ignore[arg-type]
        cleaned = df2.filter(~pl.col("teleport_flag")).drop(["lat_prev", "lon_prev", "ts_prev", "dist_nm", "dt_s", "speed_kn"])  # type: ignore[arg-type]
        return cleaned, {"teleport_removed": removed}

    # pandas 路径
    df2 = _compute_flags_pandas(df, speed_kn_max)  # type: ignore[arg-type]
    if mark_only:
        return df2, {"teleport_removed": 0}
    removed = int(df2["teleport_flag"].sum())  # type: ignore[arg-type]
    cleaned = df2.loc[~df2["teleport_flag"]].drop(columns=["lat_prev", "lon_prev", "ts_prev", "dist_nm", "dt_s", "speed_kn"]).copy()
    return cleaned, {"teleport_removed": removed}


def write_cleaned(in_parquet: str, out_parquet: str, speed_kn_max: float = 45.0, dry_run: bool = True, mark_only: bool = False) -> Dict[str, Any]:
    df, stats = clean_teleport(in_parquet, speed_kn_max=speed_kn_max, mark_only=mark_only)
    if not dry_run:
        os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
        _write_parquet(df, out_parquet)
        try:
            register_artifact(run_id=os.environ.get("RUN_ID", ""), kind="ais_parquet_clean", path=out_parquet, attrs={"speed_kn_max": speed_kn_max, **stats})
        except Exception:
            pass
    return {"out": out_parquet, "stats": stats}


__all__ = ["haversine_nm", "clean_teleport", "write_cleaned"]

