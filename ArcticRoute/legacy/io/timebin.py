"""AIS 时间分桶（binning）策略与对齐

目标：将 AIS 事件按目标时间步（如 6H）分桶，与 P1 对齐。

实现：
- make_time_bins(month, step="6H"): 生成对齐到整点的 UTC 时间桶边界（epoch seconds）。
- annotate_time_bins(...): 为 parquet 添加 time_bin_idx 列，并统计时间覆盖率。

约束：
- 优先 polars，回退 pandas。
- 非 dry-run 写盘时调用 register_artifact()。
- 路径使用 os.path.join。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None  # type: ignore

from ArcticRoute.cache.index_util import register_artifact


def _get_grid_spec_freq(grid_path: Optional[str] = None) -> Optional[str]:
    """尝试从 grid_spec.json 读取 time.freq。"""
    path = grid_path or os.path.join(os.getcwd(), "ArcticRoute", "config", "grid_spec.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        return spec.get("time", {}).get("freq")
    except Exception:
        return None


def make_time_bins(month: str, step: str = "6H") -> np.ndarray:
    """为指定月份生成对齐到整点的 UTC 时间桶（epoch seconds）。"""
    if pd is None:
        raise RuntimeError("pandas is required for make_time_bins")
    start_ts = pd.to_datetime(f"{month}-01", format="%Y%m-%d", utc=True)
    end_ts = start_ts + pd.offsets.MonthEnd(1)
    bins = pd.date_range(start=start_ts, end=end_ts, freq=step, inclusive="left")
    return bins.astype('int64') // 10**9


def _read_parquet(path: str):
    if pl is not None:
        return pl.read_parquet(path)
    if pd is not None:
        return pd.read_parquet(path)  # type: ignore
    raise RuntimeError("No dataframe engine available")


def _write_parquet(df_any: Any, out_path: str) -> None:
    if pl and isinstance(df_any, pl.DataFrame):  # type: ignore[attr-defined]
        df_any.write_parquet(out_path)
        return
    if pd and isinstance(df_any, pd.DataFrame):  # type: ignore[attr-defined]
        df_any.to_parquet(out_path)
        return
    raise RuntimeError("Unsupported dataframe type for writing")


def annotate_time_bins(parquet_in: str, parquet_out: str, step: Optional[str] = None, dry_run: bool = True) -> Dict[str, Any]:
    """为 parquet 添加 time_bin_idx 并输出覆盖统计。

    - step: 时间步长（如 '6H', '1D'），若无则从 grid_spec.json 或默认 '6H'。
    """
    df = _read_parquet(parquet_in)
    if df.is_empty() if hasattr(df, 'is_empty') else df.empty:
        return {"out": parquet_out, "stats": {"bin_count": 0, "coverage_pct": 0.0, "oob_count": 0}}

    time_step = step or _get_grid_spec_freq() or "6H"

    # 从数据中推断月份
    ts_col = df["ts"] if isinstance(df, pd.DataFrame) else df.get_column("ts")  # type: ignore
    first_ts = ts_col.iloc[0] if hasattr(ts_col, 'iloc') else ts_col[0]
    month_str = pd.to_datetime(int(first_ts), unit='s', utc=True).strftime("%Y%m")

    bins = make_time_bins(month_str, step=time_step)
    ts_values = ts_col.to_numpy() if hasattr(ts_col, 'to_numpy') else ts_col.to_numpy(zero_copy_only=False)

    # 使用 searchsorted 找索引
    bin_indices = np.searchsorted(bins, ts_values, side='right') - 1

    # 统计
    total_rows = len(df)
    oob_mask = (bin_indices < 0) | (bin_indices >= len(bins) -1)
    oob_count = int(oob_mask.sum())
    coverage_pct = (total_rows - oob_count) / total_rows if total_rows > 0 else 0.0
    stats = {
        "bin_count": len(bins),
        "coverage_pct": round(coverage_pct * 100, 2),
        "oob_count": oob_count,
        "time_step": time_step,
        "month": month_str,
    }

    if not dry_run:
        if isinstance(df, pd.DataFrame):
            df_out = df.copy()
            df_out["time_bin_idx"] = bin_indices
        else: # polars
            df_out = df.with_columns(pl.Series("time_bin_idx", bin_indices)) # type: ignore

        os.makedirs(os.path.dirname(parquet_out), exist_ok=True)
        _write_parquet(df_out, parquet_out)
        try:
            register_artifact(
                run_id=os.environ.get("RUN_ID", ""),
                kind="ais_timebinned",
                path=parquet_out,
                attrs=stats
            )
        except Exception:
            pass

    return {"out": parquet_out, "stats": stats}


__all__ = ["make_time_bins", "annotate_time_bins"]

