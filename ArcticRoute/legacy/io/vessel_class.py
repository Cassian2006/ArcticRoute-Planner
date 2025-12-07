"""船型/尺度分类（vclass）

目标：
- 依据 vessel_type（优先）或尺度（loa/beam）生成粗粒度 vclass 标签
- 写回 parquet，并输出分布统计（计数/占比）摘要

约束：
- 优先 polars，回退 pandas
- 非 dry-run 才写盘，并通过 register_artifact() 登记
- 全程 os.path.join 构造路径
"""
from __future__ import annotations

import os
from typing import Any, Dict, Tuple

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from ArcticRoute.cache.index_util import register_artifact

# 简化的类型归类表（可按需扩展/外置配置）
_TYPE_MAP = {
    "tanker": "tanker",
    "oil": "tanker",
    "lng": "tanker",
    "lpg": "tanker",
    "cargo": "cargo",
    "container": "cargo",
    "bulk": "cargo",
    "general": "cargo",
    "fishing": "fishing",
    "fish": "fishing",
    "passenger": "passenger",
    "ferry": "passenger",
    "tug": "tug",
    "ice": "icebreaker",
    "icebreaker": "icebreaker",
    "research": "other",
    "pilot": "other",
}


def _norm_type(s: Any) -> str:
    if s is None:
        return ""
    try:
        return str(s).strip().lower()
    except Exception:
        return ""


def _size_bucket(loa: Any, beam: Any) -> str:
    try:
        L = float(loa) if loa is not None and str(loa) != "" else None
    except Exception:
        L = None
    try:
        B = float(beam) if beam is not None and str(beam) != "" else None
    except Exception:
        B = None
    # 仅有 L 或 B 时也给出一个粗略分级
    metric = None
    if L is not None and B is not None:
        metric = max(L, B)
    elif L is not None:
        metric = L
    elif B is not None:
        metric = B
    else:
        return "unknown"
    if metric < 70:
        return "S"
    if metric < 150:
        return "M"
    if metric < 250:
        return "L"
    return "XL"


def _pick_vclass(vessel_type: Any, loa: Any, beam: Any) -> str:
    vt = _norm_type(vessel_type)
    # 优先 vessel_type
    for k, v in _TYPE_MAP.items():
        if k in vt and k != "":
            return v
    # 尺度退化
    return f"size_{_size_bucket(loa, beam)}"


def annotate_vclass_df(df_any: Any) -> Any:
    """为 DataFrame 添加 vclass 列。"""
    if pl is not None and isinstance(df_any, pl.DataFrame):  # type: ignore[attr-defined]
        def _fn(vt, L, B):
            return _pick_vclass(vt, L, B)
        return df_any.with_columns([
            pl.map_elements([pl.col("vessel_type"), pl.col("loa"), pl.col("beam")], _fn, return_dtype=pl.Utf8).alias("vclass")
        ])
    # pandas
    dpf = df_any.copy()
    dpf["vclass"] = dpf.apply(lambda r: _pick_vclass(r.get("vessel_type"), r.get("loa"), r.get("beam")), axis=1)
    return dpf


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


def _dist_summary(df_any: Any) -> Dict[str, Any]:
    if pl and isinstance(df_any, pl.DataFrame):  # type: ignore[attr-defined]
        vc = df_any.get_column("vclass").to_list()  # type: ignore[attr-defined]
        from collections import Counter
        cnt = Counter(vc)
        total = sum(cnt.values())
        pct = {k: (cnt[k] / total if total else 0.0) for k in cnt}
        return {"counts": dict(cnt), "percents": pct, "total": total}
    # pandas
    s = df_any["vclass"].value_counts(dropna=False)
    total = int(s.sum())
    counts = {str(k): int(v) for k, v in s.to_dict().items()}
    percents = {k: (counts[k] / total if total else 0.0) for k in counts}
    return {"counts": counts, "percents": percents, "total": total}


def write_vclass(parquet_in: str, parquet_out: str, summary_out: str, dry_run: bool = True) -> Dict[str, Any]:
    df = _read_parquet(parquet_in)
    df2 = annotate_vclass_df(df)
    summary = _dist_summary(df2)
    if not dry_run:
        os.makedirs(os.path.dirname(parquet_out), exist_ok=True)
        _write_parquet(df2, parquet_out)
        try:
            import json
            os.makedirs(os.path.dirname(summary_out), exist_ok=True)
            with open(summary_out, "w", encoding="utf-8") as fw:
                json.dump(summary, fw, ensure_ascii=False, indent=2)
            register_artifact(run_id=os.environ.get("RUN_ID", ""), kind="ais_vclass", path=parquet_out, attrs={})
            register_artifact(run_id=os.environ.get("RUN_ID", ""), kind="vclass_summary", path=summary_out, attrs={})
        except Exception:
            pass
    return {"out": parquet_out, "summary": summary_out, "stats": summary}


__all__ = ["annotate_vclass_df", "write_vclass"]

