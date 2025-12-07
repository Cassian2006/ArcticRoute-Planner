"""AIS 流式摄取（JSON/JSONL -> 分区 Parquet）

目标：低内存将大量 JSON 写成分区 Parquet（year=YYYY/month=MM/），只保留规范列。

特性与约束：
- 优先使用 polars（行批处理），无则回退 pandas+pyarrow；
- 支持 JSON（数组）与 JSONL（NDJSON）两种格式；
- 仅保留 B-02 统一列，固定类型：
  - mmsi:int64, ts:int64, lat:float32, lon:float32,
  - sog:float32, cog:float32, heading:float32, vessel_type:str,
  - loa:float32, beam:float32, nav_status:str
- 写盘路径：data_processed/ais_parquet/year=YYYY/month=MM/part-*.parquet
- 注册工件：kind="ais_parquet"，attrs: {"ym": YYYYMM, "rows": N}
- 支持基础清洗（B-04）：越界/速度阈值/去重（mmsi+ts）
- 所有写盘仅在非 dry-run；写盘/注册走 register_artifact()
- Windows 路径使用 os.path.join
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from datetime import datetime

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import ujson as _ujson  # type: ignore
except Exception:  # pragma: no cover
    _ujson = None  # type: ignore

from ArcticRoute.io.ais_norm import normalize_record, load_keymap
from ArcticRoute.cache.index_util import register_artifact

CANONICAL_ORDER: List[str] = [
    "mmsi", "ts", "lat", "lon",
    "sog", "cog", "heading", "vessel_type", "loa", "beam", "nav_status",
]

DTYPES_PL = {
    "mmsi": pl.Int64 if pl else int,  # type: ignore[attr-defined]
    "ts": pl.Int64 if pl else int,    # type: ignore[attr-defined]
    "lat": pl.Float32 if pl else float,  # type: ignore[attr-defined]
    "lon": pl.Float32 if pl else float,  # type: ignore[attr-defined]
    "sog": pl.Float32 if pl else float,  # type: ignore[attr-defined]
    "cog": pl.Float32 if pl else float,  # type: ignore[attr-defined]
    "heading": pl.Float32 if pl else float,  # type: ignore[attr-defined]
    "vessel_type": pl.Utf8 if pl else str,  # type: ignore[attr-defined]
    "loa": pl.Float32 if pl else float,  # type: ignore[attr-defined]
    "beam": pl.Float32 if pl else float,  # type: ignore[attr-defined]
    "nav_status": pl.Utf8 if pl else str,  # type: ignore[attr-defined]
}

DTYPES_PD = {
    "mmsi": "int64",
    "ts": "int64",
    "lat": "float32",
    "lon": "float32",
    "sog": "float32",
    "cog": "float32",
    "heading": "float32",
    "vessel_type": "string",
    "loa": "float32",
    "beam": "float32",
    "nav_status": "string",
}


def _iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    open_json = _ujson.loads if _ujson else json.loads
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = open_json(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue


def _iter_json_array(path: str) -> Iterator[Dict[str, Any]]:
    open_json = _ujson.loads if _ujson else json.loads
    # 逐行读到适中上限后再解析，避免载入超大文件
    # 这里简单直接一次性解析，假设单文件大小适中（若很大建议使用 JSONL 格式）
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = open_json(f.read())
        if isinstance(data, list):
            for it in data:
                if isinstance(it, dict):
                    yield it
    except Exception:
        return


def _iter_raw_records(path: str) -> Iterator[Dict[str, Any]]:
    # 简易判定 JSONL：按行解析能成功>0即认为是 JSONL
    parsed_any = False
    for obj in _iter_jsonl(path):
        parsed_any = True
        yield obj
    if parsed_any:
        return
    # 否则尝试 JSON 数组
    for obj in _iter_json_array(path):
        yield obj


def _ym_from_ts(ts: int) -> Tuple[int, int]:
    dt = datetime.utcfromtimestamp(int(ts))
    return dt.year, dt.month


def _ensure_out_dir(base_dir: str, year: int, month: int) -> str:
    p = os.path.join(base_dir, f"year={year:04d}", f"month={month:02d}")
    os.makedirs(p, exist_ok=True)
    return p


def _write_parquet(df_any: Any, out_path: str) -> None:
    if pl and isinstance(df_any, pl.DataFrame):  # type: ignore[attr-defined]
        df_any.write_parquet(out_path)
        return
    if pd and isinstance(df_any, pd.DataFrame):  # type: ignore[attr-defined]
        df_any.to_parquet(out_path, engine="pyarrow")
        return
    raise RuntimeError("No available dataframe engine to write parquet")


def ingest_months(src_dir: str, months: List[str], out_base: str, run_id: str, dry_run: bool = True, speed_max: float = 35.0, force: bool = False, part_size: int = 50000) -> Dict[str, Any]:
    """将 src_dir 下的 JSON/JSONL 流式标准化并按月分区落盘。

    - months: 目标月份列表（YYYYMM）。空列表表示全量扫，但仍按记录 ts 分区。
    返回：{"written": int, "partitions": {"YYYYMM": {"files": int, "rows": int}}, "summary_path": path_or_preview}
    """
    keymap = load_keymap(default_root=os.path.join(os.getcwd(), "reports", "recon"))

    summary = {
        "raw_cnt": 0,
        "kept_cnt": 0,
        "drop_cnt": 0,
        "drop_reasons": {"invalid": 0, "speed": 0, "dedup": 0},
    }
    partitions: Dict[str, Dict[str, int]] = {}

    # 去重缓存（mmsi, ts）
    seen: set[Tuple[int, int]] = set()

    # 扫描文件
    json_files: List[str] = []
    for dp, _, fns in os.walk(src_dir):
        for fn in fns:
            if fn.lower().endswith(".json"):
                json_files.append(os.path.join(dp, fn))
    json_files.sort()

    # 缓冲按分区聚合，定期 flush
    buffers: Dict[str, List[Dict[str, Any]]] = {}
    PART_SIZE = 50000  # 每分区每批最多条数
    part_seq: Dict[str, int] = {}

    def flush_partition(ym: str) -> None:
        nonlocal partitions
        rows = buffers.get(ym) or []
        if not rows:
            return
        # 构建 DF
        if pl:
            df = pl.DataFrame(rows, schema=DTYPES_PL)  # type: ignore[arg-type]
        elif pd:
            import pandas as _pd  # type: ignore
            df = _pd.DataFrame(rows)
            # 强制列顺序与类型
            for col, dt in DTYPES_PD.items():
                if col in df.columns:
                    try:
                        df[col] = df[col].astype(dt)
                    except Exception:
                        pass
            df = df[[c for c in CANONICAL_ORDER if c in df.columns]]
        else:
            raise RuntimeError("No dataframe engine available")
        # 输出路径
        year, month = int(ym[:4]), int(ym[4:])
        out_dir = _ensure_out_dir(out_base, year, month)
        part_seq[ym] = part_seq.get(ym, 0) + 1
        out_path = os.path.join(out_dir, f"part-{run_id}-{part_seq[ym]:05d}.parquet")
        if not dry_run:
            _write_parquet(df, out_path)
            try:
                register_artifact(run_id=run_id, kind="ais_parquet", path=out_path, attrs={"ym": ym, "rows": len(rows)})
            except Exception:
                pass
        # 更新统计
        stat = partitions.setdefault(ym, {"files": 0, "rows": 0})
        stat["files"] += 0 if dry_run else 1
        stat["rows"] += len(rows)
        # 清空缓冲
        buffers[ym] = []

    for fp in json_files:
        # 仅按文件名包含 YYYY.MM 或 YYYYMM 粗过滤（如果提供 months）
        if months:
            name = os.path.basename(fp)
            mm_ok = any(m in name.replace(".", "") for m in months)
            if not mm_ok:
                # 也可能跨月，继续读取并由 ts 决定
                pass
        for raw in _iter_raw_records(fp):
            summary["raw_cnt"] += 1
            norm = normalize_record(raw, keymap)
            if norm is None:
                summary["drop_cnt"] += 1
                summary["drop_reasons"]["invalid"] += 1
                continue
            # 速度阈值
            sog = norm.get("sog")
            if sog is not None:
                try:
                    sf = float(sog)
                    if not (0.0 <= sf <= float(speed_max)):
                        summary["drop_cnt"] += 1
                        summary["drop_reasons"]["speed"] += 1
                        continue
                except Exception:
                    pass
            # 去重
            key = (int(norm["mmsi"]), int(norm["ts"]))
            if key in seen:
                summary["drop_cnt"] += 1
                summary["drop_reasons"]["dedup"] += 1
                continue
            seen.add(key)
            summary["kept_cnt"] += 1
            # 分区
            y, m = _ym_from_ts(int(norm["ts"]))
            ym = f"{y:04d}{m:02d}"
            if months and ym not in months:
                # 非目标月，跳过
                continue
            # 只保留规范列与类型收敛
            row = {k: norm.get(k) for k in CANONICAL_ORDER}
            buffers.setdefault(ym, []).append(row)
            if len(buffers[ym]) >= PART_SIZE:
                flush_partition(ym)

    # flush 所有分区
    for ym in list(buffers.keys()):
        flush_partition(ym)

    # 写入 summary（features_summary.json）
    recon_dir = os.path.join(os.getcwd(), "reports", "recon")
    os.makedirs(recon_dir, exist_ok=True)
    summary_path = os.path.join(recon_dir, "features_summary.json")
    if not dry_run:
        try:
            with open(summary_path, "w", encoding="utf-8") as fw:
                json.dump(summary, fw, ensure_ascii=False, indent=2)
            register_artifact(run_id=run_id, kind="features_summary", path=summary_path, attrs={})
        except Exception:
            pass

    return {"written": sum(p.get("files", 0) for p in partitions.values()), "partitions": partitions, "summary_path": summary_path}


def ingest_from_cli(src: str, months: List[str], dry_run: bool = True, speed_max: float = 35.0, force: bool = False, part_size: int = 50000) -> Dict[str, Any]:
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_base = os.path.join(os.getcwd(), "data_processed", "ais_parquet")
    return ingest_months(src_dir=src, months=months, out_base=out_base, run_id=run_id, dry_run=dry_run, speed_max=speed_max)


__all__ = ["ingest_months", "ingest_from_cli"]

