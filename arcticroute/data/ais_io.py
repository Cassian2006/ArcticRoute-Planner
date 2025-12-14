from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pandas as pd
import pathlib
import json
import os


@dataclass
class AISLoadConfig:
    root: pathlib.Path
    year: int = 2024
    lat_min: float = 60.0
    lat_max: float = 90.0
    lon_min: float = -180.0
    lon_max: float = 180.0


# 基于探测与常见字段的猜测映射
FIELD_MAP_GUESS: Dict[str, List[str]] = {
    "ts": ["timestamp", "BaseDateTime", "time", "datetime", "basedatetime"],
    "lat": ["lat", "LAT", "latitude", "y"],
    "lon": ["lon", "LON", "longitude", "x"],
    "mmsi": ["mmsi", "MMSI"],
    "sog_knots": ["sog", "SOG", "speed"],
    "cog_deg": ["cog", "COG", "heading", "course"],
}


def _resolve_ais_dir(root: pathlib.Path, year: int) -> pathlib.Path:
    """优先使用 data_real/ais/raw/<year>，若不存在则回退到 data_real/ais/<year>。"""
    raw_dir = root / "data_real" / "ais" / "raw" / str(year)
    if raw_dir.exists():
        return raw_dir
    simple_dir = root / "data_real" / "ais" / str(year)
    return simple_dir


def _detect_structure(path: pathlib.Path) -> str:
    """检测文件结构：list / dict / jsonl / unknown"""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline()
            if not first:
                return "unknown"
            try:
                obj = json.loads(first)
                if isinstance(obj, dict):
                    # 可能是 jsonl，也可能是完整单对象
                    # 再读一行看看
                    second = f.readline()
                    if second.strip():
                        try:
                            json.loads(second)
                            return "jsonl"
                        except json.JSONDecodeError:
                            # 倾向于整体 dict
                            pass
                    # 回放并尝试整文件
                    f.seek(0)
                    data = json.load(f)
                    return "dict" if isinstance(data, dict) else "unknown"
                elif isinstance(obj, list):
                    return "list"
            except json.JSONDecodeError:
                # 可能是 jsonl
                try:
                    json.loads(first)
                    return "jsonl"
                except Exception:
                    return "unknown"
    except Exception:
        return "unknown"
    return "unknown"


def _iter_records_from_object(obj: Any) -> List[dict]:
    """从已解析的对象中提取候选记录列表。"""
    out: List[dict] = []
    if isinstance(obj, dict):
        # 优先从 data 字段中找列表
        data = obj.get("data") if "data" in obj else None
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    out.append(item)
        else:
            # 若本对象自身像一条记录
            if any(k.lower() in ("lat", "latitude") for k in obj.keys()):
                out.append(obj)
    elif isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                # 有些文件是 [ {code:..., data:[...]}, {...} ]
                out.extend(_iter_records_from_object(it))
    return out


def _pick_first(d: dict, candidates: List[str]) -> Optional[Any]:
    for k in candidates:
        # 不区分大小写匹配
        if k in d:
            return d.get(k)
        # 尝试大小写变体
        for key in d.keys():
            if key.lower() == k.lower():
                return d.get(key)
    return None


def _normalize_row(rec: dict) -> Optional[dict]:
    """使用 FIELD_MAP_GUESS 将原始记录映射为标准列。映射失败返回 None。"""
    row: Dict[str, Any] = {}
    row["ts_raw"] = _pick_first(rec, FIELD_MAP_GUESS["ts"])  # 先保留原始，稍后统一解析
    row["lat"] = _pick_first(rec, FIELD_MAP_GUESS["lat"])
    row["lon"] = _pick_first(rec, FIELD_MAP_GUESS["lon"])
    row["mmsi"] = _pick_first(rec, FIELD_MAP_GUESS["mmsi"])
    row["sog_knots"] = _pick_first(rec, FIELD_MAP_GUESS["sog_knots"])
    row["cog_deg"] = _pick_first(rec, FIELD_MAP_GUESS["cog_deg"]) 

    # 若核心字段全缺失，视为无效
    if row["ts_raw"] is None and (row["lat"] is None or row["lon"] is None) and row["mmsi"] is None:
        return None
    return row


def load_ais_json_to_df(config: AISLoadConfig) -> pd.DataFrame:
    """
    从 data_real/ais/raw/<year>/ 或 data_real/ais/<year>/ 目录读取所有 json，
    标准化为统一 DataFrame，至少包含:
    - ts: datetime64[ns]
    - lat, lon: float
    - mmsi: int64
    - sog_knots: float (若无则可为 NaN)
    - cog_deg: float (若无则可为 NaN)
    
    仅保留纬度在 [config.lat_min, config.lat_max] 范围内的记录。
    """
    ais_dir = _resolve_ais_dir(config.root, config.year)
    if not ais_dir.exists():
        return pd.DataFrame(columns=["ts", "lat", "lon", "mmsi", "sog_knots", "cog_deg"])  # 空

    rows: List[dict] = []

    for path in sorted(ais_dir.glob("*.json")):
        structure = _detect_structure(path)
        try:
            if structure in ("list", "dict"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    obj = json.load(f)
                for rec in _iter_records_from_object(obj):
                    norm = _normalize_row(rec)
                    if norm is not None:
                        rows.append(norm)
            elif structure == "jsonl":
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        try:
                            obj = json.loads(s)
                        except json.JSONDecodeError:
                            continue
                        for rec in _iter_records_from_object(obj):
                            norm = _normalize_row(rec)
                            if norm is not None:
                                rows.append(norm)
            else:
                # 兜底：尝试整文件 load
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        obj = json.load(f)
                    for rec in _iter_records_from_object(obj):
                        norm = _normalize_row(rec)
                        if norm is not None:
                            rows.append(norm)
                except Exception:
                    pass
        except Exception:
            # 忽略单文件错误
            continue

    if not rows:
        return pd.DataFrame(columns=["ts", "lat", "lon", "mmsi", "sog_knots", "cog_deg"])  # 空

    df = pd.DataFrame(rows)

    # 类型转换与清洗
    # 数值列先转为 numeric
    for col in ["lat", "lon", "sog_knots", "cog_deg", "mmsi"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 解析时间
    df["ts"] = pd.to_datetime(df["ts_raw"], errors="coerce", utc=True)

    # 丢弃关键字段缺失的行
    df = df.dropna(subset=["ts", "lat", "lon", "mmsi"], how="any")

    if len(df) == 0:
        return pd.DataFrame(columns=["ts", "lat", "lon", "mmsi", "sog_knots", "cog_deg"])

    # 物理范围过滤（纬度 [-90, 90]，经度 [-180, 180]）
    df = df[(df["lat"] >= -90.0) & (df["lat"] <= 90.0)]
    df = df[(df["lon"] >= -180.0) & (df["lon"] <= 180.0)]

    # 配置范围过滤（纬度和经度）
    df = df[(df["lat"] >= config.lat_min) & (df["lat"] <= config.lat_max)]
    df = df[(df["lon"] >= config.lon_min) & (df["lon"] <= config.lon_max)]

    # 速度 sanity check：|sog_knots| <= 60 或 NaN
    if "sog_knots" in df.columns:
        mask_ok = (df["sog_knots"].abs() <= 60.0) | (df["sog_knots"].isna())
        df = df[mask_ok]

    # 只保留标准列
    keep_cols = ["ts", "lat", "lon", "mmsi", "sog_knots", "cog_deg"]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = pd.NA

    df = df[keep_cols].sort_values("ts").reset_index(drop=True)

    # 强制类型转换
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["mmsi"] = df["mmsi"].astype("int64")
    df["lat"] = df["lat"].astype("float64")
    df["lon"] = df["lon"].astype("float64")
    df["sog_knots"] = df["sog_knots"].astype("float64")
    df["cog_deg"] = df["cog_deg"].astype("float64")

    return df



