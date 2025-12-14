#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scan AIS raw JSONs and emit a normalized Parquet with [mmsi, ts, lat, lon, sog, cog].

@role: pipeline
"""

"""
从 ArcticRoute/data_raw/ais 下自动检索 JSON/JSONL/GeoJSON，标准化为列[mmsi,time_utc,lat,lon,sog,cog]，
生成秒级 ts，并写出 ArcticRoute/data_processed/ais/raw_auto.parquet。

用法：
  python ArcticRoute/scripts/gen_raw_auto_from_json.py --src ArcticRoute/data_raw/ais --out ArcticRoute/data_processed/ais/raw_auto.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd  # type: ignore
import numpy as np  # type: ignore

ALIASES = {
    "mmsi": ["mmsi", "ship_id", "properties.mmsi"],
    "time_utc": ["time_utc", "timestamp", "time", "message_stamp", "time_stamp", "properties.time_utc", "basedatetime", "base_datetime", "msgtime"],
    "lat": ["lat", "latitude", "y", "geometry.coordinates[1]", "properties.lat", "lat_dd", "y_lat"],
    "lon": ["lon", "longitude", "x", "geometry.coordinates[0]", "properties.lon", "lon_dd", "x_lon"],
    "sog": ["sog", "speed_over_ground", "sog_knot", "speed", "properties.sog"],
    "cog": ["cog", "course_over_ground", "course", "properties.cog"],
}
REQUIRED = ["mmsi", "time_utc", "lat", "lon"]
OPTIONAL = ["sog", "cog"]


def _pick_col(df: pd.DataFrame, keys: list[str], new_name: str) -> bool:
    for k in keys:
        if k in df.columns:
            df[new_name] = df[k]
            return True
    return False


def _normalize_json(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        arr = None
        for key in ("features", "records", "items", "data"):
            if key in obj and isinstance(obj[key], list):
                arr = obj[key]
                break
        if arr is None:
            arr = [obj]
    elif isinstance(obj, list):
        arr = obj
    else:
        raise ValueError(f"无法识别的 JSON 结构: {path}")
    df = pd.json_normalize(arr, max_level=2)
    df.columns = [str(c) for c in df.columns]
    ok = True
    ok &= _pick_col(df, ALIASES["mmsi"], "mmsi")
    ok &= _pick_col(df, ALIASES["time_utc"], "time_utc")
    ok &= _pick_col(df, ALIASES["lat"], "lat")
    ok &= _pick_col(df, ALIASES["lon"], "lon")
    if not ok:
        raise ValueError(f"{path} 无法映射到必要列")
    _pick_col(df, ALIASES["sog"], "sog")
    _pick_col(df, ALIASES["cog"], "cog")
    out = pd.DataFrame({
        "mmsi": df.get("mmsi"),
        "time_utc": df.get("time_utc"),
        "lat": df.get("lat"),
        "lon": df.get("lon"),
        "sog": df.get("sog"),
        "cog": df.get("cog"),
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="扫描 data_raw/ais JSON 并生成 raw_auto.parquet")
    ap.add_argument("--src", default="ArcticRoute/data_raw/ais", help="JSON 源目录")
    ap.add_argument("--out", default="ArcticRoute/data_processed/ais/raw_auto.parquet", help="输出 Parquet 路径")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(list(src.rglob("*.jsonl")) + list(src.rglob("*.json")) + list(src.rglob("*.geojson")))
    if not files:
        raise FileNotFoundError(f"未在 {src} 找到 json/jsonl/geojson")

    parts: list[pd.DataFrame] = []
    for fp in files:
        try:
            if fp.suffix.lower() == ".jsonl":
                # 行式 JSON：按块读
                for ch in pd.read_json(fp, lines=True, chunksize=200_000):
                    ch.columns = [str(c).strip() for c in ch.columns]
                    ch = ch.rename(columns={c: c.lower() for c in ch.columns})
                    # 如果必要列已扁平存在，直接选取；否则回退到 normalize
                    if all(c in ch.columns for c in ("mmsi", "time_utc", "lat", "lon")):
                        df = ch[["mmsi", "time_utc", "lat", "lon"]].copy()
                        for opt in OPTIONAL:
                            if opt in ch.columns:
                                df[opt] = ch[opt]
                            else:
                                df[opt] = np.nan
                    else:
                        df = _normalize_json(fp)
                    parts.append(df)
            else:
                df = _normalize_json(fp)
                parts.append(df)
        except Exception as e:
            raise RuntimeError(f"处理 {fp} 失败: {e}") from e

    if not parts:
        raise RuntimeError("清洗后为空")

    df = pd.concat(parts, ignore_index=True)
    df["mmsi"] = df["mmsi"].astype(str).str.strip()
    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce", dayfirst=True)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["sog"] = pd.to_numeric(df["sog"], errors="coerce")
    df["cog"] = pd.to_numeric(df["cog"], errors="coerce")
    df = df.dropna(subset=["mmsi", "time_utc", "lat", "lon"]).sort_values(["mmsi", "time_utc"])  # noqa: E712
    df["ts"] = (df["time_utc"].view("int64") // 10**9).astype("int64")
    df = df[["mmsi", "ts", "lat", "lon", "sog", "cog"]]

    df.to_parquet(out, index=False)
    print(f"[OK] 写出 {out} rows={len(df)}")


if __name__ == "__main__":
    main()

