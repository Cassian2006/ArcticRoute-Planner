#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stream-parse AIS JSON/JSONL/GeoJSON into a normalized Parquet table.

@role: pipeline
"""

"""
递归扫描指定目录下的 JSON/JSONL/GeoJSON，大文件采用 ijson 流式解析，
将记录标准化为列 [mmsi, time_utc, lat, lon, sog, cog, ts] 并写为 Parquet。

用法：
  python ArcticRoute/scripts/convert_ais_json_stream.py \
    --src ArcticRoute/data_raw/ais \
    --out ArcticRoute/data_processed/ais/raw_auto.parquet

依赖：pandas, pyarrow, ijson
pip install pandas pyarrow ijson
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

try:
    import ijson  # type: ignore
except Exception as e:  # pragma: no cover
    ijson = None  # type: ignore

ALIASES = {
    "mmsi": [
        "mmsi", "MMSI", "ship_id", "properties.mmsi", "Mmsi", "MMSI_ID"
    ],
    "time_utc": [
        "time_utc", "Time_UTC", "timestamp", "time", "postime", "message_stamp", "time_stamp",
        "properties.time_utc", "basedatetime", "base_datetime", "msgtime", "BaseDateTime"
    ],
    "lat": [
        "lat", "Lat", "latitude", "Latitude", "y", "Y", "properties.lat", "lat_dd", "y_lat"
    ],
    "lon": [
        "lon", "Lon", "longitude", "Longitude", "x", "X", "properties.lon", "lon_dd", "x_lon"
    ],
    "sog": [
        "sog", "SOG", "speed_over_ground", "sog_knot", "speed", "Speed", "properties.sog"
    ],
    "cog": [
        "cog", "COG", "course_over_ground", "course", "Course", "properties.cog"
    ],
}

REQUIRED = ["mmsi", "time_utc", "lat", "lon"]
OPTIONAL = ["sog", "cog"]


def _get_nested(obj: Any, path: str) -> Optional[Any]:
    # 支持 a.b.c 和 geometry.coordinates[0]
    cur = obj
    try:
        parts = path.replace("]", "").split("[")
        # 先按点分
        segs: List[str] = []
        for p in parts:
            segs.extend(p.split("."))
        for seg in segs:
            if seg == "":
                continue
            if seg.isdigit():
                cur = cur[int(seg)]
            else:
                if isinstance(cur, dict):
                    # 兼容大小写键
                    key_map = {str(k).lower(): k for k in cur.keys()}
                    k = key_map.get(seg.lower())
                    if k is None:
                        return None
                    cur = cur[k]
                else:
                    return None
        return cur
    except Exception:
        return None


def _pick_from_obj(obj: Dict[str, Any], key_list: List[str]) -> Optional[Any]:
    for k in key_list:
        v = _get_nested(obj, k)
        if v is not None:
            return v
    return None


def _yield_concat_objects(path: Path) -> Iterable[Dict[str, Any]]:
    """处理一种非标准 JSON：文件中为用逗号分隔的一连串对象，但缺少最外层方括号。
    通过括号计数提取每个对象，尝试 json.loads 解析。
    """
    buf = []
    depth = 0
    started = False
    with open(path, "r", encoding="utf-8") as f:
        while True:
            ch = f.read(1)
            if not ch:
                # 文件结束
                if buf:
                    s = ''.join(buf).strip().rstrip(',')
                    try:
                        yield json.loads(s)
                    except Exception:
                        pass
                break
            buf.append(ch)
            if ch == '{':
                depth += 1
                started = True
            elif ch == '}':
                depth -= 1
                if started and depth == 0:
                    # 完成一个对象
                    s = ''.join(buf).strip().rstrip(',')
                    buf.clear()
                    started = False
                    try:
                        yield json.loads(s)
                    except Exception:
                        # 跳过无法解析的片段
                        continue


def _yield_records_from_json(path: Path) -> Iterable[Dict[str, Any]]:
    # 返回原始对象，后续再标准化
    # 先尝试非标准拼接对象
    try:
        # 粗略检测：文件不以 [ 开头，但包含 '{'
        head = path.read_text(encoding="utf-8", errors="ignore")[0:512]
        if ('{' in head) and ('[' not in head.split('{',1)[0]):
            # 可能是纯对象拼接
            yielded = False
            for rec in _yield_concat_objects(path):
                yielded = True
                yield rec
            if yielded:
                return
    except Exception:
        pass

    if ijson is None:
        # 退化：一次性加载（可能很大）
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            for rec in obj:
                yield rec
        elif isinstance(obj, dict):
            # 常见容器 keys
            for key in ("features", "records", "items", "data"):
                if key in obj and isinstance(obj[key], list):
                    for rec in obj[key]:
                        yield rec
                    return
            yield obj
        return
    # 优先流式
    with open(path, "r", encoding="utf-8") as f:
        try:
            # 尝试作为数组流式解析
            for rec in ijson.items(f, "item"):
                yield rec
            return
        except Exception:
            f.seek(0)
        try:
            # 尝试读取常见容器数组
            for key in ("features", "records", "items", "data"):
                f.seek(0)
                for rec in ijson.items(f, f"{key}.item"):
                    yield rec
                # 若能进入此循环，说明找到了该容器
                return
        except Exception:
            f.seek(0)
        # 退化：整文件解析
        obj = json.load(f)
        if isinstance(obj, list):
            for rec in obj:
                yield rec
        elif isinstance(obj, dict):
            yield obj


def _yield_records_from_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            flat.update(_flatten(v, key))
    elif isinstance(obj, list):
        # 仅展开前两个元素，避免爆炸
        for i, v in enumerate(obj[:2]):
            key = f"{prefix}[{i}]"
            flat.update(_flatten(v, key))
        # 同时保留整个列表本身
        flat[prefix] = obj
    else:
        flat[prefix] = obj
    return flat


def _heuristic_std(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    import pandas as _pd  # type: ignore
    flat = _flatten(obj)
    # 候选集
    lat_key, lon_key, time_key, mmsi_key = None, None, None, None
    # 先基于名字优先
    for k, v in flat.items():
        kl = str(k).lower()
        if lat_key is None and "lat" in kl and isinstance(v, (int, float, str)):
            try:
                vf = float(v)
                if -90 <= vf <= 90:
                    lat_key = k
            except Exception:
                pass
        if lon_key is None and ("lon" in kl or "lng" in kl) and isinstance(v, (int, float, str)):
            try:
                vf = float(v)
                if -180 <= vf <= 180:
                    lon_key = k
            except Exception:
                pass
        if mmsi_key is None and ("mmsi" in kl or "ship" in kl):
            sv = str(v)
            digits = ''.join(ch for ch in sv if ch.isdigit())
            if 7 <= len(digits) <= 9:
                mmsi_key = k
        if time_key is None and any(t in kl for t in ("time", "date")):
            try:
                _ = _pd.to_datetime(v)
                time_key = k
            except Exception:
                pass
    # 若仍未找到，尝试 geometry.coordinates
    coords = flat.get("geometry.coordinates")
    if (lat_key is None or lon_key is None) and isinstance(coords, list) and len(coords) >= 2:
        lon_key = lon_key or "geometry.coordinates[0]"
        lat_key = lat_key or "geometry.coordinates[1]"
        flat[lon_key] = coords[0]
        flat[lat_key] = coords[1]
    # 构造输出
    try:
        mmsi = flat[mmsi_key] if mmsi_key else None
        time_utc = flat[time_key] if time_key else None
        lat = flat[lat_key] if lat_key else None
        lon = flat[lon_key] if lon_key else None
        if None in (mmsi, time_utc, lat, lon):
            return None
        sog = flat.get("sog") or flat.get("speed")
        cog = flat.get("cog") or flat.get("course")
        return {"mmsi": mmsi, "time_utc": time_utc, "lat": lat, "lon": lon, "sog": sog, "cog": cog}
    except Exception:
        return None


def _std_from_obj(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # geometry.coordinates 支持
    if _pick_from_obj(obj, ["geometry.coordinates"]) is not None and isinstance(_get_nested(obj, "geometry.coordinates"), list):
        coords = _get_nested(obj, "geometry.coordinates")
        if isinstance(coords, list) and len(coords) >= 2:
            # lon, lat
            obj.setdefault("lon", coords[0])
            obj.setdefault("lat", coords[1])
    out: Dict[str, Any] = {}
    for field in REQUIRED + OPTIONAL:
        val = _pick_from_obj(obj, ALIASES[field])
        out[field] = val
    # 必要字段缺失则启用启发式
    if any(out[k] is None for k in REQUIRED):
        return _heuristic_std(obj)
    return out


def convert_dir(src: Path, out: Path, verbose: bool = True) -> int:
    files = sorted(list(src.rglob("*.jsonl")) + list(src.rglob("*.json")) + list(src.rglob("*.geojson")))
    if not files:
        raise FileNotFoundError(f"未在 {src} 找到 json/jsonl/geojson")
    print(f"[INFO] 扫描目录: {src}，发现文件数: {len(files)}")

    rows: List[Dict[str, Any]] = []
    total = 0
    for idx, fp in enumerate(files, start=1):
        processed_before = total
        file_rows = 0
        print(f"[INFO] ({idx}/{len(files)}) 开始处理: {fp}")
        try:
            if fp.suffix.lower() == ".jsonl":
                it = _yield_records_from_jsonl(fp)
            else:
                it = _yield_records_from_json(fp)
            n = 0
            for rec in it:
                # 若对象包含 data 数组（如 {code:"", data:[{...}, {...}] }），将其中元素作为独立记录展开
                expanded = False
                if isinstance(rec, dict) and isinstance(rec.get("data"), list):
                    for item in rec.get("data"):
                        if not isinstance(item, dict):
                            continue
                        std = _std_from_obj(item)
                        if std is None:
                            # 对 item 启用父对象上下文启发式（扁平 keys like data[0].lat）
                            std = _std_from_obj(rec)
                        if std is None:
                            continue
                        rows.append(std)
                        file_rows += 1
                        n += 1
                        if len(rows) >= 200_000:
                            total += _flush_rows(rows, out, append=(out.exists()))
                            print(f"[INFO]    flush 累计写出，总计行数: {total}")
                            rows.clear()
                        if n % 10000 == 0:
                            print(f"[PROG]    {fp.name}: 已解析 {n} 条（临时缓冲 {len(rows)} 条）")
                    expanded = True
                if expanded:
                    continue
                # 常规单对象
                std = _std_from_obj(rec)
                if std is None:
                    continue
                rows.append(std)
                file_rows += 1
                n += 1
                if len(rows) >= 200_000:
                    total += _flush_rows(rows, out, append=(out.exists()))
                    print(f"[INFO]    flush 累计写出，总计行数: {total}")
                    rows.clear()
                if n % 10000 == 0:
                    print(f"[PROG]    {fp.name}: 已解析 {n} 条（临时缓冲 {len(rows)} 条）")
        except Exception as e:
            if verbose:
                print(f"[WARN] 跳过 {fp}: {e}")
            continue
        finally:
            print(f"[INFO] 完成 {fp.name}: 本文件解析 {file_rows} 条，累计总计 {processed_before + file_rows} 条（未含未flush缓冲）")
    if rows:
        total += _flush_rows(rows, out, append=(out.exists()))
        rows.clear()
    print(f"[OK] 转换完成，共写出 {total} 行到 {out}")
    return total


def _flush_rows(rows: List[Dict[str, Any]], out: Path, append: bool) -> int:
    df = pd.DataFrame(rows)
    df["mmsi"] = df["mmsi"].astype(str).str.strip()
    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce", dayfirst=True)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    if "sog" in df:
        df["sog"] = pd.to_numeric(df["sog"], errors="coerce")
    else:
        df["sog"] = np.nan
    if "cog" in df:
        df["cog"] = pd.to_numeric(df["cog"], errors="coerce")
    else:
        df["cog"] = np.nan
    df = df.dropna(subset=["mmsi", "time_utc", "lat", "lon"]).sort_values(["mmsi", "time_utc"])  # noqa: E712
    df["ts"] = (df["time_utc"].view("int64") // 10**9).astype("int64")

    # 以追加模式写 parquet（简化：读旧再覆写）
    if append and out.exists():
        old = pd.read_parquet(out)
        df = pd.concat([old, df], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[OK] flush -> {out} +{len(df)}")
    return int(len(df))


def main() -> None:
    ap = argparse.ArgumentParser(description="递归转换 AIS JSON -> Parquet")
    ap.add_argument("--src", default="ArcticRoute/data_raw/ais", help="源目录")
    ap.add_argument("--out", default="ArcticRoute/data_processed/ais/raw_auto.parquet", help="输出 Parquet 路径")
    args = ap.parse_args()
    src = Path(args.src)
    out = Path(args.out)
    total = convert_dir(src, out)
    print(f"[OK] 共写出 {total} 行到 {out}")


if __name__ == "__main__":
    main()
