"""AIS 规范化映射（KeyMap）

- normalize_record(raw: dict, keymap: dict) -> dict | None
- 时间统一为 UTC 秒（ts:int）；支持 iso8601、epoch_ms、epoch_s
- 经纬度范围校验（lat: -90..90, lon: -180..180）
- 关键字段缺失或非法返回 None
- 允许从 B-01 产物加载 keymap（reports/recon/ais_keymap_suggest.json），也允许调用方传入覆盖

约束：
- 不写盘（除非未来扩展 CLI 导出，届时必须支持 --dry-run 并使用 register_artifact）
- Windows 路径统一使用 os.path.join
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime, timezone

_CANONICAL_ORDER = [
    "mmsi",
    "ts",
    "lat",
    "lon",
    "sog",
    "cog",
    "heading",
    "vessel_type",
    "loa",
    "beam",
    "nav_status",
]


def _coerce_int(val: Any) -> Optional[int]:
    try:
        if isinstance(val, bool):
            return None
        if isinstance(val, (int,)):
            return int(val)
        if isinstance(val, float):
            return int(val)
        if isinstance(val, str) and val.strip():
            return int(float(val.strip()))
    except Exception:
        return None
    return None


def _parse_ts_to_epoch_seconds(v: Any) -> Optional[int]:
    """解析时间为 UTC 秒。
    支持：
    - epoch_ms: 整型/字符串，长度>=13 或 数值>1e12
    - epoch_s: 整型/字符串
    - iso8601: 形如 '2024-01-02T03:04:05Z'、'2024-01-02 03:04:05+00:00'
    """
    # 1) 直接数字
    iv = _coerce_int(v)
    if iv is not None:
        # 判断 ms 级
        if iv > 10**12:
            return iv // 1000
        # 认为是秒
        return iv

    # 2) 字符串形式的 ISO8601
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        # 常见的 Z 或 无时区形式
        try:
            s2 = s.replace("Z", "+00:00").replace(" ", "T")
            dt = datetime.fromisoformat(s2)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except Exception:
            # 兜底：常见格式再尝试一遍
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    dt = datetime.strptime(s.split(".")[0], fmt).replace(tzinfo=timezone.utc)
                    return int(dt.timestamp())
                except Exception:
                    pass
    return None


def _get_with_keymap(raw: Dict[str, Any], keymap: Dict[str, str], canonical_key: str) -> Any:
    """根据 keymap 查找 canonical_key 对应的原始键名，并返回值。
    匹配策略：
    - 先找显式映射（值==canonical_key 的原始 key）
    - 再尝试不区分大小写匹配原始键
    - 再尝试去下划线/横线/空格并小写后匹配
    """
    # 1) 显式映射
    for raw_key, mapped in keymap.items():
        if mapped == canonical_key and raw_key in raw:
            return raw.get(raw_key)
    # 2) 不区分大小写直接命中
    lower_index = {k.lower(): k for k in raw.keys()}
    if canonical_key in lower_index:
        return raw.get(lower_index[canonical_key])
    # 3) 归一化再命中
    def norm(s: str) -> str:
        return s.replace(" ", "_").replace("-", "_").lower()
    norm_index = {norm(k): k for k in raw.keys()}
    if canonical_key in norm_index:
        return raw.get(norm_index[canonical_key])
    return None


def normalize_record(raw: Dict[str, Any], keymap: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """将原始记录字段归一化为规范列；非法/缺关键字段返回 None。

    关键字段：mmsi, ts, lat, lon
    """
    out: Dict[str, Any] = {}

    # mmsi
    mmsi = _get_with_keymap(raw, keymap, "mmsi")
    mmsi_i = _coerce_int(mmsi)
    if not mmsi_i or mmsi_i <= 0:
        return None
    out["mmsi"] = mmsi_i

    # 时间 ts
    ts_v = _get_with_keymap(raw, keymap, "ts")
    ts_i = _parse_ts_to_epoch_seconds(ts_v)
    if ts_i is None or ts_i <= 0:
        return None
    out["ts"] = ts_i

    # 经纬度
    lat_v = _get_with_keymap(raw, keymap, "lat")
    lon_v = _get_with_keymap(raw, keymap, "lon")
    try:
        lat_f = float(lat_v)
        lon_f = float(lon_v)
    except Exception:
        return None
    if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0):
        return None
    out["lat"] = lat_f
    out["lon"] = lon_f

    # 可选字段
    def _flt(x: Any) -> Optional[float]:
        try:
            if x is None or x == "":
                return None
            return float(x)
        except Exception:
            return None

    for k in ("sog", "cog", "heading", "loa", "beam"):
        v = _get_with_keymap(raw, keymap, k)
        fv = _flt(v)
        if fv is not None:
            out[k] = fv

    for k in ("vessel_type", "nav_status"):
        v = _get_with_keymap(raw, keymap, k)
        if v is not None and v != "":
            out[k] = v

    # 保持规范列顺序（非强制，只用于整洁性）
    ordered = {k: out[k] for k in _CANONICAL_ORDER if k in out}
    # 追加额外键
    for k, v in out.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def load_keymap(default_root: Optional[str] = None) -> Dict[str, str]:
    """加载 B-01 产物的键映射建议；文件缺失时返回空映射。"""
    root = default_root or os.path.join(os.getcwd(), "reports", "recon")
    path = os.path.join(root, "ais_keymap_suggest.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                # 约定：形如 {"Longitude":"lon", ...}
                return {str(k): str(v) for k, v in data.items()}
    except Exception:
        return {}
    return {}


__all__ = ["normalize_record", "load_keymap"]

