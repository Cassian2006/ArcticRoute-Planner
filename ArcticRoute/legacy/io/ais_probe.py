"""AIS JSON 架构探测（Schema Probe）

目标：在不加载全量的前提下，抽样各 JSON 文件的字段名/示例值与差异，生成字段映射建议。

约束：
- 先搜索再改动：本模块为新增文件，独立实现，不影响现有流程。
- CLI 必须支持 --dry-run：提供 generate_reports(..., dry_run=True)。
- 所有写盘走 register_artifact()：仅在非 dry-run 时写盘并登记。
- Windows 路径用 os.path.join。
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from ArcticRoute.cache.index_util import register_artifact
except Exception:  # 允许独立调用时缺失注册能力
    register_artifact = None  # type: ignore


CanonicalKeys = [
    "mmsi", "ts", "lat", "lon", "sog", "cog", "heading", "vessel_type", "loa", "beam", "nav_status",
]

# 常见键名归一规则：小写，去下划线/空格；特殊替换
def _norm_key(k: str) -> str:
    k2 = (k or "").strip()
    k2 = k2.replace(" ", "_").replace("-", "_")
    k2 = k2.lower()
    return k2

_SYNONYM_MAP = {
    # 经纬度
    "latitude": "lat", "lat": "lat", "lat_dd": "lat", "y": "lat",
    "longitude": "lon", "lon": "lon", "lng": "lon", "x": "lon",
    "lon_dd": "lon", "long": "lon", "e": "lon",
    "lat_deg": "lat", "lon_deg": "lon",
    "latd": "lat", "lond": "lon", "lat_": "lat", "lon_": "lon",
    # 时间
    "time": "ts", "timestamp": "ts", "datetime": "ts", "pos_time": "ts",
    "base_date_time": "ts", "recvtime": "ts", "event_time": "ts",
    # 运动姿态
    "speed": "sog", "sog": "sog",
    "course": "cog", "cog": "cog", "true_heading": "heading", "heading": "heading",
    # 船舶信息
    "ship_type": "vessel_type", "vessel_type": "vessel_type", "type": "vessel_type",
    "length": "loa", "loa": "loa", "beam": "beam", "breadth": "beam",
    "status": "nav_status", "nav_status": "nav_status", "navigational_status": "nav_status",
    # 标识
    "mmsi": "mmsi",
}

# 进一步处理大小写与缩写（如 LON/Longitude）
_DEF_VARIANTS = {
    "lat": {"latitude", "lat", "LAT", "Latitude"},
    "lon": {"longitude", "lon", "LON", "Longitude", "LONG"},
    "ts": {"timestamp", "Timestamp", "time", "Time", "DATETIME", "datetime"},
}


@dataclass
class KeyStat:
    count: int
    examples: List[Any]


def _try_parse_json_line(line: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _iter_objects_from_file(path: str, max_lines: int) -> List[Dict[str, Any]]:
    objs: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            # 优先按 NDJSON 逐行解析
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                rec = _try_parse_json_line(line.strip())
                if rec is not None:
                    objs.append(rec)
            if objs:
                return objs
    except Exception:
        return objs
    # 若非 NDJSON，尝试读取小块并解析为数组
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = f.read(1024 * 512)  # 512KB 以内
        arr = json.loads(head)
        if isinstance(arr, list):
            for it in arr[: max_lines]:
                if isinstance(it, dict):
                    objs.append(it)
        return objs
    except Exception:
        return objs


def probe_json_schema(root_dir: str, max_files: int = 50, max_lines: int = 200) -> Dict[str, Any]:
    """在 root_dir 下随机抽样若干 JSON 文件，每个文件取前 N 行/项，统计键频次与示例。

    返回：{"files_scanned": int, "keys": {key: {"count": int, "examples": []}}, "suggest_keymap": {...}}
    """
    root = root_dir
    all_json_files: List[str] = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".json"):
                all_json_files.append(os.path.join(dp, fn))
    if not all_json_files:
        return {"files_scanned": 0, "keys": {}, "suggest_keymap": {}}
    random.shuffle(all_json_files)
    pick = all_json_files[: max_files]

    key_stats: Dict[str, KeyStat] = {}
    def _bump(k: str, v: Any) -> None:
        st = key_stats.get(k)
        if st is None:
            st = KeyStat(count=0, examples=[])
            key_stats[k] = st
        st.count += 1
        if len(st.examples) < 5:
            st.examples.append(v)

    for fp in pick:
        objs = _iter_objects_from_file(fp, max_lines=max_lines)
        for obj in objs:
            for k, v in obj.items():
                _bump(k, v)

    # 生成建议 keymap：按规范小写化与同义词归并
    suggest: Dict[str, str] = {}
    for raw_key in key_stats.keys():
        nk = _norm_key(raw_key)
        if nk in _SYNONYM_MAP:
            suggest[raw_key] = _SYNONYM_MAP[nk]
            continue
        # 变体匹配（如 LON/LAT 等）
        mapped = None
        for cano, variants in _DEF_VARIANTS.items():
            if raw_key in variants:
                mapped = cano
                break
        if mapped is not None:
            suggest[raw_key] = mapped
            continue
        # 简单启发：完全小写后直接命中 canonical keys
        if nk in CanonicalKeys:
            suggest[raw_key] = nk

    keys_block = {k: asdict(v) for k, v in key_stats.items()}
    return {
        "files_scanned": len(pick),
        "keys": keys_block,
        "suggest_keymap": suggest,
    }


def generate_reports(root_dir: str, out_dir: str, run_id: str, dry_run: bool = True) -> Dict[str, str]:
    """生成 ais_schema.json 与 ais_keymap_suggest.json，遵循 dry-run 与登记规则。

    返回：{"schema": path_or_preview, "keymap": path_or_preview}
    """
    result = probe_json_schema(root_dir=root_dir)
    os.makedirs(out_dir, exist_ok=True)
    schema_path = os.path.join(out_dir, "ais_schema.json")
    keymap_path = os.path.join(out_dir, "ais_keymap_suggest.json")

    outputs: Dict[str, str] = {}
    if dry_run:
        # 只返回文件将要写入的位置与统计摘要，不落盘
        outputs["schema"] = schema_path
        outputs["keymap"] = keymap_path
        return outputs

    # 非 dry-run：写盘并登记
    with open(schema_path, "w", encoding="utf-8") as fw:
        json.dump(result, fw, ensure_ascii=False, indent=2)
    with open(keymap_path, "w", encoding="utf-8") as fw:
        json.dump(result.get("suggest_keymap", {}), fw, ensure_ascii=False, indent=2)

    if register_artifact is not None:
        try:
            register_artifact(run_id=run_id, kind="recon_ais_schema", path=schema_path, attrs={})
            register_artifact(run_id=run_id, kind="recon_ais_keymap", path=keymap_path, attrs={})
        except Exception:
            pass

    outputs["schema"] = schema_path
    outputs["keymap"] = keymap_path
    return outputs


__all__ = ["probe_json_schema", "generate_reports"]

