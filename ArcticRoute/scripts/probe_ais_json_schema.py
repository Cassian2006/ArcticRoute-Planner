#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sample and list candidate keys from AIS JSON to infer mmsi/time/lat/lon schema.

@role: analysis
"""

"""
递归抽样 ArcticRoute/data_raw/ais 下 JSON/JSONL/GeoJSON 的字段，
尝试识别 mmsi / time / lat / lon 候选键并打印统计。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    import ijson  # type: ignore
except Exception:
    ijson = None  # type: ignore


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def iter_json(path: Path) -> Iterable[Dict[str, Any]]:
    # 尝试流式
    if ijson is not None:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                n = 0
                for rec in ijson.items(f, 'item'):
                    yield rec
                    n += 1
                    if n >= 1000:
                        return
        except Exception:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    n = 0
                    for key in ('features','records','items','data'):
                        f.seek(0)
                        for rec in ijson.items(f, f'{key}.item'):
                            yield rec
                            n += 1
                            if n >= 1000:
                                return
            except Exception:
                pass
    # 回退：一次性加载（限前 1000 条）
    try:
        obj = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return
    if isinstance(obj, list):
        for rec in obj[:1000]:
            if isinstance(rec, dict):
                yield rec
    elif isinstance(obj, dict):
        for key in ('features','records','items','data'):
            if key in obj and isinstance(obj[key], list):
                for rec in obj[key][:1000]:
                    if isinstance(rec, dict):
                        yield rec
                return
        # 单对象
        yield obj


def flatten(d: Any, prefix: str = '') -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f'{prefix}.{k}' if prefix else str(k)
            out.update(flatten(v, key))
    elif isinstance(d, list):
        out[prefix] = d
        for i, v in enumerate(d[:2]):
            out.update(flatten(v, f'{prefix}[{i}]'))
    else:
        out[prefix] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='ArcticRoute/data_raw/ais')
    args = ap.parse_args()
    src = Path(args.src)
    files = sorted(list(src.rglob('*.jsonl')) + list(src.rglob('*.json')) + list(src.rglob('*.geojson')))
    if not files:
        print('[ERR] 未找到 JSON 文件')
        return
    key_counts: Dict[str, int] = {}
    examples: Dict[str, Any] = {}
    scanned = 0
    for fp in files:
        it = iter_jsonl(fp) if fp.suffix.lower()=='.jsonl' else iter_json(fp)
        cnt = 0
        for rec in it:
            flat = flatten(rec)
            for k, v in flat.items():
                key_counts[k] = key_counts.get(k, 0) + 1
                if k not in examples:
                    examples[k] = v
            cnt += 1
            if cnt >= 100:
                break
        scanned += 1
        if scanned >= 50:
            break
    # 打印最常见的包含关键子串的键
    def top_keys(substrs: Tuple[str, ...], topn=20):
        items = [(k, c) for k, c in key_counts.items() if any(s in k.lower() for s in substrs)]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:topn]
    print('=== 候选 mmsi 相关键 ===')
    for k, c in top_keys(("mmsi","ship")):
        print(c, k, '->', str(examples.get(k))[:80])
    print('=== 候选 时间 相关键 ===')
    for k, c in top_keys(("time","date")):
        print(c, k, '->', str(examples.get(k))[:80])
    print('=== 候选 纬度 相关键 ===')
    for k, c in top_keys(("lat","y")):
        print(c, k, '->', str(examples.get(k))[:80])
    print('=== 候选 经度 相关键 ===')
    for k, c in top_keys(("lon","lng","x")):
        print(c, k, '->', str(examples.get(k))[:80])

if __name__ == '__main__':
    main()

