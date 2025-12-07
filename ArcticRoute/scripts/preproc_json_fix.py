#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preprocess non-standard AIS JSON files into valid array JSON under data_raw/ais_fixed.

@role: pipeline
"""

"""
将 data_raw/ais 下的 JSON 预处理到 data_raw/ais_fixed：
- 标准 JSON/GeoJSON/JSONL 原样复制
- 无外层数组的对象拼接文件：用括号计数拆分为对象，并包裹为数组
"""
from __future__ import annotations
import json, re
from pathlib import Path

def ensure_array_json(txt: str) -> str:
    s = txt.strip()
    if not s:
        return "[]"
    # 已是数组
    if s.lstrip().startswith("["):
        return s
    # 通过括号计数切分对象
    out = []
    buf = []
    depth = 0
    started = False
    for ch in s:
        buf.append(ch)
        if ch == '{':
            depth += 1
            started = True
        elif ch == '}':
            depth -= 1
            if started and depth == 0:
                frag = ''.join(buf).strip().rstrip(',')
                buf.clear(); started = False
                try:
                    json.loads(frag)
                    out.append(frag)
                except Exception:
                    frag2 = re.sub(r",\s*$", "", frag)
                    try:
                        json.loads(frag2); out.append(frag2)
                    except Exception:
                        pass
    if out:
        return "[" + ",".join(out) + "]"
    # 兜底：尝试整体包裹 [] 并修复对象间逗号
    s2 = "[" + re.sub(r"}\s*,?\s*{", "},{", s.strip().strip(',')) + "]"
    try:
        json.loads(s2); return s2
    except Exception:
        return "[]"

def main():
    src = Path("ArcticRoute/data_raw/ais")
    dst = Path("ArcticRoute/data_raw/ais_fixed")
    dst.mkdir(parents=True, exist_ok=True)
    # 处理 .json
    for fp in src.rglob("*.json"):
        rel = fp.relative_to(src)
        outp = dst / rel
        outp.parent.mkdir(parents=True, exist_ok=True)
        try:
            txt = fp.read_text(encoding="utf-8", errors="ignore")
            fixed = ensure_array_json(txt)
            outp.write_text(fixed, encoding="utf-8")
        except Exception as e:
            print(f"[WARN] 跳过 {fp}: {e}")
    # 复制 .jsonl / .geojson
    for pat in ("*.jsonl", "*.geojson"):
        for fp in src.rglob(pat):
            rel = fp.relative_to(src)
            outp = dst / rel
            outp.parent.mkdir(parents=True, exist_ok=True)
            try:
                outp.write_text(fp.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
            except Exception as e:
                print(f"[WARN] 跳过 {fp}: {e}")
    print("[OK] 预处理完成 -> ArcticRoute/data_raw/ais_fixed")

if __name__ == "__main__":
    main()

