#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

try:
    from ArcticRoute.cache.index_util import register_artifact
except Exception:
    register_artifact = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_accept_markdown(ym: str, summary_json: str, png_path: str) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    summary: Dict[str, Any] = {}
    try:
        if os.path.exists(summary_json):
            with open(summary_json, "r", encoding="utf-8") as f:
                summary = json.load(f)
    except Exception:
        summary = {}
    dims = summary.get("dims") or {}
    coverage = summary.get("coverage_pct")
    nonzero = summary.get("nonzero")
    maxv = summary.get("max")
    meanv = summary.get("mean")

    lines = []
    lines.append(f"# Phase B 验收（{ym}）")
    lines.append("")
    lines.append(f"生成时间: {ts}")
    lines.append("")
    lines.append("## QA 指标摘要")
    lines.append("")
    lines.append(f"- 维度: {dims}")
    lines.append(f"- 非零格点: {nonzero}")
    lines.append(f"- 最大值: {maxv}")
    lines.append(f"- 均值: {meanv}")
    lines.append(f"- 覆盖率(%): {coverage}")
    lines.append("")
    if os.path.exists(png_path):
        # 在 Markdown 中引用图片相对路径
        relp = os.path.relpath(png_path, start=os.path.join(str(REPO_ROOT)))
        lines.append("## 样例图")
        lines.append("")
        lines.append(f"![features_{ym}]({relp.replace('\\\\','/')})")
        lines.append("")
    lines.append("## 产物")
    lines.append("")
    lines.append("- " + os.path.relpath(summary_json, start=str(REPO_ROOT)).replace('\\\\','/'))
    if os.path.exists(png_path):
        lines.append("- " + os.path.relpath(png_path, start=str(REPO_ROOT)).replace('\\\\','/'))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 Phase B 验收报告 (ACCEPT_YYYYMM.md)")
    ap.add_argument("--ym", required=True, help="月份 YYYYMM")
    ap.add_argument("--dry-run", action="store_true")
    ns = ap.parse_args()

    ym = str(ns.ym)
    out_dir = os.path.join(str(REPO_ROOT), "reports", "phaseB")
    os.makedirs(out_dir, exist_ok=True)
    out_md = os.path.join(out_dir, f"ACCEPT_{ym}.md")

    features_summary_json = os.path.join(str(REPO_ROOT), "outputs", f"features_summary_{ym}.json")
    features_png = os.path.join(str(REPO_ROOT), "outputs", f"features_{ym}.png")

    md = make_accept_markdown(ym, features_summary_json, features_png)
    if ns.dry_run:
        print(md)
        return 0

    with open(out_md, "w", encoding="utf-8") as fw:
        fw.write(md)
    if register_artifact is not None:
        try:
            register_artifact(run_id=datetime.utcnow().strftime("%Y%m%dT%H%M%S"), kind="phaseB_accept", path=out_md, attrs={"ym": ym})
        except Exception:
            pass
    print(json.dumps({"accept_md": out_md}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

