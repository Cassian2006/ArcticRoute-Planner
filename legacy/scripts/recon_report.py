"""A-10: Report System Recon (只读)

- 优先解析最新 reports/run_report_*.html：提取章节(h1/h2)与可插槽点。
- 若无 HTML，则静态解析 ArcticRoute/api/cli.py 的报告生成逻辑，基于 sec(title, body) 与关键词推断插槽点。
- 输出 reports/recon/report_slots.md。
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import List, Tuple


def _latest_report_html(reports_dir: Path) -> Path | None:
    pats = [str(reports_dir / "run_report_*.html"), str(reports_dir / "*.html")]
    cands: List[Tuple[float, Path]] = []
    for pat in pats:
        for p in glob.glob(pat):
            try:
                st = os.stat(p)
                cands.append((st.st_mtime, Path(p)))
            except Exception:
                continue
    if not cands:
        return None
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]


ess_h_tag = re.compile(r"<h([12])[^>]*>(.*?)</h\1>", re.I | re.S)

def _extract_sections_from_html(html_text: str) -> List[str]:
    titles: List[str] = []
    for m in ess_h_tag.finditer(html_text):
        title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        if title:
            titles.append(title)
    # 去重，保持顺序
    seen = set()
    uniq = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _analyze_cli_fallback(cli_path: Path) -> List[str]:
    try:
        text = cli_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    # 寻找 sec("Title", ...) 样式
    titles: List[str] = []
    for m in re.finditer(r"sec\(\s*\"([^\"]+)\"\s*,", text):
        titles.append(m.group(1))
    # 常见关键词兜底
    keywords = ["Health", "Metrics", "Route", "Repro", "Warning", "环境", "参数", "显著性", "区域", "地图", "时间序列"]
    for kw in keywords:
        if kw not in titles and re.search(kw, text, re.I):
            titles.append(kw)
    # 去重
    seen = set(); uniq = []
    for t in titles:
        if t not in seen:
            seen.add(t); uniq.append(t)
    return uniq


def _slot_suggestions(section_titles: List[str]) -> List[Tuple[str, str]]:
    """基于已存在章节名建议插槽位置。"""
    slots: List[Tuple[str, str]] = []
    # 风险层贡献：优先插在 Metrics/区域统计/显著性 后面
    anchor_metrics = next((t for t in section_titles if re.search(r"(Metrics|区域|显著|eval|统计)", t, re.I)), None)
    if anchor_metrics:
        slots.append((anchor_metrics, "风险层贡献（risk/prior/congest对整体指标的影响拆解）"))
    else:
        slots.append(("Metrics", "风险层贡献（risk/prior/congest对整体指标的影响拆解）"))
    # 候选路线比较：优先插在 Route 或 地图 可视化章节附近
    anchor_route = next((t for t in section_titles if re.search(r"(Route|路线|路径|地图)", t, re.I)), None)
    if anchor_route:
        slots.append((anchor_route, "候选路线比较（多起止/多参数/多代价层对比）"))
    else:
        slots.append(("Route", "候选路线比较（多起止/多参数/多代价层对比）"))
    # Repro：保留在末尾或参数/环境章节附近
    anchor_repro = next((t for t in section_titles if re.search(r"(参数|环境|Repro|复现)", t, re.I)), None)
    if anchor_repro:
        slots.append((anchor_repro, "复现实验配置（paths/config/hash/env）"))
    else:
        slots.append(("Repro", "复现实验配置（paths/config/hash/env）"))
    # Warning：靠近 Warning/Health
    anchor_warn = next((t for t in section_titles if re.search(r"(Warn|警告|Health|健康)", t, re.I)), None)
    if anchor_warn:
        slots.append((anchor_warn, "告警与健康检查摘要"))
    else:
        slots.append(("Warnings", "告警与健康检查摘要"))
    return slots


def main():
    parser = argparse.ArgumentParser(description="Recon report system and extract slots")
    repo_root = Path(__file__).resolve().parents[1]
    reports_dir = repo_root / "reports"
    cli_path = repo_root / "ArcticRoute" / "api" / "cli.py"
    out_path = reports_dir / "recon" / "report_slots.md"

    rpt = _latest_report_html(reports_dir)
    titles: List[str] = []
    if rpt and rpt.exists():
        text = rpt.read_text(encoding="utf-8", errors="ignore")
        titles = _extract_sections_from_html(text)
        src = f"HTML: {rpt.name}"
    else:
        titles = _analyze_cli_fallback(cli_path)
        src = f"CLI: {cli_path.name} (fallback)"

    slots = _slot_suggestions(titles)

    lines = [
        "# Report Slots Recon",
        "",
        f"Source: {src}",
        "",
        "## Detected Sections",
    ]
    for t in titles:
        lines.append(f"- {t}")
    lines.append("")
    lines.append("## Suggested Insertion Points")
    for anchor, desc in slots:
        lines.append(f"- After/Under '{anchor}': {desc}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report slots saved to: {out_path}")


if __name__ == "__main__":
    main()

