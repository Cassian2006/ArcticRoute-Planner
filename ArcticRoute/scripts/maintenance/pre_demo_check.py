#!/usr/bin/env python3
"""Pre-demo checklist runner; retained for historical reference.

@role: legacy
"""

"""
Pre-demo asset checklist for ArcticRoute.

Validates that key resources required for demos are present and coherent.
Outputs a status report to logs/pre_demo_check.txt.
The process exits with code 0 when all checks pass, otherwise 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = PROJECT_ROOT / "logs" / "pre_demo_check.txt"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OVERLAYS_DIR = OUTPUTS_DIR / "overlays"
HOTSPOTS_PATH = OUTPUTS_DIR / "acc_hotspots.geojson"
ACCIDENT_STATIC = PROJECT_ROOT / "data_processed" / "accident_density_static.nc"
ROUTE_VIEWER = PROJECT_ROOT / "web" / "route_viewer.html"


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: Optional[str] = None
    suggestion: Optional[str] = None


def check_route_viewer() -> CheckResult:
    if ROUTE_VIEWER.exists():
        return CheckResult("Route viewer HTML", True, str(ROUTE_VIEWER))
    suggestion = (
        "缺少 web/route_viewer.html。若已存在备份，可复制回位；否则参考资料重新生成。"
    )
    return CheckResult("Route viewer HTML", False, "web/route_viewer.html 未找到", suggestion)


def check_overlays() -> CheckResult:
    if not OVERLAYS_DIR.exists():
        return CheckResult("Overlay 图层", False, f"{OVERLAYS_DIR} 不存在", "执行 risk overlay 导出脚本重新生成")
    png_files = sorted(OVERLAYS_DIR.glob("risk_t*.png"))
    json_files = sorted(OVERLAYS_DIR.glob("risk_t*_bounds.json"))
    if not png_files or not json_files:
        return CheckResult(
            "Overlay 图层",
            False,
            f"缺少 {'PNG' if not png_files else 'bounds JSON'} 文件",
            "运行 python scripts/export_risk_overlay.py 生成热力图与边界",
        )
    png_tags = {path.stem for path in png_files}
    json_tags = {path.stem.replace("_bounds", "") for path in json_files}
    missing_bounds = sorted(png_tags - json_tags)
    missing_png = sorted(json_tags - png_tags)
    if missing_bounds or missing_png:
        detail_lines: List[str] = []
        if missing_bounds:
            detail_lines.append(f"下列 PNG 缺少 bounds：{', '.join(missing_bounds)}")
        if missing_png:
            detail_lines.append(f"下列 bounds JSON 缺少 PNG：{', '.join(missing_png)}")
        return CheckResult(
            "Overlay 图层",
            False,
            "; ".join(detail_lines),
            "重新运行 python scripts/export_risk_overlay.py --tidx <T> 生成匹配的 PNG/JSON",
        )
    count = len(png_files)
    return CheckResult("Overlay 图层", True, f"{count} 对 risk_t*.png / bounds.json 匹配")


def check_route_artifacts() -> CheckResult:
    if not OUTPUTS_DIR.exists():
        return CheckResult(
            "路线产物",
            False,
            f"{OUTPUTS_DIR} 不存在",
            "运行 python scripts/run_scenarios.py 或 api CLI 生成路线输出",
        )
    geojson_files = sorted(OUTPUTS_DIR.glob("route_*.geojson"))
    run_reports = sorted(OUTPUTS_DIR.glob("run_report_*.json"))
    if not geojson_files or not run_reports:
        missing = []
        if not geojson_files:
            missing.append("route_*.geojson")
        if not run_reports:
            missing.append("run_report_*.json")
        return CheckResult(
            "路线产物",
            False,
            f"缺少 {', '.join(missing)}",
            "执行 python scripts/run_scenarios.py 或 python scripts/route_astar_min.py 生成路线与报告",
        )
    # Check for at least one matching pair by tag.
    geo_stems = {path.stem.replace("route_", "", 1) for path in geojson_files}
    report_stems = {path.stem.replace("run_report_", "", 1) for path in run_reports}
    matched = geo_stems & report_stems
    if not matched:
        return CheckResult(
            "路线产物",
            False,
            "存在 route_*.geojson 和 run_report_*.json 但未找到匹配标签",
            "确认生成流程输出一致，或重新运行规划脚本以更新全部产物",
        )
    return CheckResult("路线产物", True, f"找到 {len(matched)} 组匹配的路线与报告")


def check_hotspots() -> CheckResult:
    if not ACCIDENT_STATIC.exists():
        return CheckResult("事故热点建议", True, "未检测到 accident_density_static.nc，跳过热点建议")
    if HOTSPOTS_PATH.exists():
        return CheckResult("事故热点建议", True, f"已存在 {HOTSPOTS_PATH}")
    suggestion = (
        "建议生成顶层热点文件：python scripts/export_acc_hotspots.py "
        "--nc data_processed/accident_density_static.nc "
        "--out outputs/acc_hotspots.geojson"
    )
    return CheckResult("事故热点建议", False, "未找到 outputs/acc_hotspots.geojson", suggestion)


def render_report(results: List[CheckResult]) -> List[str]:
    lines = [
        f"Pre-demo checklist - {PROJECT_ROOT}",
        "",
    ]
    for item in results:
        status = "OK" if item.passed else "MISSING"
        lines.append(f"[{status}] {item.name}")
        if item.details:
            lines.append(f"  {item.details}")
        if not item.passed and item.suggestion:
            lines.append(f"  建议：{item.suggestion}")
        lines.append("")
    issues = [item for item in results if not item.passed]
    if issues:
        lines.append("总体状态：存在待修复项")
    else:
        lines.append("总体状态：通过所有检查")
    return lines


def write_report(lines: List[str]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    results = [
        check_route_viewer(),
        check_overlays(),
        check_route_artifacts(),
        check_hotspots(),
    ]
    report_lines = render_report(results)
    write_report(report_lines)
    print("\n".join(report_lines))
    return 1 if any(not r.passed for r in results if r.name != "事故热点建议") else 0


if __name__ == "__main__":
    raise SystemExit(main())

