"""A-20: Recon Summary Report

聚合 A-01~A-19 的产物，生成 Phase B 前置 RECON_SUMMARY.md。
- 优先读取 reports/recon 下的各类产物（若缺失则跳过并标注）
- 汇总：网格/时间轴约定、路径/缓存/flags、可复用函数、CLI 占位、缺失点/建议
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
RECON_DIR = REPO_ROOT / "reports" / "recon"
OUT_PATH = RECON_DIR / "RECON_SUMMARY.md"


def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _safe_text(p: Path, n_lines: int = 400) -> str:
    try:
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="ignore")
            return "\n".join(txt.splitlines()[:n_lines])
    except Exception:
        return ""
    return ""


def main() -> None:
    RECON_DIR.mkdir(parents=True, exist_ok=True)

    # 基础输入
    health_json = _read_json(RECON_DIR / "health_summary.json") or {}
    health_md = _safe_text(RECON_DIR / "health_summary.md")
    health_extras = _read_json(RECON_DIR / "health_extras.json") or {}
    grid_spec = _read_json(RECON_DIR / "grid_spec.json") or _read_json(RECON_DIR / "grid_spec.json") or {}
    ui_map = _read_json(RECON_DIR / "ui_map.json") or {}
    report_slots_md = _safe_text(RECON_DIR / "report_slots.md")
    p1_reg_md = _safe_text(RECON_DIR / "p1_regression.md")
    dryrun_samples_md = _safe_text(RECON_DIR / "dryrun_samples.md")

    # 结构与 CLI 地图（来自 recon_scan）
    structure = _read_json(RECON_DIR / "structure.json") or {}
    cli_map = _read_json(RECON_DIR / "cli_map.json") or {}
    gaps = _read_json(RECON_DIR / "gaps.json") or {}

    # paths.yaml 与 feature_flags.yaml
    paths_yaml = (REPO_ROOT / "configs" / "paths.yaml")
    flags_yaml = (REPO_ROOT / "configs" / "feature_flags.yaml")

    # 版本
    app_ver = (health_extras or {}).get("version")

    # 建议的网格/时间轴/路径约定
    contract = grid_spec.get("contract", {}) if isinstance(grid_spec, dict) else {}
    grid_res = contract.get("grid_resolution", {}) if isinstance(contract, dict) else {}
    time_info = (grid_spec.get("sic_summary") or {}).get("time") if isinstance(grid_spec, dict) else None
    freq = (time_info or {}).get("freq") if isinstance(time_info, dict) else None

    # 可复用函数清单
    reusable = [
        "ArcticRoute.io.align.ensure_common_grid",
        "ArcticRoute.io.align.align_time",
        "ArcticRoute.cache.index_util.register_artifact",
        "ArcticRoute.cache.index_util.find_artifacts",
    ]

    # CLI 占位
    placeholders = [
        "features.build --dry-run",
        "risk.build --dry-run",
        "prior.build --dry-run",
        "congest.build --dry-run",
        "health.check",
    ]

    # 缺失点清单（基于 recon_scan 的 gaps.json）
    gaps_list = []
    if isinstance(gaps, dict):
        for k in ("warnings", "risks", "suggestions"):
            vals = gaps.get(k) or []
            if isinstance(vals, list):
                for v in vals:
                    gaps_list.append(str(v))

    lines = []
    lines.append(f"# Recon Summary (Phase A)\n")
    lines.append(f"- App Version: {app_ver or 'N/A'}\n")

    lines.append("## Phase B 所需输入就绪情况\n")
    lines.append("- grid_spec: OK" if grid_spec else "- grid_spec: MISSING")
    lines.append("- paths.yaml: OK" if paths_yaml.exists() else "- paths.yaml: MISSING")
    lines.append("- feature_flags.yaml: OK" if flags_yaml.exists() else "- feature_flags.yaml: MISSING")
    lines.append("- CLI 占位: features/risk/prior/congest.build 已就绪")
    lines.append("")

    lines.append("## 建议的网格/时间轴/路径约定\n")
    lines.append(f"- grid.type: {grid_res.get('type')!r}\n")
    lines.append(f"- grid.resolution: lat={grid_res.get('lat')} (~{grid_res.get('lat_deg')}°), lon={grid_res.get('lon')} (~{grid_res.get('lon_deg')}°)\n")
    lines.append(f"- time.freq: {freq!r}\n")
    lines.append("")

    lines.append("## 可复用函数清单\n")
    for item in reusable:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## CLI 占位与 dry-run 样例\n")
    for c in placeholders:
        lines.append(f"- {c}")
    if dryrun_samples_md:
        lines.append("\n<details><summary>dry-run samples</summary>\n\n" + dryrun_samples_md + "\n\n</details>\n")

    lines.append("## UI 面板与导出能力\n")
    if ui_map:
        tabs = (ui_map.get("structure") or {}).get("tabs")
        exports = ui_map.get("exports") or []
        lines.append(f"- tabs: {tabs}")
        lines.append(f"- exports: {[e.get('label') for e in exports]}")
    else:
        lines.append("- ui_map: MISSING")
    lines.append("")

    lines.append("## 报告插槽（可扩展点）\n")
    if report_slots_md:
        lines.append("\n<details><summary>slots</summary>\n\n" + report_slots_md + "\n\n</details>\n")
    else:
        lines.append("- report_slots: MISSING")

    lines.append("## 健康检查摘要\n")
    if health_json:
        lines.append(f"- all_ok: {health_json.get('all_ok')}")
        lines.append(f"- repo: {health_json.get('repo_root')}")
    else:
        lines.append("- health_summary.json: MISSING")
    if health_md:
        lines.append("\n<details><summary>health.md</summary>\n\n" + health_md + "\n\n</details>\n")

    lines.append("## P1 回归核验（只读）\n")
    if p1_reg_md:
        lines.append("\n<details><summary>p1_regression</summary>\n\n" + p1_reg_md + "\n\n</details>\n")
    else:
        lines.append("- p1_regression.md: MISSING")

    lines.append("## 缺失点与建议\n")
    if gaps_list:
        for g in gaps_list[:50]:
            lines.append(f"- {g}")
    else:
        lines.append("- gaps: (none)")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Recon summary written to: {OUT_PATH}")


if __name__ == "__main__":
    main()














