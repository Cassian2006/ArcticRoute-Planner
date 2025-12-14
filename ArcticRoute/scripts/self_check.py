# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from ArcticRoute.diagnostics.self_check import (
    run_all_checks as _run_all_checks_full,
    run_config_checks,
    run_data_checks,
    run_planning_checks,
    run_history_checks,
    run_ui_checks,
    render_report_markdown,
    save_to_reports,
    save_structured_json,
    Section,
)


def run_endpoints_checks() -> Section:
    from ArcticRoute.diagnostics.self_check import Item
    try:
        from ArcticRoute.core import planner_service as ps
    except Exception as e:
        return Section("起终点与规划域自检", "error", [Item("planner_service", "ERROR", f"导入失败: {e}")])

    items = []
    # 加载一个标准环境
    try:
        root = ps.get_project_root()
        scn_file = root / "configs" / "scenarios.yaml"
        ym = "202412"
        if scn_file.exists():
            import yaml
            obj = yaml.safe_load(scn_file.read_text(encoding="utf-8")) or {}
            for s in (obj.get("scenarios") or []) or []:
                ym = str(s.get("ym", ym))
                break
    except Exception:
        ym = "202412"

    try:
        env_ctx = ps.load_environment(ym=ym, w_ice=1.0, w_accident=1.0, prior_weight=0.3, profile_name="balanced")
        dom = getattr(env_ctx, "domain", None)
        if dom is None:
            items.append(Item("planning_domain", "WARN", "缺失 domain（已回退 env 网格）"))
        else:
            items.append(Item("planning_domain", "OK", f"lat=[{dom.lat_min:.3f},{dom.lat_max:.3f}], lon=[{dom.lon_min:.3f},{dom.lon_max:.3f}]"))
    except Exception as e:
        return Section("起终点与规划域自检", "error", [Item("load_environment", "ERROR", f"失败: {e}")])

    # 默认起终点
    try:
        (s_lat, s_lon), (g_lat, g_lon) = ps.get_default_start_end(env_ctx)
        def _in(dom, la, lo):
            return (dom is None) or (dom.lat_min <= la <= dom.lat_max and dom.lon_min <= lo <= dom.lon_max)
        ok_in = _in(dom, s_lat, s_lon) and _in(dom, g_lat, g_lon)
        # 简单落陆检查
        lm = getattr(env_ctx, "land_mask", None)
        ocean_ok = True
        if lm is not None:
            sij = ps.latlon_to_ij(env_ctx, s_lat, s_lon)
            gij = ps.latlon_to_ij(env_ctx, g_lat, g_lon)
            if sij and gij:
                ocean_ok = (not bool(lm[sij[0], sij[1]])) and (not bool(lm[gij[0], gij[1]]))
        items.append(Item("default_points", "OK" if (ok_in and ocean_ok) else "WARN", f"start=({s_lat:.3f},{s_lon:.3f}), end=({g_lat:.3f},{g_lon:.3f})"))
    except Exception as e:
        items.append(Item("default_points", "ERROR", f"生成失败: {e}"))

    # 随机点采样
    try:
        pts = ps.sample_random_ocean_points(env_ctx, n=10)
        items.append(Item("random_points", "OK" if pts and len(pts) >= 2 else "WARN", f"样本数={len(pts) if pts else 0}"))
    except Exception as e:
        items.append(Item("random_points", "WARN", f"采样失败: {e}"))

    # 规划一次并检查端点未被拉到边界
    try:
        sij = ps.latlon_to_ij(env_ctx, s_lat, s_lon)
        gij = ps.latlon_to_ij(env_ctx, g_lat, g_lon)
        if not sij or not gij:
            raise RuntimeError("默认起止点无法映射到网格")
        env2, rr = ps.run_planning_pipeline(
            ym=ym,
            start_ij=sij, goal_ij=gij,
            w_ice=1.0, w_accident=1.0, prior_weight=0.3,
            allow_diagonal=True, heuristic="euclidean", eco_enabled=False, profile_name="balanced",
        )
        H, W = env2.cost_da.shape[-2:]
        def _not_on_border(ij):
            i, j = int(ij[0]), int(ij[1])
            return (i > 0 and i < H-1 and j > 0 and j < W-1)
        ok_border = True
        if rr and rr.path_ij:
            ok_border = _not_on_border(rr.path_ij[0]) and _not_on_border(rr.path_ij[-1])
        items.append(Item("route_endpoints", "OK" if ok_border else "WARN", f"steps={len(rr.path_ij) if rr else 0}"))
    except Exception as e:
        items.append(Item("route_endpoints", "WARN", f"检查失败: {e}"))

    status = "ok"
    if any(it.level == "ERROR" for it in items):
        status = "error"
    elif any(it.level == "WARN" for it in items):
        status = "warn"
    return Section("起终点与规划域自检", status, items)


def run_all_checks(focus: str = "all") -> Tuple[List[Section], Path]:
    """
    支持按 focus 选择性运行：
      - all: 运行所有小节（默认，与原行为一致）
      - prior: 仅运行“历史航线与拥挤风险自检”小节
      - corridor: 等价于 prior（corridor 属于 prior 范畴）
      - eco: 仅运行“路线规划管线自检”（包含 Eco 差异与成本分解）
    返回 (sections, report_path)
    """
    focus = (focus or "all").lower()
    if focus == "all":
        return _run_all_checks_full()

    sections: List[Section] = []
    if focus in ("prior", "corridor"):
        try:
            sections.append(run_history_checks())
        except Exception as e:
            from ArcticRoute.diagnostics.self_check import Item
            sections.append(Section("4. 历史航线与拥挤风险自检", "error", [Item("history", "ERROR", str(e))]))
    elif focus in ("eco", "plan", "planning"):
        try:
            sections.append(run_planning_checks())
        except Exception as e:
            from ArcticRoute.diagnostics.self_check import Item
            sections.append(Section("3. 路线规划管线自检", "error", [Item("planning", "ERROR", str(e))]))
    elif focus in ("endpoints", "points", "domain"):
        try:
            sections.append(run_endpoints_checks())
        except Exception as e:
            from ArcticRoute.diagnostics.self_check import Item
            sections.append(Section("起终点与规划域自检", "error", [Item("endpoints", "ERROR", str(e))]))
    else:
        # 未知 focus 回退为 all
        return _run_all_checks_full()

    report_md = render_report_markdown(sections)
    out_md = save_to_reports(report_md)
    save_structured_json(sections)
    return sections, out_md


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--ym", default=None)
    parser.add_argument("--skip-planning", action="store_true")
    parser.add_argument("--focus", choices=["all", "prior", "corridor", "eco", "plan", "planning", "endpoints"], default="all")
    args = parser.parse_args()

    # 当前最小版未使用 scenario/ym/skip-planning，这里仅占位以保持兼容
    sections, out_md = run_all_checks(focus=args.focus)
    print(f"Self-check finished. Report saved to {out_md}")
