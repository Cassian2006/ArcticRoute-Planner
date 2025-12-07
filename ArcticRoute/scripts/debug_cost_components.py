# -*- coding: utf-8 -*-
from __future__ import annotations
"""
打印成本构成各组件的总贡献与占比，便于人工确认各维度是否生效。

用法：
  python ArcticRoute/scripts/debug_cost_components.py [--ym 202412]

输出示例：
  component   value    fraction
  ice         12345    0.65
  wave        4567     0.24
  prior       2100     0.11
  accident    0        0.00
  interact    0        0.00
"""
import argparse
from pathlib import Path
import yaml
import numpy as np

from ArcticRoute.core import planner_service as ps


def _load_scenario():
    root = Path(__file__).resolve().parents[2]
    scn_file = root / "configs" / "scenarios.yaml"
    ym = "202412"
    start_latlon = (69.0, 33.0)
    goal_latlon = (70.5, 170.0)
    if scn_file.exists():
        try:
            obj = yaml.safe_load(scn_file.read_text(encoding="utf-8")) or {}
            scens = (obj or {}).get("scenarios") or []
            if scens:
                s0 = scens[0]
                ym = str(s0.get("ym", ym))
                start_latlon = (float((s0.get("start") or [start_latlon[0], start_latlon[1]])[0]), float((s0.get("start") or [start_latlon[0], start_latlon[1]])[1]))
                goal_latlon = (float((s0.get("goal") or [goal_latlon[0], goal_latlon[1]])[0]), float((s0.get("goal") or [goal_latlon[0], goal_latlon[1]])[1]))
        except Exception:
            pass
    return ym, start_latlon, goal_latlon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ym", default=None)
    args = ap.parse_args()

    ym0, s_ll, g_ll = _load_scenario()
    ym = str(args.ym or ym0)

    # 先加载 env 获取起止索引
    env = ps.load_environment(ym=ym, w_ice=1.0, w_accident=1.0, prior_weight=0.3, profile_name="balanced")
    si = ps.latlon_to_ij(env, s_ll[0], s_ll[1])
    gi = ps.latlon_to_ij(env, g_ll[0], g_ll[1])
    if not si or not gi:
        print("[ERROR] 无法映射起止点到网格")
        return 1

    env2, route = ps.run_planning_pipeline(
        ym=ym, start_ij=si, goal_ij=gi,
        w_ice=1.0, w_accident=1.0, prior_weight=0.3,
        allow_diagonal=True, heuristic="euclidean",
        eco_enabled=True, profile_name="balanced",
    )
    comp = ps.analyze_route_cost(env2, route) or {}
    rc = comp.get("risk_components") or {}
    rn = comp.get("risk_components_normalized") or {}

    # 打印表格
    names = ["ice", "wave", "accident", "congestion", "prior"]
    print("component,value,fraction")
    s = sum(float(rc.get(k, 0.0) or 0.0) for k in names)
    for k in names:
        v = float(rc.get(k, 0.0) or 0.0)
        frac = float(rn.get(k, (v / s if s > 0 else 0.0)))
        print(f"{k},{v:.2f},{frac:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

