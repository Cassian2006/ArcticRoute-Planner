# -*- coding: utf-8 -*-
"""
快速排查：风险贡献分解是否为空
运行：
  python ArcticRoute/scripts/debug_risk_breakdown.py
"""
from __future__ import annotations

import sys
from ArcticRoute.core import planner_service as ps


def main():
    ym = "202412"
    profile = "balanced"
    # 选取一组经纬度（与测试用例一致）
    start = (69.0, 33.0)
    end = (70.5, 150.0)

    # 加载环境
    env = ps.load_environment(ym=ym, profile_name=profile)

    # 规划一条严格路线
    rr = ps.compute_route_strict_from_latlon(
        env_ctx=env,
        start_lat=float(start[0]),
        start_lon=float(start[1]),
        end_lat=float(end[0]),
        end_lon=float(end[1]),
        allow_diagonal=True,
        heuristic="euclidean",
    )

    # 若严格模式不可达，回退到栅格最近邻 + compute_route
    if not rr.reachable:
        try:
            si, sj = ps.find_nearest_grid_index(float(start[0]), float(start[1]), env)
            gi, gj = ps.find_nearest_grid_index(float(end[0]), float(end[1]), env)
            rr2 = ps.compute_route(
                env,
                start_ij=(int(si), int(sj)),
                goal_ij=(int(gi), int(gj)),
                allow_diagonal=True,
                heuristic="euclidean",
            )
            if rr2 and rr2.reachable:
                rr = rr2
        except Exception:
            pass

    if not rr.reachable:
        print("[DEBUG_RISK] route not reachable; debug:", getattr(rr, "debug", {}))
        sys.exit(1)

    # 成本分解
    res = ps.analyze_route_cost(env, rr)
    print("[DEBUG_RISK] result:", res)
    rc = (res or {}).get("risk_components") or {}
    print("[DEBUG_RISK] risk_components:", rc)
    ok = any((v is not None and float(v) != 0.0) for v in rc.values())
    if not ok:
        sys.exit(2)
    print("[DEBUG_RISK] OK: non-zero components present")


if __name__ == "__main__":
    main()
