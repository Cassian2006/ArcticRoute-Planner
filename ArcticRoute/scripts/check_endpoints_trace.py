# -*- coding: utf-8 -*-
from __future__ import annotations
"""
ArcticRoute/scripts/check_endpoints_trace.py
一键调试起终点链路：加载环境 → 打印规划域 → 对 start/end 执行权威吸附并检查 land/cost → 可选规划一次。

用法示例：
- 作为模块运行：
  python -m ArcticRoute.scripts.check_endpoints_trace \
    --ym 202412 \
    --start-lat 69.0 --start-lon 33.0 \
    --end-lat 71.0 --end-lon 140.0 \
    --profile balanced \
    --plan

- 直接脚本：
  python ArcticRoute/scripts/check_endpoints_trace.py --ym 202412 --plan
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# 确保作为脚本直接运行时也能导入 ArcticRoute 包（将项目根加入 sys.path）
try:
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
except Exception:
    pass

from ArcticRoute.core.planner_service import (
    load_environment,
    compute_planning_domain,
    snap_point_to_domain_and_ocean,
    latlon_to_ij,
    compute_route,
    summarize_route,
)


def _fmt_domain(dom) -> str:
    try:
        return f"lat[{dom.lat_min:.3f},{dom.lat_max:.3f}], lon[{dom.lon_min:.3f},{dom.lon_max:.3f}]"
    except Exception:
        return str(dom)


def _inspect_cell(env_ctx, ij: Tuple[int, int]) -> dict:
    info = {"ij": (int(ij[0]), int(ij[1]))}
    try:
        lm = getattr(env_ctx, "land_mask", None)
        if lm is not None:
            info["land_mask"] = bool(np.asarray(lm)[ij[0], ij[1]])
        else:
            info["land_mask"] = None
    except Exception:
        info["land_mask"] = None
    try:
        if getattr(env_ctx, "cost_da", None) is not None:
            v = float(np.asarray(env_ctx.cost_da.values, dtype=float)[ij[0], ij[1]])
            info["cost_val"] = v
            info["cost_class"] = ("inf_or_nan" if (not np.isfinite(v)) else (">=1e6" if v >= 1e6 else "finite"))
        else:
            info["cost_val"] = None
            info["cost_class"] = "no_cost"
    except Exception:
        info["cost_val"] = None
        info["cost_class"] = "err"
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="检查起终点吸附与可通行性")
    ap.add_argument("--ym", default="202412", help="年月，形如 202412")
    ap.add_argument("--start-lat", type=float, default=69.0)
    ap.add_argument("--start-lon", type=float, default=33.0)
    ap.add_argument("--end-lat", type=float, default=71.0)
    ap.add_argument("--end-lon", type=float, default=140.0)
    ap.add_argument("--profile", default="balanced", choices=["balanced", "safe", "efficient", "baseline"], help="策略配置")
    ap.add_argument("--plan", action="store_true", help="是否尝试规划一次并打印摘要")
    args = ap.parse_args(argv)

    print("[CHECK] loading environment…", flush=True)
    env = load_environment(
        ym=str(args.ym),
        w_ice=1.0,
        w_accident=0.5,
        prior_weight=0.2,
        profile_name=str(args.profile),
    )

    # 规划域
    try:
        dom = getattr(env, "domain", None) or compute_planning_domain(env)
    except Exception:
        dom = getattr(env, "domain", None)
    print(f"[PLANNING_DOMAIN] {_fmt_domain(dom)}")

    # Start
    print("\n[START]")
    s_lat_in, s_lon_in = float(args.start_lat), float(args.start_lon)
    s_lat_s, s_lon_s, s_info = snap_point_to_domain_and_ocean(env_ctx=env, lat=s_lat_in, lon=s_lon_in)
    s_ij = (int(s_info.get("grid_i")), int(s_info.get("grid_j")))
    s_cell = _inspect_cell(env, s_ij)
    print(f"input=({s_lat_in:.4f},{s_lon_in:.4f}) -> snapped=({s_lat_s:.4f},{s_lon_s:.4f}), ij={s_ij}")
    print(f"land_mask={s_cell.get('land_mask')} cost={s_cell.get('cost_val')} class={s_cell.get('cost_class')}")

    # End
    print("\n[END]")
    e_lat_in, e_lon_in = float(args.end_lat), float(args.end_lon)
    e_lat_s, e_lon_s, e_info = snap_point_to_domain_and_ocean(env_ctx=env, lat=e_lat_in, lon=e_lon_in)
    e_ij = (int(e_info.get("grid_i")), int(e_info.get("grid_j")))
    e_cell = _inspect_cell(env, e_ij)
    print(f"input=({e_lat_in:.4f},{e_lon_in:.4f}) -> snapped=({e_lat_s:.4f},{e_lon_s:.4f}), ij={e_ij}")
    print(f"land_mask={e_cell.get('land_mask')} cost={e_cell.get('cost_val')} class={e_cell.get('cost_class')}")

    # 可选：规划一次
    if bool(args.plan):
        print("\n[PLAN] computing route…")
        try:
            rr = compute_route(env, start_ij=s_ij, goal_ij=e_ij, allow_diagonal=True, heuristic="euclidean")
            s = summarize_route(rr)
            reachable = bool(getattr(rr, "reachable", False))
            print(f"reachable={reachable} steps={int(s.get('steps', 0))} distance_km={float(s.get('distance_km', 0.0)):.1f} cost_sum={float(s.get('risk_score', s.get('cost_sum', 0.0))):.2f}")
            if not reachable:
                print("note: 不可达；请检查起终点是否仍落在不可通行区域（land/inf/1e6）。")
        except Exception as e:
            print(f"[PLAN] failed: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

