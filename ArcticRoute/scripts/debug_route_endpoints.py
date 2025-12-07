# -*- coding: utf-8 -*-
"""
调试脚本：验证三条方案（balanced/safe/efficient）的路线起止点是否与指定经纬度一致，并输出起止偏差（km）。

用法示例：
  python ArcticRoute/scripts/debug_route_endpoints.py --ym 202412 \
      --start 75 10 --goal 72 40 \
      --diag --use-escort --w-interact 0.0

备注：
- 若未指定 start/goal，默认使用 Planner 中常见的示例经纬度。
- 偏差阈值默认 20 km，超出则打印 WARNING。
"""
from __future__ import annotations
import argparse
from pathlib import Path
import math
import numpy as np

from ArcticRoute.core import planner_service as ps

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    rad = math.pi / 180.0
    dlat = (lat2 - lat1) * rad
    dlon = (lon2 - lon1) * rad
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1 * rad) * math.cos(lat2 * rad) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.asin(min(1.0, math.sqrt(a)))
    return R * c


def route_endpoints_offset_km(env: ps.EnvironmentContext, route: ps.RouteResult, start_lat, start_lon, goal_lat, goal_lon) -> tuple[float, float]:
    if not route or not route.path_lonlat:
        return float("inf"), float("inf")
    lat_s, lon_s = float(route.path_lonlat[0][0]), float(route.path_lonlat[0][1])
    lat_g, lon_g = float(route.path_lonlat[-1][0]), float(route.path_lonlat[-1][1])
    d_start = haversine_km(start_lat, start_lon, lat_s, lon_s)
    d_goal = haversine_km(goal_lat, goal_lon, lat_g, lon_g)
    return float(d_start), float(d_goal)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ym", required=True, help="YYYYMM")
    ap.add_argument("--start", nargs=2, type=float, default=[75.0, 10.0], metavar=("LAT", "LON"))
    ap.add_argument("--goal", nargs=2, type=float, default=[72.0, 40.0], metavar=("LAT", "LON"))
    ap.add_argument("--thr-km", type=float, default=20.0, help="偏差阈值（km），超过则报警")
    ap.add_argument("--use-escort", action="store_true", help="启用护航走廊折减")
    ap.add_argument("--w-interact", type=float, default=0.0, help="交互/拥挤权重")
    ap.add_argument("--diag", action="store_true", help="允许对角步")
    args = ap.parse_args()

    ym = str(args.ym)
    s_lat, s_lon = float(args.start[0]), float(args.start[1])
    g_lat, g_lon = float(args.goal[0]), float(args.goal[1])
    allow_diag = bool(args.diag)

    # 1) 加载一次环境用于坐标→索引转换
    env0 = ps.load_environment(ym, w_ice=0.7, w_accident=0.2, prior_weight=0.2, use_escort=bool(args.use_escort), w_interact=float(args.w_interact))
    s_ij = ps.latlon_to_ij(env0, s_lat, s_lon)
    g_ij = ps.latlon_to_ij(env0, g_lat, g_lon)
    if s_ij is None or g_ij is None:
        print("[ERR] 起止点超出网格范围，无法投射到索引。")
        return

    # 2) 三方案规划（balanced/safe/efficient）
    presets = [
        ("balanced", {"profile_name": "balanced"}),
        ("safe", {"profile_name": "safe"}),
        ("efficient", {"profile_name": "efficient"}),
    ]

    for name, params in presets:
        env, rr = ps.run_planning_pipeline(
            ym=ym,
            start_ij=s_ij,
            goal_ij=g_ij,
            w_ice=0.7,
            w_accident=0.2,
            prior_weight=0.2,
            allow_diagonal=allow_diag,
            heuristic="euclidean",
            eco_enabled=False,
            **params,
        )
        d_s, d_g = route_endpoints_offset_km(env, rr, s_lat, s_lon, g_lat, g_lon)
        flag_s = "WARNING" if d_s > args.thr_km else "OK"
        flag_g = "WARNING" if d_g > args.thr_km else "OK"
        print(f"[{name}] steps={rr.len} reachable={rr.reachable} start_off={d_s:.1f}km({flag_s}) end_off={d_g:.1f}km({flag_g})")


if __name__ == "__main__":
    main()




















