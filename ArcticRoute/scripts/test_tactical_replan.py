"""
自测：战术重规划 compute_tactical_replan

用法：
  python ArcticRoute/scripts/test_tactical_replan.py

流程：
- 加载环境与基线航线
- 取中点 current_idx
- 在当前点附近构造一个矩形 Polygon 作为 hazard_shapes
- 调用 compute_tactical_replan
- 输出 baseline / tactical / delta 摘要
"""
from __future__ import annotations

import sys
from pathlib import Path

# -- 将项目根加入 sys.path --
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import List, Dict
import json

from ArcticRoute.core import planner_service


def _make_rect_polygon(lat_c: float, lon_c: float, dlat: float = 1.0, dlon: float = 2.0) -> Dict:
    """在 (lat_c, lon_c) 周围构造一个简单矩形，多边形以 GeoJSON 格式返回（WGS84）。"""
    lon1 = ((lon_c - dlon + 180.0) % 360.0) - 180.0
    lon2 = ((lon_c + dlon + 180.0) % 360.0) - 180.0
    lat1 = max(-90.0, lat_c - dlat)
    lat2 = min(90.0, lat_c + dlat)
    coords = [
        [lon1, lat1],
        [lon2, lat1],
        [lon2, lat2],
        [lon1, lat2],
        [lon1, lat1],
    ]
    return {"type": "Polygon", "coordinates": [coords]}


def main() -> None:
    ym = "202412"
    start_ij = (60, 150)
    goal_ij = (60, 1000)

    print("[1] 加载环境…")
    env = planner_service.load_environment(ym=ym, w_ice=0.7, w_accident=0.2, prior_weight=0.2)
    if env.cost_da is None:
        print("  -> 失败：缺少 cost_da")
        return

    print("[2] 计算基线航线…")
    base = planner_service.compute_route(env, start_ij, goal_ij, allow_diagonal=True, heuristic="manhattan")
    base_sum = planner_service.summarize_route(base)
    print("  -> OK. ", base_sum)

    steps = len(base.path_ij)
    if steps < 3:
        print("  -> 基线航线过短，无法继续测试。")
        return

    current_idx = steps // 2

    # 使用基线中点的经纬度，构造一个矩形 hazard
    try:
        mid_ll = base.path_lonlat[current_idx]
        lat_c, lon_c = float(mid_ll[0]), float(mid_ll[1])
    except Exception:
        # 回退：将索引转换为经纬度
        ll = planner_service.path_ij_to_lonlat(env, [base.path_ij[current_idx]])[0]
        lat_c, lon_c = float(ll[0]), float(ll[1])

    hazard = _make_rect_polygon(lat_c, lon_c, dlat=0.8, dlon=1.5)
    hazard_shapes: List[Dict] = [hazard]

    print("[3] 执行战术重规划…")
    tactical, info = planner_service.compute_tactical_replan(
        env_ctx=env,
        base_route=base,
        current_idx=current_idx,
        hazard_shapes=hazard_shapes,
        soft_penalty=50.0,
        allow_diagonal=True,
        heuristic="euclidean",
    )

    print("  -> 战术结果摘要：", planner_service.summarize_route(tactical))
    print("[4] 对比信息：")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()























