"""
Demo 路由规划脚本。

在 demo 网格上规划一条从起点到终点的路径。
"""

from __future__ import annotations

from arcticroute.core.astar import plan_route_latlon
from arcticroute.core.cost import build_demo_cost
from arcticroute.core.grid import make_demo_grid


def main() -> None:
    """主函数：规划并打印 demo 路由。"""
    grid, land_mask = make_demo_grid()
    cf = build_demo_cost(grid, land_mask)

    start_lat, start_lon = 66.0, 5.0
    end_lat, end_lon = 78.0, 150.0

    path = plan_route_latlon(cf, start_lat, start_lon, end_lat, end_lon)

    if not path:
        print("[DEMO] route is not reachable")
        return

    print(f"[DEMO] route length (points): {len(path)}")
    print(f"[DEMO] start -> {path[0]}")
    print(f"[DEMO] end   -> {path[-1]}")


if __name__ == "__main__":
    main()













