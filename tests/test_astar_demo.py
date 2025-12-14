"""
A* 路径规划的 demo 测试。

测试 A* 算法在 demo 网格上的基本功能。
"""

from __future__ import annotations

import numpy as np

from arcticroute.core.astar import plan_route_latlon
from arcticroute.core.cost import build_demo_cost
from arcticroute.core.grid import Grid2D, make_demo_grid


def _path_to_ij(
    grid: Grid2D, path: list[tuple[float, float]]
) -> list[tuple[int, int]]:
    """
    将 (lat, lon) 路径转换为 (i, j) 索引路径。

    对每个 (lat, lon)，找最近的网格索引。
    """
    lat2d = grid.lat2d
    lon2d = grid.lon2d

    ij_path = []
    for lat, lon in path:
        dist = np.sqrt((lat2d - lat) ** 2 + (lon2d - lon) ** 2)
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        ij_path.append((i, j))

    return ij_path


def test_astar_demo_route_exists():
    """测试 demo 路径是否可达。"""
    grid, land_mask = make_demo_grid()
    cf = build_demo_cost(grid, land_mask)

    path = plan_route_latlon(
        cf, start_lat=66.0, start_lon=5.0, end_lat=78.0, end_lon=150.0
    )

    assert path, "demo route should be reachable"
    assert len(path) > 0, "path should not be empty"


def test_astar_demo_route_not_cross_land():
    """测试路径不穿过陆地。"""
    grid, land_mask = make_demo_grid()
    cf = build_demo_cost(grid, land_mask)

    path = plan_route_latlon(
        cf, start_lat=66.0, start_lon=5.0, end_lat=78.0, end_lon=150.0
    )

    assert path, "demo route should be reachable"

    # 将路径转换为 (i, j) 索引
    ij_path = _path_to_ij(grid, path)

    # 检查路径上的每一个格点都不在陆地上
    for i, j in ij_path:
        assert not land_mask[i, j], f"path crosses land at ({i}, {j})"


def test_astar_start_end_near_input():
    """测试路径首尾点与输入起终点的距离在合理范围内。"""
    grid, land_mask = make_demo_grid()
    cf = build_demo_cost(grid, land_mask)

    start_lat, start_lon = 66.0, 5.0
    end_lat, end_lon = 78.0, 150.0

    path = plan_route_latlon(cf, start_lat, start_lon, end_lat, end_lon)

    assert path, "demo route should be reachable"

    # 检查起点
    path_start_lat, path_start_lon = path[0]
    start_dist = np.sqrt((path_start_lat - start_lat) ** 2 + (path_start_lon - start_lon) ** 2)
    assert start_dist <= 3.0, f"start point distance {start_dist} > 3.0 degrees"

    # 检查终点
    path_end_lat, path_end_lon = path[-1]
    end_dist = np.sqrt((path_end_lat - end_lat) ** 2 + (path_end_lon - end_lon) ** 2)
    assert end_dist <= 15.0, f"end point distance {end_dist} > 15.0 degrees"


def test_neighbor8_vs_neighbor4_path_length():
    """测试 neighbor8=False 时路径步数应该 >= neighbor8=True 的步数。"""
    grid, land_mask = make_demo_grid()
    cf = build_demo_cost(grid, land_mask)

    start_lat, start_lon = 66.0, 5.0
    end_lat, end_lon = 78.0, 150.0

    # 8 邻接路径
    path_8 = plan_route_latlon(
        cf, start_lat, start_lon, end_lat, end_lon, neighbor8=True
    )
    assert path_8, "8-neighbor path should be reachable"

    # 4 邻接路径
    path_4 = plan_route_latlon(
        cf, start_lat, start_lon, end_lat, end_lon, neighbor8=False
    )
    assert path_4, "4-neighbor path should be reachable"

    # 4 邻接路径应该不短于 8 邻接路径（因为 8 邻接更灵活）
    assert len(path_4) >= len(path_8), (
        f"4-neighbor path length {len(path_4)} should be >= "
        f"8-neighbor path length {len(path_8)}"
    )

