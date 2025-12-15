"""
路线与陆地掩码一致性测试模块。

测试 demo 路线是否踩陆。
"""

import numpy as np

from arcticroute.core.grid import make_demo_grid
from arcticroute.core.landmask import evaluate_route_against_landmask
from arcticroute.core.cost import build_demo_cost
from arcticroute.core.astar import plan_route_latlon


def test_demo_routes_do_not_cross_land():
    """
    测试 demo 路线不穿陆。

    - 构建 demo 网格与 landmask；
    - 规划三条不同冰带权重的路线（efficient / balanced / safe）；
    - 对每条路线调用 evaluate_route_against_landmask；
    - 断言 on_land_steps == 0（不踩陆）。
    """
    grid, land_mask = make_demo_grid()

    # 三组不同的 ice_penalty，模拟 efficient / balanced / safe
    configs = [("efficient", 1.0), ("balanced", 4.0), ("safe", 8.0)]

    for label, ice_penalty in configs:
        cost_field = build_demo_cost(grid, land_mask, ice_penalty=ice_penalty)
        route = plan_route_latlon(
            cost_field=cost_field,
            start_lat=66.0,
            start_lon=5.0,
            end_lat=78.0,
            end_lon=150.0,
            neighbor8=True,
        )

        stats = evaluate_route_against_landmask(grid, land_mask, route)

        # Demo 世界里的路线应该完全不踩陆
        assert stats.on_land_steps == 0, (
            f"{label} route should not cross land in demo mask, "
            f"but got {stats.on_land_steps} on_land_steps"
        )
        assert stats.total_steps == len(route), (
            f"total_steps should equal route length, "
            f"got {stats.total_steps} vs {len(route)}"
        )


def test_empty_route():
    """
    测试空路线的统计。

    - 传入空列表作为路线；
    - 断言返回的统计信息全为 0 / None。
    """
    grid, land_mask = make_demo_grid()

    stats = evaluate_route_against_landmask(grid, land_mask, [])

    assert stats.total_steps == 0
    assert stats.on_land_steps == 0
    assert stats.on_ocean_steps == 0
    assert stats.first_land_index is None
    assert stats.first_land_latlon is None


def test_route_with_single_point():
    """
    测试单点路线的统计。

    - 传入只有一个点的路线；
    - 断言 total_steps == 1；
    - 根据该点是否在陆地上，on_land_steps 或 on_ocean_steps 应为 1。
    """
    grid, land_mask = make_demo_grid()

    # 选择一个海洋点（左侧）
    ocean_point = (70.0, 50.0)
    stats = evaluate_route_against_landmask(grid, land_mask, [ocean_point])

    assert stats.total_steps == 1
    assert stats.on_land_steps + stats.on_ocean_steps == 1

    # 选择一个陆地点（右侧，因为 demo 网格右侧 10 列是陆地）
    land_point = (70.0, 155.0)
    stats = evaluate_route_against_landmask(grid, land_mask, [land_point])

    assert stats.total_steps == 1
    assert stats.on_land_steps + stats.on_ocean_steps == 1
    # 这个点应该在陆地上
    assert stats.on_land_steps == 1
    assert stats.first_land_index == 0
    assert stats.first_land_latlon == land_point

















