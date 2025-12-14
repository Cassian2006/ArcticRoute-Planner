"""
成本分解与路线剖面分析的测试。

测试成本组件分解、路线成本分解等功能。
"""

from __future__ import annotations

import numpy as np

from arcticroute.core.analysis import compute_route_cost_breakdown
from arcticroute.core.cost import build_demo_cost
from arcticroute.core.grid import make_demo_grid


def test_breakdown_components_sum_to_total():
    """测试成本分解组件之和等于总成本。"""
    grid, land_mask = make_demo_grid()
    cost_field = build_demo_cost(grid, land_mask)

    # 构造一条简单的对角线路线
    route = [(65.0, 0.0), (66.0, 2.0), (67.0, 4.0)]

    breakdown = compute_route_cost_breakdown(grid, cost_field, route)
    total = breakdown.total_cost
    comp_sum = sum(breakdown.component_totals.values())

    assert total >= 0
    assert comp_sum >= 0
    # 允许小的浮点误差
    assert abs(total - comp_sum) < 1e-5


def test_empty_route_breakdown_zero():
    """测试空路线的成本分解为零。"""
    grid, land_mask = make_demo_grid()
    cost_field = build_demo_cost(grid, land_mask)
    breakdown = compute_route_cost_breakdown(grid, cost_field, [])
    
    assert breakdown.total_cost == 0
    assert breakdown.component_totals == {} or all(v == 0 for v in breakdown.component_totals.values())
    assert breakdown.s_km == []
    assert breakdown.component_along_path == {}


def test_breakdown_has_expected_components():
    """测试成本分解包含预期的组件。"""
    grid, land_mask = make_demo_grid()
    cost_field = build_demo_cost(grid, land_mask)
    route = [(70.0, 10.0), (71.0, 11.0)]
    
    breakdown = compute_route_cost_breakdown(grid, cost_field, route)

    # demo 模式下应该至少有 base_distance 和 ice_risk 两个组件
    assert "base_distance" in breakdown.component_totals
    assert "ice_risk" in breakdown.component_totals


def test_breakdown_fractions_sum_to_one():
    """测试成本分解的占比之和为 1（或接近 1）。"""
    grid, land_mask = make_demo_grid()
    cost_field = build_demo_cost(grid, land_mask)
    route = [(65.0, 5.0), (70.0, 50.0), (75.0, 100.0)]
    
    breakdown = compute_route_cost_breakdown(grid, cost_field, route)
    
    if breakdown.total_cost > 0:
        fraction_sum = sum(breakdown.component_fractions.values())
        # 允许小的浮点误差
        assert abs(fraction_sum - 1.0) < 1e-5


def test_breakdown_s_km_monotonic():
    """测试沿程距离 s_km 单调递增。"""
    grid, land_mask = make_demo_grid()
    cost_field = build_demo_cost(grid, land_mask)
    route = [(65.0, 0.0), (66.0, 10.0), (67.0, 20.0), (68.0, 30.0)]
    
    breakdown = compute_route_cost_breakdown(grid, cost_field, route)
    
    # s_km 应该单调递增
    for i in range(1, len(breakdown.s_km)):
        assert breakdown.s_km[i] >= breakdown.s_km[i - 1]
    
    # 第一个点应该是 0
    assert breakdown.s_km[0] == 0.0


def test_breakdown_component_along_path_length():
    """测试沿程组件数据长度与路径长度一致。"""
    grid, land_mask = make_demo_grid()
    cost_field = build_demo_cost(grid, land_mask)
    route = [(65.0, 0.0), (66.0, 10.0), (67.0, 20.0)]
    
    breakdown = compute_route_cost_breakdown(grid, cost_field, route)
    
    # s_km 长度应该等于路径长度
    assert len(breakdown.s_km) == len(route)
    
    # 每个组件的沿程数据长度也应该等于路径长度
    for comp_name, comp_values in breakdown.component_along_path.items():
        assert len(comp_values) == len(route), f"Component {comp_name} length mismatch"


def test_cost_field_components_shape():
    """测试 CostField 的组件形状与成本场一致。"""
    grid, land_mask = make_demo_grid()
    cost_field = build_demo_cost(grid, land_mask)
    
    # 检查 components 中每个数组的形状
    for comp_name, comp_array in cost_field.components.items():
        assert comp_array.shape == cost_field.cost.shape, \
            f"Component {comp_name} shape {comp_array.shape} != cost shape {cost_field.cost.shape}"


def test_cost_field_components_sum():
    """测试 CostField 的组件之和等于总成本（在海洋区域）。"""
    grid, land_mask = make_demo_grid()
    cost_field = build_demo_cost(grid, land_mask)
    
    # 在海洋区域，组件之和应该等于总成本
    ocean_mask = ~cost_field.land_mask
    
    # 计算组件之和
    comp_sum = np.zeros_like(cost_field.cost)
    for comp_array in cost_field.components.values():
        comp_sum += comp_array
    
    # 在海洋区域检查
    ocean_cost = cost_field.cost[ocean_mask]
    ocean_comp_sum = comp_sum[ocean_mask]
    
    # 允许小的浮点误差
    assert np.allclose(ocean_cost, ocean_comp_sum, atol=1e-5)


def test_breakdown_with_different_ice_penalties():
    """测试不同冰带权重下的成本分解。"""
    grid, land_mask = make_demo_grid()
    
    # 低冰带权重
    cost_field_low = build_demo_cost(grid, land_mask, ice_penalty=1.0)
    route = [(75.0, 50.0), (76.0, 60.0), (77.0, 70.0)]
    breakdown_low = compute_route_cost_breakdown(grid, cost_field_low, route)
    
    # 高冰带权重
    cost_field_high = build_demo_cost(grid, land_mask, ice_penalty=10.0)
    breakdown_high = compute_route_cost_breakdown(grid, cost_field_high, route)
    
    # 高权重的 ice_risk 应该更大
    assert breakdown_high.component_totals["ice_risk"] > breakdown_low.component_totals["ice_risk"]
    
    # base_distance 应该相同
    assert abs(breakdown_high.component_totals["base_distance"] - 
               breakdown_low.component_totals["base_distance"]) < 1e-5

