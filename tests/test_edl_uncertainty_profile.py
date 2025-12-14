"""
EDL 不确定性剖面的测试。

测试 EDL 不确定性在 CostField 和 RouteCostProfile 中的流转。
"""

from __future__ import annotations

import numpy as np

from arcticroute.core.analysis import compute_route_profile
from arcticroute.core.cost import CostField, build_demo_cost
from arcticroute.core.grid import Grid2D, make_demo_grid


def test_cost_field_edl_uncertainty_optional():
    """测试 CostField 的 edl_uncertainty 字段是可选的。"""
    grid, land_mask = make_demo_grid()
    cost_field = build_demo_cost(grid, land_mask)
    
    # 默认情况下 edl_uncertainty 应该是 None
    assert cost_field.edl_uncertainty is None


def test_cost_field_edl_uncertainty_shape():
    """测试 CostField 的 edl_uncertainty 形状与成本场一致。"""
    grid, land_mask = make_demo_grid()
    ny, nx = grid.shape()
    
    # 手工构造一个有 edl_uncertainty 的 CostField
    cost = np.ones((ny, nx), dtype=float)
    cost[land_mask] = np.inf
    
    edl_uncertainty = np.random.rand(ny, nx).astype(float)
    
    cost_field = CostField(
        grid=grid,
        cost=cost,
        land_mask=land_mask,
        components={"base": np.ones((ny, nx), dtype=float)},
        edl_uncertainty=edl_uncertainty,
    )
    
    assert cost_field.edl_uncertainty.shape == cost_field.cost.shape
    assert cost_field.edl_uncertainty.shape == (ny, nx)


def test_route_profile_edl_uncertainty_none():
    """测试空路线的 edl_uncertainty 为 None。"""
    grid, land_mask = make_demo_grid()
    cost_field = build_demo_cost(grid, land_mask)
    
    profile = compute_route_profile([], cost_field)
    
    assert profile.edl_uncertainty is None
    assert len(profile.distance_km) == 0


def test_route_profile_edl_uncertainty_sampling():
    """测试沿路线采样 edl_uncertainty。"""
    grid, land_mask = make_demo_grid()
    ny, nx = grid.shape()
    
    # 构造一个递增的 edl_uncertainty 模式
    edl_uncertainty = np.zeros((ny, nx), dtype=float)
    for i in range(ny):
        edl_uncertainty[i, :] = i / ny  # 从 0 到 1 递增
    
    cost = np.ones((ny, nx), dtype=float)
    cost[land_mask] = np.inf
    
    cost_field = CostField(
        grid=grid,
        cost=cost,
        land_mask=land_mask,
        components={"base": np.ones((ny, nx), dtype=float)},
        edl_uncertainty=edl_uncertainty,
    )
    
    # 构造一条从南到北的路线
    route = [(65.0, 10.0), (70.0, 10.0), (75.0, 10.0), (80.0, 10.0)]
    
    profile = compute_route_profile(route, cost_field)
    
    # edl_uncertainty 应该不为 None
    assert profile.edl_uncertainty is not None
    
    # 长度应该与路线长度一致
    assert len(profile.edl_uncertainty) == len(route)
    
    # 由于纬度递增，不确定性应该大致递增
    # （允许一些偏差因为映射到网格点）
    valid_vals = profile.edl_uncertainty[np.isfinite(profile.edl_uncertainty)]
    if len(valid_vals) > 1:
        # 检查大致趋势：最后的值应该大于最前的值
        assert valid_vals[-1] >= valid_vals[0] - 0.2  # 允许一些误差


def test_route_profile_edl_uncertainty_clipped():
    """测试 edl_uncertainty 值在 [0, 1] 范围内。"""
    grid, land_mask = make_demo_grid()
    ny, nx = grid.shape()
    
    # 构造一个有极端值的 edl_uncertainty
    edl_uncertainty = np.random.rand(ny, nx) * 2 - 0.5  # 范围 [-0.5, 1.5]
    
    cost = np.ones((ny, nx), dtype=float)
    cost[land_mask] = np.inf
    
    cost_field = CostField(
        grid=grid,
        cost=cost,
        land_mask=land_mask,
        components={"base": np.ones((ny, nx), dtype=float)},
        edl_uncertainty=edl_uncertainty,
    )
    
    route = [(70.0, 50.0), (71.0, 51.0), (72.0, 52.0)]
    profile = compute_route_profile(route, cost_field)
    
    # 所有有限的值应该在 [0, 1] 范围内
    if profile.edl_uncertainty is not None:
        valid_vals = profile.edl_uncertainty[np.isfinite(profile.edl_uncertainty)]
        assert np.all(valid_vals >= 0.0)
        assert np.all(valid_vals <= 1.0)


def test_route_profile_distance_km_monotonic():
    """测试 distance_km 单调递增。"""
    grid, land_mask = make_demo_grid()
    ny, nx = grid.shape()
    
    edl_uncertainty = np.ones((ny, nx), dtype=float) * 0.5
    cost = np.ones((ny, nx), dtype=float)
    cost[land_mask] = np.inf
    
    cost_field = CostField(
        grid=grid,
        cost=cost,
        land_mask=land_mask,
        components={"base": np.ones((ny, nx), dtype=float)},
        edl_uncertainty=edl_uncertainty,
    )
    
    route = [(65.0, 0.0), (66.0, 10.0), (67.0, 20.0), (68.0, 30.0)]
    profile = compute_route_profile(route, cost_field)
    
    # distance_km 应该单调递增
    for i in range(1, len(profile.distance_km)):
        assert profile.distance_km[i] >= profile.distance_km[i - 1]
    
    # 第一个点应该是 0
    assert profile.distance_km[0] == 0.0


def test_route_profile_components_shape():
    """测试 RouteCostProfile 的组件形状与路线长度一致。"""
    grid, land_mask = make_demo_grid()
    ny, nx = grid.shape()
    
    edl_uncertainty = np.ones((ny, nx), dtype=float) * 0.5
    cost = np.ones((ny, nx), dtype=float)
    cost[land_mask] = np.inf
    
    components = {
        "base": np.ones((ny, nx), dtype=float),
        "ice": np.ones((ny, nx), dtype=float) * 0.5,
    }
    
    cost_field = CostField(
        grid=grid,
        cost=cost,
        land_mask=land_mask,
        components=components,
        edl_uncertainty=edl_uncertainty,
    )
    
    route = [(70.0, 50.0), (71.0, 51.0), (72.0, 52.0)]
    profile = compute_route_profile(route, cost_field)
    
    # 检查形状一致性
    assert len(profile.distance_km) == len(route)
    assert len(profile.total_cost) == len(route)
    
    for comp_name, comp_values in profile.components.items():
        assert len(comp_values) == len(route), \
            f"Component {comp_name} length {len(comp_values)} != route length {len(route)}"
    
    if profile.edl_uncertainty is not None:
        assert len(profile.edl_uncertainty) == len(route)


def test_route_profile_without_edl_uncertainty():
    """测试没有 edl_uncertainty 的 CostField 也能正常工作。"""
    grid, land_mask = make_demo_grid()
    cost_field = build_demo_cost(grid, land_mask)
    
    # build_demo_cost 不包含 edl_uncertainty
    assert cost_field.edl_uncertainty is None
    
    route = [(70.0, 50.0), (71.0, 51.0), (72.0, 52.0)]
    profile = compute_route_profile(route, cost_field)
    
    # 应该能正常计算，edl_uncertainty 为 None
    assert profile.edl_uncertainty is None
    assert len(profile.distance_km) == len(route)
    assert len(profile.total_cost) == len(route)


def test_route_profile_edl_uncertainty_constant():
    """测试常数 edl_uncertainty 的采样。"""
    grid, land_mask = make_demo_grid()
    ny, nx = grid.shape()
    
    # 常数不确定性
    constant_unc = 0.7
    edl_uncertainty = np.ones((ny, nx), dtype=float) * constant_unc
    
    cost = np.ones((ny, nx), dtype=float)
    cost[land_mask] = np.inf
    
    cost_field = CostField(
        grid=grid,
        cost=cost,
        land_mask=land_mask,
        components={"base": np.ones((ny, nx), dtype=float)},
        edl_uncertainty=edl_uncertainty,
    )
    
    route = [(70.0, 50.0), (71.0, 51.0), (72.0, 52.0)]
    profile = compute_route_profile(route, cost_field)
    
    # 所有有限的值应该接近常数
    if profile.edl_uncertainty is not None:
        valid_vals = profile.edl_uncertainty[np.isfinite(profile.edl_uncertainty)]
        assert np.allclose(valid_vals, constant_unc, atol=0.01)











