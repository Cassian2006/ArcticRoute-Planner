#!/usr/bin/env python3
"""
Phase 5 验证脚本：成本分解 & 路线剖面。

验证以下功能：
1. CostField 支持可解释的成本组件
2. compute_route_cost_breakdown 能正确分解成本
3. UI 能正确调用分解工具
"""

from arcticroute.core.grid import make_demo_grid
from arcticroute.core.cost import build_demo_cost
from arcticroute.core.astar import plan_route_latlon
from arcticroute.core.analysis import compute_route_cost_breakdown
from arcticroute.ui.planner_minimal import plan_three_routes


def test_cost_field_components():
    """验证 CostField 的组件分解。"""
    print("\n=== 测试 1: CostField 组件分解 ===")
    grid, land_mask = make_demo_grid()
    cost_field = build_demo_cost(grid, land_mask, ice_penalty=4.0)
    
    # 检查组件是否存在
    assert "base_distance" in cost_field.components, "缺少 base_distance 组件"
    assert "ice_risk" in cost_field.components, "缺少 ice_risk 组件"
    
    # 检查组件形状
    assert cost_field.components["base_distance"].shape == cost_field.cost.shape
    assert cost_field.components["ice_risk"].shape == cost_field.cost.shape
    
    print("✓ CostField 包含正确的组件")
    print(f"  - base_distance 形状: {cost_field.components['base_distance'].shape}")
    print(f"  - ice_risk 形状: {cost_field.components['ice_risk'].shape}")


def test_route_cost_breakdown():
    """验证路线成本分解。"""
    print("\n=== 测试 2: 路线成本分解 ===")
    grid, land_mask = make_demo_grid()
    cost_field = build_demo_cost(grid, land_mask, ice_penalty=4.0)
    
    # 规划一条路线
    route = plan_route_latlon(
        cost_field,
        start_lat=66.0,
        start_lon=5.0,
        end_lat=78.0,
        end_lon=150.0,
        neighbor8=True,
    )
    
    assert route, "路线规划失败"
    print(f"✓ 规划了一条路线，包含 {len(route)} 个点")
    
    # 计算成本分解
    breakdown = compute_route_cost_breakdown(grid, cost_field, route)
    
    assert breakdown.total_cost > 0, "总成本应该大于 0"
    assert "base_distance" in breakdown.component_totals
    assert "ice_risk" in breakdown.component_totals
    
    print(f"✓ 成本分解成功")
    print(f"  - 总成本: {breakdown.total_cost:.2f}")
    print(f"  - base_distance: {breakdown.component_totals['base_distance']:.2f} ({breakdown.component_fractions['base_distance']:.1%})")
    print(f"  - ice_risk: {breakdown.component_totals['ice_risk']:.2f} ({breakdown.component_fractions['ice_risk']:.1%})")
    
    # 检查占比之和
    fraction_sum = sum(breakdown.component_fractions.values())
    assert abs(fraction_sum - 1.0) < 1e-5, f"占比之和应该为 1，实际为 {fraction_sum}"
    print(f"✓ 占比之和: {fraction_sum:.1%}")
    
    # 检查剖面数据
    assert len(breakdown.s_km) == len(route), "s_km 长度应该等于路径长度"
    assert breakdown.s_km[0] == 0.0, "起点距离应该为 0"
    assert all(breakdown.s_km[i] <= breakdown.s_km[i+1] for i in range(len(breakdown.s_km)-1)), "s_km 应该单调递增"
    
    print(f"✓ 剖面数据正确")
    print(f"  - 路径总长: {breakdown.s_km[-1]:.2f} km")
    print(f"  - 沿程数据点数: {len(breakdown.s_km)}")


def test_plan_three_routes_with_cost_fields():
    """验证 plan_three_routes 返回 cost_fields。"""
    print("\n=== 测试 3: 三方案规划与成本场返回 ===")
    grid, land_mask = make_demo_grid()
    
    routes_info, cost_fields = plan_three_routes(
        grid,
        land_mask,
        start_lat=66.0,
        start_lon=5.0,
        end_lat=78.0,
        end_lon=150.0,
        allow_diag=True,
        vessel=None,
    )
    
    assert len(routes_info) == 3, "应该规划 3 条路线"
    assert len(cost_fields) == 3, "应该返回 3 个成本场"
    
    print(f"✓ 规划了 3 条路线")
    
    # 检查每条路线
    for route_info in routes_info:
        print(f"  - {route_info.label}: {'可达' if route_info.reachable else '不可达'}")
        if route_info.reachable:
            assert route_info.label in cost_fields, f"缺少 {route_info.label} 的成本场"
            cost_field = cost_fields[route_info.label]
            assert cost_field.components, f"{route_info.label} 的成本场缺少组件"
            
            # 计算该路线的成本分解
            breakdown = compute_route_cost_breakdown(grid, cost_field, route_info.coords)
            print(f"    - 成本分解: base_distance={breakdown.component_totals['base_distance']:.2f}, ice_risk={breakdown.component_totals['ice_risk']:.2f}")


def test_empty_route():
    """验证空路线的处理。"""
    print("\n=== 测试 4: 空路线处理 ===")
    grid, land_mask = make_demo_grid()
    cost_field = build_demo_cost(grid, land_mask)
    
    breakdown = compute_route_cost_breakdown(grid, cost_field, [])
    
    assert breakdown.total_cost == 0, "空路线的总成本应该为 0"
    assert breakdown.s_km == [], "空路线的 s_km 应该为空"
    
    print("✓ 空路线处理正确")


def main():
    """运行所有验证测试。"""
    print("=" * 60)
    print("Phase 5 验证：成本分解 & 路线剖面")
    print("=" * 60)
    
    try:
        test_cost_field_components()
        test_route_cost_breakdown()
        test_plan_three_routes_with_cost_fields()
        test_empty_route()
        
        print("\n" + "=" * 60)
        print("✓ 所有验证测试通过！")
        print("=" * 60)
        print("\n下一步：运行 Streamlit UI")
        print("  streamlit run run_ui.py")
        
    except AssertionError as e:
        print(f"\n✗ 验证失败: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ 出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

















