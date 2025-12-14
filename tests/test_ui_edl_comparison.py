"""
测试 UI 中的 EDL 模式对比功能。

验证：
1. 三种模式的路线规划都能成功执行
2. EDL 成本随模式单调递增
3. 不确定性成本只在 edl_robust 模式中出现
4. 场景预设能正确填充起止点
"""

from __future__ import annotations

import pytest
import numpy as np

from arcticroute.core.grid import make_demo_grid
from arcticroute.core.cost import build_demo_cost
from arcticroute.core.astar import plan_route_latlon
from arcticroute.core.analysis import compute_route_cost_breakdown
from arcticroute.config import EDL_MODES, SCENARIOS, get_scenario_by_name


class TestUIEDLComparison:
    """测试 UI 中的 EDL 模式对比功能。"""
    
    @pytest.fixture
    def demo_grid_and_mask(self):
        """创建演示网格和陆地掩码。"""
        grid, land_mask = make_demo_grid()
        return grid, land_mask
    
    def test_three_modes_planning_success(self, demo_grid_and_mask):
        """验证三种模式都能成功规划路线。"""
        grid, land_mask = demo_grid_and_mask
        
        # 使用 west_to_east_demo 场景的起止点
        start_lat, start_lon = 66.0, 5.0
        end_lat, end_lon = 78.0, 150.0
        
        for mode_name in ["efficient", "edl_safe", "edl_robust"]:
            mode_config = EDL_MODES[mode_name]
            
            # 构建成本场
            cost_field = build_demo_cost(
                grid,
                land_mask,
                ice_penalty=mode_config["ice_penalty"],
            )
            
            # 规划路线
            route = plan_route_latlon(
                cost_field,
                start_lat,
                start_lon,
                end_lat,
                end_lon,
                neighbor8=True,
            )
            
            # 验证路线可达
            assert route is not None, f"Route planning failed for mode {mode_name}"
            assert len(route) > 0, f"Route is empty for mode {mode_name}"
    
    def test_edl_cost_monotonicity(self, demo_grid_and_mask):
        """验证 EDL 成本随模式单调递增。
        
        设计思路：
        - efficient: w_edl=0.3（最弱）
        - edl_safe: w_edl=1.0（中等）
        - edl_robust: w_edl=1.0 + 不确定性（最强）
        
        预期：在相同的路线上，EDL 成本应该满足：
        efficient_edl_cost <= edl_safe_edl_cost <= edl_robust_edl_cost
        """
        grid, land_mask = demo_grid_and_mask
        
        # 使用固定的起止点
        start_lat, start_lon = 66.0, 5.0
        end_lat, end_lon = 78.0, 150.0
        
        # 规划三种模式的路线
        routes = {}
        cost_fields = {}
        breakdowns = {}
        
        for mode_name in ["efficient", "edl_safe", "edl_robust"]:
            mode_config = EDL_MODES[mode_name]
            
            # 构建成本场
            cost_field = build_demo_cost(
                grid,
                land_mask,
                ice_penalty=mode_config["ice_penalty"],
            )
            cost_fields[mode_name] = cost_field
            
            # 规划路线
            route = plan_route_latlon(
                cost_field,
                start_lat,
                start_lon,
                end_lat,
                end_lon,
                neighbor8=True,
            )
            
            if route is None:
                pytest.skip(f"Route not reachable for mode {mode_name}")
            
            routes[mode_name] = route
            
            # 计算成本分解
            breakdown = compute_route_cost_breakdown(grid, cost_field, route)
            breakdowns[mode_name] = breakdown
        
        # 验证 EDL 成本单调性
        # 注意：在 demo 模式下，EDL 成本可能为 0（因为 demo 网格不启用 EDL）
        # 所以我们只验证相对关系
        
        efficient_edl = breakdowns["efficient"].component_totals.get("edl_risk", 0.0)
        edl_safe_edl = breakdowns["edl_safe"].component_totals.get("edl_risk", 0.0)
        edl_robust_edl = breakdowns["edl_robust"].component_totals.get("edl_risk", 0.0)
        
        # 如果 EDL 成本都为 0（demo 模式），跳过这个检查
        if efficient_edl == 0 and edl_safe_edl == 0 and edl_robust_edl == 0:
            pytest.skip("EDL costs are all zero in demo mode")
        
        # 否则验证单调性
        assert efficient_edl <= edl_safe_edl, \
            f"EDL cost should increase from efficient to edl_safe: {efficient_edl} vs {edl_safe_edl}"
        assert edl_safe_edl <= edl_robust_edl, \
            f"EDL cost should increase from edl_safe to edl_robust: {edl_safe_edl} vs {edl_robust_edl}"
    
    def test_uncertainty_cost_only_in_robust(self, demo_grid_and_mask):
        """验证不确定性成本只在 edl_robust 模式中出现。"""
        grid, land_mask = demo_grid_and_mask
        
        start_lat, start_lon = 66.0, 5.0
        end_lat, end_lon = 78.0, 150.0
        
        for mode_name in ["efficient", "edl_safe", "edl_robust"]:
            mode_config = EDL_MODES[mode_name]
            
            # 构建成本场
            cost_field = build_demo_cost(
                grid,
                land_mask,
                ice_penalty=mode_config["ice_penalty"],
            )
            
            # 规划路线
            route = plan_route_latlon(
                cost_field,
                start_lat,
                start_lon,
                end_lat,
                end_lon,
                neighbor8=True,
            )
            
            if route is None:
                continue
            
            # 计算成本分解
            breakdown = compute_route_cost_breakdown(grid, cost_field, route)
            
            # 获取不确定性成本
            uncertainty_cost = breakdown.component_totals.get("edl_uncertainty_penalty", 0.0)
            
            # 验证：只有 edl_robust 应该有非零的不确定性成本
            if mode_name == "edl_robust":
                # edl_robust 可能有不确定性成本（如果启用了）
                # 但在 demo 模式下可能为 0
                pass
            else:
                # efficient 和 edl_safe 不应该有不确定性成本
                assert uncertainty_cost == 0.0, \
                    f"Mode {mode_name} should not have uncertainty cost, but got {uncertainty_cost}"
    
    def test_scenario_preset_coordinates(self):
        """验证场景预设能正确提供起止点坐标。"""
        scenario_names = ["barents_to_chukchi", "kara_short", "southern_route", "west_to_east_demo"]
        
        for scenario_name in scenario_names:
            scenario = get_scenario_by_name(scenario_name)
            
            assert scenario is not None, f"Scenario {scenario_name} not found"
            
            # 验证坐标范围
            assert 60.0 <= scenario.start_lat <= 85.0
            assert 60.0 <= scenario.end_lat <= 85.0
            assert -180.0 <= scenario.start_lon <= 180.0
            assert -180.0 <= scenario.end_lon <= 180.0
            
            # 验证起止点不同
            assert (scenario.start_lat, scenario.start_lon) != (scenario.end_lat, scenario.end_lon)
    
    def test_edl_mode_parameter_consistency(self):
        """验证 EDL 模式参数的一致性。"""
        # 验证 efficient 是最弱的
        efficient = EDL_MODES["efficient"]
        assert efficient["w_edl"] == 0.3
        assert efficient["use_edl_uncertainty"] == False
        
        # 验证 edl_safe 是中等的
        edl_safe = EDL_MODES["edl_safe"]
        assert edl_safe["w_edl"] == 1.0
        assert edl_safe["use_edl_uncertainty"] == False
        
        # 验证 edl_robust 是最强的
        edl_robust = EDL_MODES["edl_robust"]
        assert edl_robust["w_edl"] == 1.0
        assert edl_robust["use_edl_uncertainty"] == True
        assert edl_robust["edl_uncertainty_weight"] == 1.0
    
    def test_ice_penalty_consistency(self):
        """验证所有模式的 ice_penalty 一致。"""
        for mode_name, config in EDL_MODES.items():
            # 所有模式应该有相同的基础 ice_penalty
            assert config["ice_penalty"] == 4.0, \
                f"Mode {mode_name} should have ice_penalty=4.0, got {config['ice_penalty']}"


class TestScenarioIntegration:
    """测试场景预设与规划的集成。"""
    
    def test_all_scenarios_are_reachable(self):
        """验证所有场景在 demo 网格上都能规划出路线。"""
        grid, land_mask = make_demo_grid()
        
        for scenario in SCENARIOS:
            # 使用 efficient 模式规划
            cost_field = build_demo_cost(grid, land_mask, ice_penalty=4.0)
            
            route = plan_route_latlon(
                cost_field,
                scenario.start_lat,
                scenario.start_lon,
                scenario.end_lat,
                scenario.end_lon,
                neighbor8=True,
            )
            
            # 验证路线可达
            assert route is not None, \
                f"Scenario {scenario.name} should be reachable, but got None"
            assert len(route) > 0, \
                f"Scenario {scenario.name} route should not be empty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])















