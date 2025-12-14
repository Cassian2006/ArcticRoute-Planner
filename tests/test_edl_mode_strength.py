"""
EDL 模式强度测试。

验证三种模式（efficient / edl_safe / edl_robust）的相对强度关系：
  - efficient: 弱 EDL（w_edl = 0.3）
  - edl_safe: 中等 EDL（w_edl = 1.0）
  - edl_robust: 强 EDL（w_edl = 1.0 + uncertainty）

测试断言：
  1. 三种模式都能规划出路线（或都不可达）
  2. efficient 的 EDL 成本 > 0（启用了 EDL）
  3. efficient 的 EDL 成本 < edl_safe 的 EDL 成本
  4. edl_safe 的 EDL 成本 <= edl_robust 的 EDL 成本
  5. edl_robust 的不确定性成本 >= edl_safe 的不确定性成本
"""

from __future__ import annotations

import numpy as np
import pytest

from arcticroute.core.grid import make_demo_grid
from arcticroute.core.cost import build_cost_from_real_env, build_demo_cost
from arcticroute.core.env_real import RealEnvLayers
from arcticroute.core.astar import plan_route_latlon
from arcticroute.core.analysis import compute_route_cost_breakdown
from arcticroute.core.eco.vessel_profiles import get_default_profiles

from scripts.run_edl_sensitivity_study import MODES


class TestEDLModeStrength:
    """测试三种 EDL 模式的相对强度。"""
    
    @pytest.fixture
    def demo_grid_and_landmask(self):
        """创建 demo 网格和陆地掩码。"""
        grid, land_mask = make_demo_grid()
        return grid, land_mask
    
    @pytest.fixture
    def dummy_env(self, demo_grid_and_landmask):
        """创建虚拟环境数据（全零或常数）。"""
        grid, _ = demo_grid_and_landmask
        ny, nx = grid.shape()
        
        # 创建虚拟环境：
        # - sic: 全 0.5（中等冰况）
        # - wave_swh: 全 2.0（中等波浪）
        # - ice_thickness_m: 全 0.5（中等冰厚）
        env = RealEnvLayers(
            sic=np.full((ny, nx), 0.5, dtype=float),
            wave_swh=np.full((ny, nx), 2.0, dtype=float),
            ice_thickness_m=np.full((ny, nx), 0.5, dtype=float),
        )
        
        return env
    
    def test_modes_configuration(self):
        """测试模式配置的基本属性。"""
        # 检查三种模式都存在
        assert "efficient" in MODES
        assert "edl_safe" in MODES
        assert "edl_robust" in MODES
        
        # 检查 efficient 现在启用了 EDL
        efficient = MODES["efficient"]
        assert efficient["use_edl"] is True, "efficient 应该启用 EDL"
        assert efficient["w_edl"] > 0.0, "efficient 的 w_edl 应该 > 0"
        assert efficient["use_edl_uncertainty"] is False, "efficient 不应该启用不确定性"
        
        # 检查 edl_safe
        edl_safe = MODES["edl_safe"]
        assert edl_safe["use_edl"] is True
        assert edl_safe["w_edl"] > efficient["w_edl"], "edl_safe 的 w_edl 应该 > efficient"
        assert edl_safe["use_edl_uncertainty"] is False
        
        # 检查 edl_robust
        edl_robust = MODES["edl_robust"]
        assert edl_robust["use_edl"] is True
        assert edl_robust["w_edl"] >= edl_safe["w_edl"], "edl_robust 的 w_edl 应该 >= edl_safe"
        assert edl_robust["use_edl_uncertainty"] is True, "edl_robust 应该启用不确定性"
        assert edl_robust["edl_uncertainty_weight"] > 0.0
    
    def test_edl_weight_hierarchy(self):
        """测试 EDL 权重的层级关系。"""
        efficient_w = MODES["efficient"]["w_edl"]
        safe_w = MODES["edl_safe"]["w_edl"]
        robust_w = MODES["edl_robust"]["w_edl"]
        
        # 权重应该满足：efficient < safe <= robust
        assert efficient_w > 0.0, "efficient 应该有 EDL 权重"
        assert efficient_w < safe_w, f"efficient ({efficient_w}) 应该 < safe ({safe_w})"
        assert safe_w <= robust_w, f"safe ({safe_w}) 应该 <= robust ({robust_w})"
        
        # efficient 的权重应该约为 safe 的 1/3 或更小
        ratio = efficient_w / safe_w
        assert ratio < 0.5, f"efficient/safe 的比例 ({ratio:.2f}) 应该 < 0.5"
    
    def test_cost_field_construction(self, demo_grid_and_landmask, dummy_env):
        """测试三种模式的成本场构建。"""
        grid, land_mask = demo_grid_and_landmask
        
        cost_fields = {}
        
        for mode_name in ["efficient", "edl_safe", "edl_robust"]:
            cfg = MODES[mode_name]
            
            # 构建成本场
            cost_field = build_cost_from_real_env(
                grid=grid,
                land_mask=land_mask,
                env=dummy_env,
                ice_penalty=cfg["ice_penalty"],
                wave_penalty=0.0,
                vessel_profile=None,
                w_edl=cfg["w_edl"],
                use_edl=cfg["use_edl"],
                use_edl_uncertainty=cfg["use_edl_uncertainty"],
                edl_uncertainty_weight=cfg["edl_uncertainty_weight"],
            )
            
            cost_fields[mode_name] = cost_field
            
            # 检查成本场的基本属性
            assert cost_field.cost is not None
            assert cost_field.cost.shape == grid.shape()
            assert np.any(np.isfinite(cost_field.cost)), f"{mode_name} 的成本场应该有有限值"
    
    def test_route_planning_and_cost_accumulation(self, demo_grid_and_landmask, dummy_env):
        """测试三种模式的路线规划和成本积累。"""
        grid, land_mask = demo_grid_and_landmask
        
        # 定义起终点（使用 demo 网格的合理坐标）
        start_lat, start_lon = 66.0, 5.0
        end_lat, end_lon = 78.0, 150.0
        
        edl_costs = {}
        total_costs = {}
        reachable = {}
        
        for mode_name in ["efficient", "edl_safe", "edl_robust"]:
            cfg = MODES[mode_name]
            
            # 构建成本场
            cost_field = build_cost_from_real_env(
                grid=grid,
                land_mask=land_mask,
                env=dummy_env,
                ice_penalty=cfg["ice_penalty"],
                wave_penalty=0.0,
                vessel_profile=None,
                w_edl=cfg["w_edl"],
                use_edl=cfg["use_edl"],
                use_edl_uncertainty=cfg["use_edl_uncertainty"],
                edl_uncertainty_weight=cfg["edl_uncertainty_weight"],
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
            
            reachable[mode_name] = route is not None
            
            if route:
                # 计算成本分解
                breakdown = compute_route_cost_breakdown(grid, cost_field, route)
                
                # 提取 EDL 相关成本
                edl_risk = breakdown.component_totals.get("edl_risk", 0.0)
                edl_uncertainty = breakdown.component_totals.get("edl_uncertainty_penalty", 0.0)
                
                edl_costs[mode_name] = edl_risk + edl_uncertainty
                total_costs[mode_name] = breakdown.total_cost
                
                print(
                    f"\n{mode_name}:"
                    f"\n  total_cost: {breakdown.total_cost:.4f}"
                    f"\n  edl_risk: {edl_risk:.4f}"
                    f"\n  edl_uncertainty: {edl_uncertainty:.4f}"
                    f"\n  edl_total: {edl_costs[mode_name]:.4f}"
                )
        
        # 检查至少有一个模式可达
        assert any(reachable.values()), "至少有一个模式应该可达"
        
        # 如果三个模式都可达，检查 EDL 成本的相对关系
        if all(reachable.values()):
            efficient_edl = edl_costs.get("efficient", 0.0)
            safe_edl = edl_costs.get("edl_safe", 0.0)
            robust_edl = edl_costs.get("edl_robust", 0.0)
            
            # 检查 efficient 的 EDL 成本 > 0（启用了 EDL）
            assert efficient_edl > 0.0, "efficient 应该有 EDL 成本（现在启用了 EDL）"
            
            # 检查相对关系：efficient < safe <= robust
            assert efficient_edl < safe_edl, (
                f"efficient EDL ({efficient_edl:.4f}) 应该 < safe ({safe_edl:.4f})"
            )
            assert safe_edl <= robust_edl, (
                f"safe EDL ({safe_edl:.4f}) 应该 <= robust ({robust_edl:.4f})"
            )
    
    def test_uncertainty_cost_hierarchy(self, demo_grid_and_landmask, dummy_env):
        """测试不确定性成本的层级关系。"""
        grid, land_mask = demo_grid_and_landmask
        
        start_lat, start_lon = 66.0, 5.0
        end_lat, end_lon = 78.0, 150.0
        
        uncertainty_costs = {}
        
        for mode_name in ["efficient", "edl_safe", "edl_robust"]:
            cfg = MODES[mode_name]
            
            cost_field = build_cost_from_real_env(
                grid=grid,
                land_mask=land_mask,
                env=dummy_env,
                ice_penalty=cfg["ice_penalty"],
                wave_penalty=0.0,
                vessel_profile=None,
                w_edl=cfg["w_edl"],
                use_edl=cfg["use_edl"],
                use_edl_uncertainty=cfg["use_edl_uncertainty"],
                edl_uncertainty_weight=cfg["edl_uncertainty_weight"],
            )
            
            route = plan_route_latlon(
                cost_field,
                start_lat,
                start_lon,
                end_lat,
                end_lon,
                neighbor8=True,
            )
            
            if route:
                breakdown = compute_route_cost_breakdown(grid, cost_field, route)
                unc_cost = breakdown.component_totals.get("edl_uncertainty_penalty", 0.0)
                uncertainty_costs[mode_name] = unc_cost
        
        # 检查不确定性成本的关系
        if all(mode in uncertainty_costs for mode in ["efficient", "edl_safe", "edl_robust"]):
            efficient_unc = uncertainty_costs["efficient"]
            safe_unc = uncertainty_costs["edl_safe"]
            robust_unc = uncertainty_costs["edl_robust"]
            
            # efficient 和 safe 都不应该有不确定性成本
            assert efficient_unc == 0.0, "efficient 不应该有不确定性成本"
            assert safe_unc == 0.0, "edl_safe 不应该有不确定性成本"
            
            # robust 可能有不确定性成本（如果启用了）
            # 这里只是检查它不是负数
            assert robust_unc >= 0.0, "不确定性成本应该 >= 0"
    
    def test_mode_descriptions(self):
        """测试模式描述的一致性。"""
        # 检查每个模式都有描述
        for mode_name, cfg in MODES.items():
            assert "description" in cfg
            assert isinstance(cfg["description"], str)
            assert len(cfg["description"]) > 0
        
        # 检查描述中反映了模式的特点
        efficient_desc = MODES["efficient"]["description"].lower()
        assert "弱" in efficient_desc or "weak" in efficient_desc or "efficient" in efficient_desc
        
        safe_desc = MODES["edl_safe"]["description"].lower()
        assert "中等" in safe_desc or "safe" in safe_desc
        
        robust_desc = MODES["edl_robust"]["description"].lower()
        assert "强" in robust_desc or "robust" in robust_desc or "uncertainty" in robust_desc


class TestUIRouteProfilesConsistency:
    """测试 UI 中的 ROUTE_PROFILES 与脚本中的 MODES 的一致性。"""
    
    def test_route_profiles_exist(self):
        """测试 UI 中的 ROUTE_PROFILES 存在。"""
        try:
            from arcticroute.ui.planner_minimal import ROUTE_PROFILES
            assert ROUTE_PROFILES is not None
            assert len(ROUTE_PROFILES) == 3
        except ImportError:
            pytest.skip("UI module not available")
    
    def test_route_profiles_keys_match_modes(self):
        """测试 ROUTE_PROFILES 的 key 与 MODES 的 key 一致。"""
        try:
            from arcticroute.ui.planner_minimal import ROUTE_PROFILES
            
            profile_keys = {p["key"] for p in ROUTE_PROFILES}
            mode_keys = set(MODES.keys())
            
            assert profile_keys == mode_keys, (
                f"ROUTE_PROFILES keys {profile_keys} 应该与 MODES keys {mode_keys} 一致"
            )
        except ImportError:
            pytest.skip("UI module not available")
    
    def test_route_profiles_edl_weight_factors(self):
        """测试 ROUTE_PROFILES 的 EDL 权重因子与 MODES 的相对关系一致。"""
        try:
            from arcticroute.ui.planner_minimal import ROUTE_PROFILES
            
            # 构建 key -> profile 映射
            profiles = {p["key"]: p for p in ROUTE_PROFILES}
            
            # 检查 edl_weight_factor 的相对关系
            efficient_factor = profiles["efficient"]["edl_weight_factor"]
            safe_factor = profiles["edl_safe"]["edl_weight_factor"]
            robust_factor = profiles["edl_robust"]["edl_weight_factor"]
            
            # efficient 的因子应该 < safe 的因子
            assert efficient_factor < safe_factor, (
                f"efficient factor ({efficient_factor}) 应该 < safe ({safe_factor})"
            )
            
            # safe 和 robust 的因子应该相等或 robust 更大
            assert safe_factor <= robust_factor, (
                f"safe factor ({safe_factor}) 应该 <= robust ({robust_factor})"
            )
        except ImportError:
            pytest.skip("UI module not available")
    
    def test_route_profiles_uncertainty_settings(self):
        """测试 ROUTE_PROFILES 的不确定性设置。"""
        try:
            from arcticroute.ui.planner_minimal import ROUTE_PROFILES
            
            profiles = {p["key"]: p for p in ROUTE_PROFILES}
            
            # efficient 和 safe 不应该启用不确定性
            assert profiles["efficient"]["use_edl_uncertainty"] is False
            assert profiles["edl_safe"]["use_edl_uncertainty"] is False
            
            # robust 应该启用不确定性
            assert profiles["edl_robust"]["use_edl_uncertainty"] is True
            assert profiles["edl_robust"]["edl_uncertainty_weight"] > 0.0
        except ImportError:
            pytest.skip("UI module not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

