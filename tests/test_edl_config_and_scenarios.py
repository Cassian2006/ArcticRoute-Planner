"""
测试 EDL 模式配置和场景预设的统一性。

验证：
1. EDL_MODES 配置的完整性和一致性
2. SCENARIOS 场景预设的完整性
3. 参数单调性（EDL 成本随模式递增）
4. UI 和 CLI 使用相同的配置
"""

from __future__ import annotations

import pytest
from arcticroute.config import (
    EDL_MODES,
    SCENARIOS,
    get_edl_mode_config,
    get_scenario_by_name,
    list_edl_modes,
    list_scenarios,
    list_scenario_descriptions,
)
from arcticroute.config.edl_modes import validate_edl_mode_config


class TestEDLModesConfiguration:
    """测试 EDL 模式配置。"""
    
    def test_edl_modes_exist(self):
        """验证三种 EDL 模式都存在。"""
        assert "efficient" in EDL_MODES
        assert "edl_safe" in EDL_MODES
        assert "edl_robust" in EDL_MODES
    
    def test_edl_modes_count(self):
        """验证 EDL 模式数量。"""
        assert len(EDL_MODES) == 3
    
    def test_edl_mode_config_completeness(self):
        """验证每个 EDL 模式配置的完整性。"""
        for mode_name, mode_config in EDL_MODES.items():
            # 验证必需的字段
            assert validate_edl_mode_config(mode_config), f"Mode {mode_name} has incomplete config"
            
            # 验证显示名称
            assert "display_name" in mode_config or "name" in mode_config
            
            # 验证参数范围
            assert 0.0 <= mode_config["w_edl"] <= 2.0, f"w_edl out of range for {mode_name}"
            assert 0.0 <= mode_config["ice_penalty"] <= 10.0, f"ice_penalty out of range for {mode_name}"
            assert 0.0 <= mode_config["edl_uncertainty_weight"] <= 3.0, f"edl_uncertainty_weight out of range for {mode_name}"
    
    def test_edl_mode_monotonicity(self):
        """验证 EDL 成本随模式单调递增。
        
        设计思路：
        - efficient: 弱 EDL（w_edl=0.3）
        - edl_safe: 中等 EDL（w_edl=1.0）
        - edl_robust: 强 EDL（w_edl=1.0 + 不确定性）
        
        预期：edl_safe 的 w_edl >= efficient 的 w_edl
             edl_robust 的总 EDL 成本 >= edl_safe 的总 EDL 成本
        """
        efficient = EDL_MODES["efficient"]
        edl_safe = EDL_MODES["edl_safe"]
        edl_robust = EDL_MODES["edl_robust"]
        
        # 验证 w_edl 单调性
        assert efficient["w_edl"] <= edl_safe["w_edl"], "w_edl should increase from efficient to edl_safe"
        assert edl_safe["w_edl"] <= edl_robust["w_edl"], "w_edl should increase from edl_safe to edl_robust"
        
        # 验证不确定性单调性
        assert efficient["use_edl_uncertainty"] == False
        assert edl_safe["use_edl_uncertainty"] == False
        assert edl_robust["use_edl_uncertainty"] == True
    
    def test_get_edl_mode_config(self):
        """测试 get_edl_mode_config 函数。"""
        config = get_edl_mode_config("efficient")
        assert config["w_edl"] == 0.3
        assert config["use_edl"] == True
        
        # 测试不存在的模式
        with pytest.raises(ValueError):
            get_edl_mode_config("nonexistent")
    
    def test_list_edl_modes(self):
        """测试 list_edl_modes 函数。"""
        modes = list_edl_modes()
        assert len(modes) == 3
        assert "efficient" in modes
        assert "edl_safe" in modes
        assert "edl_robust" in modes


class TestScenariosConfiguration:
    """测试场景预设配置。"""
    
    def test_scenarios_exist(self):
        """验证四个标准场景都存在。"""
        scenario_names = [s.name for s in SCENARIOS]
        assert "barents_to_chukchi" in scenario_names
        assert "kara_short" in scenario_names
        assert "southern_route" in scenario_names
        assert "west_to_east_demo" in scenario_names
    
    def test_scenarios_count(self):
        """验证场景数量。"""
        assert len(SCENARIOS) == 4
    
    def test_scenario_completeness(self):
        """验证每个场景的完整性。"""
        for scenario in SCENARIOS:
            # 验证必需的字段
            assert scenario.name is not None
            assert scenario.description is not None
            assert scenario.ym is not None
            assert 60.0 <= scenario.start_lat <= 85.0
            assert -180.0 <= scenario.start_lon <= 180.0
            assert 60.0 <= scenario.end_lat <= 85.0
            assert -180.0 <= scenario.end_lon <= 180.0
            assert scenario.vessel_profile is not None
    
    def test_get_scenario_by_name(self):
        """测试 get_scenario_by_name 函数。"""
        scenario = get_scenario_by_name("west_to_east_demo")
        assert scenario is not None
        assert scenario.start_lat == 66.0
        assert scenario.start_lon == 5.0
        assert scenario.end_lat == 78.0
        assert scenario.end_lon == 150.0
        
        # 测试不存在的场景
        scenario = get_scenario_by_name("nonexistent")
        assert scenario is None
    
    def test_list_scenarios(self):
        """测试 list_scenarios 函数。"""
        names = list_scenarios()
        assert len(names) == 4
        assert "barents_to_chukchi" in names
    
    def test_list_scenario_descriptions(self):
        """测试 list_scenario_descriptions 函数。"""
        descriptions = list_scenario_descriptions()
        assert len(descriptions) == 4
        assert "west_to_east_demo" in descriptions
        assert isinstance(descriptions["west_to_east_demo"], str)


class TestConfigurationConsistency:
    """测试配置的一致性（CLI 和 UI 使用相同的配置）。"""
    
    def test_cli_and_ui_use_same_edl_modes(self):
        """验证 CLI 和 UI 使用相同的 EDL 模式配置。"""
        # 从 scripts/run_edl_sensitivity_study.py 导入
        from scripts.run_edl_sensitivity_study import MODES as CLI_MODES
        
        # 验证 CLI 使用的是共享配置
        assert CLI_MODES is EDL_MODES, "CLI should use shared EDL_MODES"
    
    def test_cli_and_ui_use_same_scenarios(self):
        """验证 CLI 和 UI 使用相同的场景预设。"""
        # 从 scripts/run_edl_sensitivity_study.py 导入
        from scripts.run_edl_sensitivity_study import SCENARIOS as CLI_SCENARIOS
        
        # 验证 CLI 使用的是共享配置
        assert CLI_SCENARIOS is SCENARIOS, "CLI should use shared SCENARIOS"


class TestParameterRanges:
    """测试参数范围的合理性。"""
    
    def test_w_edl_range(self):
        """验证 w_edl 在合理范围内。"""
        for mode_name, config in EDL_MODES.items():
            w_edl = config["w_edl"]
            assert 0.0 <= w_edl <= 2.0, f"w_edl={w_edl} out of range for {mode_name}"
    
    def test_ice_penalty_range(self):
        """验证 ice_penalty 在合理范围内。"""
        for mode_name, config in EDL_MODES.items():
            ice_penalty = config["ice_penalty"]
            assert 2.0 <= ice_penalty <= 10.0, f"ice_penalty={ice_penalty} out of range for {mode_name}"
    
    def test_edl_uncertainty_weight_range(self):
        """验证 edl_uncertainty_weight 在合理范围内。"""
        for mode_name, config in EDL_MODES.items():
            weight = config["edl_uncertainty_weight"]
            assert 0.0 <= weight <= 3.0, f"edl_uncertainty_weight={weight} out of range for {mode_name}"
    
    def test_factor_ranges(self):
        """验证相对因子在合理范围内。"""
        for mode_name, config in EDL_MODES.items():
            ice_factor = config.get("ice_penalty_factor", 1.0)
            wave_factor = config.get("wave_weight_factor", 1.0)
            edl_factor = config.get("edl_weight_factor", 1.0)
            
            assert 0.1 <= ice_factor <= 5.0, f"ice_penalty_factor out of range for {mode_name}"
            assert 0.1 <= wave_factor <= 5.0, f"wave_weight_factor out of range for {mode_name}"
            assert 0.1 <= edl_factor <= 5.0, f"edl_weight_factor out of range for {mode_name}"


class TestScenarioGeography:
    """测试场景的地理合理性。"""
    
    def test_scenario_coordinates_in_arctic(self):
        """验证所有场景的坐标在北极地区。"""
        for scenario in SCENARIOS:
            # 纬度应该在 60-85°N
            assert 60.0 <= scenario.start_lat <= 85.0
            assert 60.0 <= scenario.end_lat <= 85.0
            
            # 经度应该在 -180-180°
            assert -180.0 <= scenario.start_lon <= 180.0
            assert -180.0 <= scenario.end_lon <= 180.0
    
    def test_scenario_start_end_different(self):
        """验证每个场景的起点和终点不同。"""
        for scenario in SCENARIOS:
            assert (scenario.start_lat, scenario.start_lon) != (scenario.end_lat, scenario.end_lon)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])















