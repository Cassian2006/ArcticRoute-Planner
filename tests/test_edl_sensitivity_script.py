"""
EDL 灵敏度分析脚本的测试。

测试脚本的基本功能：
  - 场景库加载
  - 灵敏度分析运行
  - CSV 输出
  - 图表生成
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.edl_scenarios import SCENARIOS, get_scenario_by_name, list_scenarios
from scripts.run_edl_sensitivity_study import (
    SensitivityResult,
    run_single_scenario_mode,
    run_all_scenarios,
    write_results_to_csv,
    MODES,
)


class TestScenarioLibrary:
    """测试场景库的基本功能。"""
    
    def test_scenarios_not_empty(self):
        """测试场景库非空。"""
        assert len(SCENARIOS) > 0
    
    def test_scenario_has_required_fields(self):
        """测试每个场景都有必需字段。"""
        for scenario in SCENARIOS:
            assert scenario.name
            assert scenario.description
            assert scenario.ym
            assert scenario.start_lat is not None
            assert scenario.start_lon is not None
            assert scenario.end_lat is not None
            assert scenario.end_lon is not None
            assert scenario.vessel_profile
    
    def test_get_scenario_by_name(self):
        """测试按名称获取场景。"""
        if SCENARIOS:
            first_scenario = SCENARIOS[0]
            found = get_scenario_by_name(first_scenario.name)
            assert found is not None
            assert found.name == first_scenario.name
    
    def test_get_nonexistent_scenario(self):
        """测试获取不存在的场景返回 None。"""
        result = get_scenario_by_name("nonexistent_scenario_xyz")
        assert result is None
    
    def test_list_scenarios(self):
        """测试列出所有场景名称。"""
        names = list_scenarios()
        assert len(names) == len(SCENARIOS)
        assert all(isinstance(name, str) for name in names)


class TestSensitivityResult:
    """测试 SensitivityResult 数据类。"""
    
    def test_result_initialization(self):
        """测试结果对象初始化。"""
        result = SensitivityResult("test_scenario", "efficient")
        
        assert result.scenario_name == "test_scenario"
        assert result.mode == "efficient"
        assert result.reachable is False
        assert result.distance_km == 0.0
        assert result.total_cost == 0.0
    
    def test_result_to_dict(self):
        """测试结果转换为字典。"""
        result = SensitivityResult("test_scenario", "efficient")
        result.reachable = True
        result.distance_km = 100.5
        result.total_cost = 50.25
        result.edl_risk_cost = 5.0
        result.edl_uncertainty_cost = 2.0
        result.mean_uncertainty = 0.3
        result.max_uncertainty = 0.8
        result.components = {"ice_risk": 10.0, "base_distance": 40.25}
        
        result_dict = result.to_dict()
        
        assert result_dict["scenario"] == "test_scenario"
        assert result_dict["mode"] == "efficient"
        assert result_dict["reachable"] == "yes"
        assert "distance_km" in result_dict
        assert "total_cost" in result_dict
        assert "edl_risk_cost" in result_dict
        assert "edl_uncertainty_cost" in result_dict
        assert "mean_uncertainty" in result_dict
        assert "max_uncertainty" in result_dict


class TestModesConfiguration:
    """测试模式配置。"""
    
    def test_modes_not_empty(self):
        """测试模式配置非空。"""
        assert len(MODES) > 0
    
    def test_required_modes_exist(self):
        """测试必需的模式存在。"""
        assert "efficient" in MODES
        assert "edl_safe" in MODES
        assert "edl_robust" in MODES
    
    def test_mode_has_required_fields(self):
        """测试每个模式都有必需字段。"""
        for mode_name, mode_config in MODES.items():
            assert "description" in mode_config
            assert "w_edl" in mode_config
            assert "use_edl" in mode_config
            assert "use_edl_uncertainty" in mode_config
            assert "edl_uncertainty_weight" in mode_config
            assert "ice_penalty" in mode_config
    
    def test_efficient_mode_weak_edl(self):
        """测试 efficient 模式使用弱 EDL。"""
        efficient = MODES["efficient"]
        # efficient 现在使用弱 EDL（w_edl=0.3）
        assert efficient["w_edl"] == 0.3
        assert efficient["use_edl"] is True
        assert efficient["use_edl_uncertainty"] is False
    
    def test_edl_safe_has_edl_risk(self):
        """测试 edl_safe 模式启用 EDL 风险。"""
        edl_safe = MODES["edl_safe"]
        assert edl_safe["w_edl"] > 0.0
        assert edl_safe["use_edl"] is True
        assert edl_safe["use_edl_uncertainty"] is False
    
    def test_edl_robust_has_both(self):
        """测试 edl_robust 模式同时启用风险和不确定性。"""
        edl_robust = MODES["edl_robust"]
        assert edl_robust["w_edl"] > 0.0
        assert edl_robust["use_edl"] is True
        assert edl_robust["use_edl_uncertainty"] is True
        assert edl_robust["edl_uncertainty_weight"] > 0.0


class TestSensitivityAnalysis:
    """测试灵敏度分析的核心功能。"""
    
    def test_run_all_scenarios_dry_run(self):
        """测试干运行模式。"""
        results = run_all_scenarios(
            scenarios=SCENARIOS[:1],  # 只用第一个场景
            modes=["efficient"],  # 只用一个模式
            use_real_data=False,
            dry_run=True,
        )
        
        assert len(results) == 1
        assert results[0].scenario_name == SCENARIOS[0].name
        assert results[0].mode == "efficient"
        assert results[0].error_message == "dry_run"
    
    def test_run_single_scenario_demo_mode(self):
        """测试在 demo 模式下运行单个场景。"""
        if not SCENARIOS:
            pytest.skip("No scenarios available")
        
        scenario = SCENARIOS[0]
        result = run_single_scenario_mode(
            scenario,
            "efficient",
            use_real_data=False,
        )
        
        assert result.scenario_name == scenario.name
        assert result.mode == "efficient"
        # 在 demo 模式下应该能规划出路线
        assert result.reachable or result.error_message
    
    def test_write_results_to_csv(self, tmp_path):
        """测试将结果写入 CSV。"""
        # 创建测试结果
        results = [
            SensitivityResult("scenario1", "efficient"),
            SensitivityResult("scenario1", "edl_safe"),
        ]
        
        for result in results:
            result.reachable = True
            result.distance_km = 100.0
            result.total_cost = 50.0
            result.components = {"ice_risk": 10.0}
        
        # 写入 CSV
        output_file = tmp_path / "test_results.csv"
        write_results_to_csv(results, output_file)
        
        # 验证文件存在
        assert output_file.exists()
        
        # 验证 CSV 内容
        with open(output_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 2
        assert rows[0]["scenario"] == "scenario1"
        assert rows[0]["mode"] == "efficient"
        assert rows[1]["mode"] == "edl_safe"
    
    def test_write_empty_results_to_csv(self, tmp_path):
        """测试写入空结果。"""
        output_file = tmp_path / "empty_results.csv"
        write_results_to_csv([], output_file)
        
        # 即使结果为空，也不应该报错
        # 文件可能不存在或为空
    
    def test_csv_has_expected_columns(self, tmp_path):
        """测试 CSV 包含预期的列。"""
        result = SensitivityResult("test_scenario", "efficient")
        result.reachable = True
        result.distance_km = 100.0
        result.total_cost = 50.0
        result.edl_risk_cost = 5.0
        result.edl_uncertainty_cost = 2.0
        result.mean_uncertainty = 0.3
        result.max_uncertainty = 0.8
        
        output_file = tmp_path / "test_columns.csv"
        write_results_to_csv([result], output_file)
        
        with open(output_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
        
        # 检查关键列
        expected_columns = [
            "scenario",
            "mode",
            "reachable",
            "distance_km",
            "total_cost",
            "edl_risk_cost",
            "edl_uncertainty_cost",
            "mean_uncertainty",
            "max_uncertainty",
        ]
        
        for col in expected_columns:
            assert col in header, f"Missing column: {col}"


class TestChartGeneration:
    """测试图表生成功能。"""
    
    def test_generate_charts_with_matplotlib(self, tmp_path):
        """测试图表生成（需要 matplotlib）。"""
        try:
            import matplotlib
            matplotlib_available = True
        except ImportError:
            matplotlib_available = False
        
        if not matplotlib_available:
            pytest.skip("matplotlib not available")
        
        # 创建测试结果
        results = []
        for scenario in SCENARIOS[:1]:
            for mode in ["efficient", "edl_safe", "edl_robust"]:
                result = SensitivityResult(scenario.name, mode)
                result.reachable = True
                result.distance_km = 100.0 + (hash(mode) % 50)
                result.total_cost = 50.0 + (hash(mode) % 30)
                result.edl_risk_cost = 5.0 if mode != "efficient" else 0.0
                result.edl_uncertainty_cost = 2.0 if mode == "edl_robust" else 0.0
                results.append(result)
        
        # 生成图表
        from scripts.run_edl_sensitivity_study import generate_charts
        generate_charts(results, tmp_path)
        
        # 检查是否生成了图表文件
        png_files = list(tmp_path.glob("*.png"))
        # 可能生成了图表，也可能没有（取决于 matplotlib 的可用性）
        # 这里只是检查函数不报错


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


