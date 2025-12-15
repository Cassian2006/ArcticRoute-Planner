"""
实验导出与 UI 下载测试。

测试 run_single_case、run_case_grid 等导出功能。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from arcticroute.experiments.runner import (
    SingleRunResult,
    run_single_case,
    run_case_grid,
)


class TestSingleRunResult:
    """测试 SingleRunResult 数据类。"""
    
    def test_single_run_result_creation(self):
        """测试 SingleRunResult 的创建。"""
        result = SingleRunResult(
            scenario="barents_to_chukchi",
            mode="efficient",
            reachable=True,
            distance_km=2165.2,
            total_cost=25.6,
            edl_risk_cost=1.9,
            edl_unc_cost=6.7,
            ice_cost=10.0,
            wave_cost=2.0,
            ice_class_soft_cost=None,
            ice_class_hard_cost=None,
            meta={
                "ym": "202412",
                "use_real_data": False,
                "vessel_profile": "panamax",
            },
        )
        
        assert result.scenario == "barents_to_chukchi"
        assert result.mode == "efficient"
        assert result.reachable is True
        assert result.distance_km == 2165.2
        assert result.total_cost == 25.6
    
    def test_single_run_result_to_dict(self):
        """测试 SingleRunResult 转换为字典。"""
        result = SingleRunResult(
            scenario="barents_to_chukchi",
            mode="efficient",
            reachable=True,
            distance_km=2165.2,
            total_cost=25.6,
            edl_risk_cost=1.9,
            edl_unc_cost=6.7,
            ice_cost=10.0,
            wave_cost=2.0,
            ice_class_soft_cost=None,
            ice_class_hard_cost=None,
            meta={"ym": "202412"},
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["scenario"] == "barents_to_chukchi"
        assert result_dict["mode"] == "efficient"
        assert result_dict["reachable"] is True
        assert "meta" in result_dict
    
    def test_single_run_result_to_flat_dict(self):
        """测试 SingleRunResult 转换为扁平字典。"""
        result = SingleRunResult(
            scenario="barents_to_chukchi",
            mode="efficient",
            reachable=True,
            distance_km=2165.2,
            total_cost=25.6,
            edl_risk_cost=1.9,
            edl_unc_cost=6.7,
            ice_cost=10.0,
            wave_cost=2.0,
            ice_class_soft_cost=None,
            ice_class_hard_cost=None,
            meta={"ym": "202412", "vessel_profile": "panamax"},
        )
        
        flat_dict = result.to_flat_dict()
        
        assert isinstance(flat_dict, dict)
        assert flat_dict["scenario"] == "barents_to_chukchi"
        assert "meta" not in flat_dict
        assert "meta_ym" in flat_dict
        assert flat_dict["meta_ym"] == "202412"
        assert flat_dict["meta_vessel_profile"] == "panamax"


class TestRunSingleCase:
    """测试 run_single_case 函数。"""
    
    def test_run_single_case_efficient_demo(self):
        """测试运行单个案例（efficient 模式，demo 数据）。"""
        result = run_single_case(
            scenario="barents_to_chukchi",
            mode="efficient",
            use_real_data=False,
        )
        
        assert result.scenario == "barents_to_chukchi"
        assert result.mode == "efficient"
        assert isinstance(result.reachable, bool)
        
        # 如果可达，检查必要字段
        if result.reachable:
            assert result.distance_km is not None
            assert result.total_cost is not None
            assert result.distance_km > 0
            assert result.total_cost >= 0
    
    def test_run_single_case_edl_safe_demo(self):
        """测试运行单个案例（edl_safe 模式，demo 数据）。"""
        result = run_single_case(
            scenario="kara_short",
            mode="edl_safe",
            use_real_data=False,
        )
        
        assert result.scenario == "kara_short"
        assert result.mode == "edl_safe"
        assert isinstance(result.reachable, bool)
    
    def test_run_single_case_edl_robust_demo(self):
        """测试运行单个案例（edl_robust 模式，demo 数据）。"""
        result = run_single_case(
            scenario="southern_route",
            mode="edl_robust",
            use_real_data=False,
        )
        
        assert result.scenario == "southern_route"
        assert result.mode == "edl_robust"
        assert isinstance(result.reachable, bool)
    
    def test_run_single_case_invalid_scenario(self):
        """测试运行单个案例（无效场景）。"""
        with pytest.raises(ValueError):
            run_single_case(
                scenario="invalid_scenario",
                mode="efficient",
                use_real_data=False,
            )
    
    def test_run_single_case_invalid_mode(self):
        """测试运行单个案例（无效模式）。"""
        with pytest.raises(ValueError):
            run_single_case(
                scenario="barents_to_chukchi",
                mode="invalid_mode",
                use_real_data=False,
            )
    
    def test_run_single_case_meta_fields(self):
        """测试运行单个案例的元数据字段。"""
        result = run_single_case(
            scenario="barents_to_chukchi",
            mode="efficient",
            use_real_data=False,
        )
        
        assert "ym" in result.meta
        assert "use_real_data" in result.meta
        assert "cost_mode" in result.meta
        assert "vessel_profile" in result.meta
        assert "edl_backend" in result.meta
        assert result.meta["use_real_data"] is False


class TestRunCaseGrid:
    """测试 run_case_grid 函数。"""
    
    def test_run_case_grid_basic(self):
        """测试运行案例网格（基础）。"""
        df = run_case_grid(
            scenarios=["barents_to_chukchi", "kara_short"],
            modes=["efficient", "edl_safe"],
            use_real_data=False,
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # 2 scenarios * 2 modes
        assert "scenario" in df.columns
        assert "mode" in df.columns
    
    def test_run_case_grid_shape(self):
        """测试运行案例网格的形状。"""
        scenarios = ["barents_to_chukchi", "kara_short", "southern_route"]
        modes = ["efficient", "edl_safe", "edl_robust"]
        
        df = run_case_grid(
            scenarios=scenarios,
            modes=modes,
            use_real_data=False,
        )
        
        assert df.shape[0] == len(scenarios) * len(modes)  # 3*3=9
    
    def test_run_case_grid_columns(self):
        """测试运行案例网格的列。"""
        df = run_case_grid(
            scenarios=["barents_to_chukchi"],
            modes=["efficient"],
            use_real_data=False,
        )
        
        # 检查关键列
        expected_cols = {"scenario", "mode", "reachable", "distance_km", "total_cost"}
        assert expected_cols <= set(df.columns)
    
    def test_run_case_grid_to_csv(self):
        """测试运行案例网格并导出为 CSV。"""
        df = run_case_grid(
            scenarios=["barents_to_chukchi"],
            modes=["efficient"],
            use_real_data=False,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_grid.csv"
            df.to_csv(csv_path, index=False)
            
            assert csv_path.exists()
            
            # 读回并验证
            df_read = pd.read_csv(csv_path)
            assert len(df_read) == len(df)
            assert set(df_read.columns) == set(df.columns)
    
    def test_run_case_grid_to_json(self):
        """测试运行案例网格并导出为 JSON。"""
        df = run_case_grid(
            scenarios=["barents_to_chukchi"],
            modes=["efficient"],
            use_real_data=False,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "test_grid.json"
            
            # 转换为 JSON 格式
            records = df.to_dict(orient="records")
            with open(json_path, "w") as f:
                json.dump(records, f)
            
            assert json_path.exists()
            
            # 读回并验证
            with open(json_path) as f:
                data = json.load(f)
            
            assert len(data) == len(df)


class TestExportFormats:
    """测试导出格式的一致性。"""
    
    def test_single_case_export_consistency(self):
        """测试单个案例导出的一致性。"""
        result = run_single_case(
            scenario="barents_to_chukchi",
            mode="efficient",
            use_real_data=False,
        )
        
        # 转换为 DataFrame
        df = pd.DataFrame([result.to_flat_dict()])
        
        # 检查关键字段
        assert df.loc[0, "scenario"] == "barents_to_chukchi"
        assert df.loc[0, "mode"] == "efficient"
        assert df.loc[0, "reachable"] == result.reachable
        
        if result.reachable:
            assert df.loc[0, "distance_km"] == result.distance_km
            assert df.loc[0, "total_cost"] == result.total_cost
    
    def test_grid_export_consistency(self):
        """测试网格导出的一致性。"""
        df = run_case_grid(
            scenarios=["barents_to_chukchi"],
            modes=["efficient", "edl_safe"],
            use_real_data=False,
        )
        
        # 检查每一行
        for idx, row in df.iterrows():
            assert row["scenario"] == "barents_to_chukchi"
            assert row["mode"] in ["efficient", "edl_safe"]
            assert isinstance(row["reachable"], (bool, int))


class TestExportEdgeCases:
    """测试导出的边界情况。"""
    
    def test_unreachable_case_export(self):
        """测试不可达案例的导出。"""
        # 尝试找到一个不可达的案例
        result = run_single_case(
            scenario="barents_to_chukchi",
            mode="efficient",
            use_real_data=False,
        )
        
        # 无论是否可达，都应该能导出
        result_dict = result.to_flat_dict()
        assert "scenario" in result_dict
        assert "mode" in result_dict
        assert "reachable" in result_dict
    
    def test_empty_grid_export(self):
        """测试空网格的导出。"""
        df = run_case_grid(
            scenarios=[],
            modes=[],
            use_real_data=False,
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    def test_single_scenario_single_mode(self):
        """测试单个场景单个模式的导出。"""
        df = run_case_grid(
            scenarios=["barents_to_chukchi"],
            modes=["efficient"],
            use_real_data=False,
        )
        
        assert len(df) == 1
        assert df.loc[0, "scenario"] == "barents_to_chukchi"
        assert df.loc[0, "mode"] == "efficient"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])













