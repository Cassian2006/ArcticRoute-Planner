"""
AIS Phase 1 集成测试：验证完整的 AIS 数据流
"""

import numpy as np
import pytest
from pathlib import Path

from arcticroute.core.ais_ingest import inspect_ais_csv, build_ais_density_for_grid
from arcticroute.core.grid import make_demo_grid
from arcticroute.core.cost import build_cost_from_real_env
from arcticroute.core.env_real import RealEnvLayers


def get_real_ais_csv_path() -> str:
    """获取真实 AIS CSV 路径。"""
    return str(Path(__file__).resolve().parents[1] / "data_real" / "ais" / "raw" / "ais_2024_sample.csv")


def test_ais_phase1_complete_workflow():
    """测试完整的 AIS Phase 1 工作流程。"""
    # Step 0: 检查 AIS 数据文件
    ais_csv_path = get_real_ais_csv_path()
    assert Path(ais_csv_path).exists(), f"AIS CSV 文件不存在：{ais_csv_path}"
    
    # Step 1: 探测 AIS schema
    summary = inspect_ais_csv(ais_csv_path, sample_n=100)
    assert summary.has_mmsi, "缺少 mmsi 列"
    assert summary.has_lat, "缺少 lat 列"
    assert summary.has_lon, "缺少 lon 列"
    assert summary.has_timestamp, "缺少 timestamp 列"
    assert summary.num_rows > 0, "AIS 数据为空"
    print(f"✓ Step 1 通过：AIS schema 探测成功，{summary.num_rows} 行数据")
    
    # Step 2: 构建 AIS 密度场
    grid, land_mask = make_demo_grid(ny=30, nx=30)
    ais_result = build_ais_density_for_grid(
        ais_csv_path,
        grid.lat2d,
        grid.lon2d,
        max_rows=50000,
    )
    assert ais_result.da.shape == (30, 30), "密度场形状不对"
    assert ais_result.num_binned > 0, "没有有效的 AIS 点被栅格化"
    assert np.max(ais_result.da.values) <= 1.0, "密度场未正确归一化"
    print(f"✓ Step 2 通过：AIS 栅格化成功，{ais_result.num_binned}/{ais_result.num_points} 有效点")
    
    # Step 3: 集成到成本模型
    env = RealEnvLayers(
        grid=grid,
        sic=np.ones((30, 30), dtype=float) * 0.3,
        wave_swh=None,
        ice_thickness_m=None,
        land_mask=land_mask,
    )
    
    # 不使用 AIS
    cost_without_ais = build_cost_from_real_env(
        grid, land_mask, env,
        ais_density=None,
        ais_weight=0.0,
    )
    
    # 使用 AIS
    cost_with_ais = build_cost_from_real_env(
        grid, land_mask, env,
        ais_density=ais_result.da.values,
        ais_weight=1.5,
    )
    
    # 验证 AIS 成本被正确应用
    assert "ais_density" in cost_with_ais.components, "AIS 密度未在 components 中"
    ais_cost = cost_with_ais.components["ais_density"]
    assert np.max(ais_cost) > 0, "AIS 成本全为 0"
    assert np.max(ais_cost) <= 1.5, "AIS 成本超出预期范围"
    
    # 验证总成本增加
    ocean_mask = ~land_mask
    if np.any(ocean_mask):
        ocean_cost_without = cost_without_ais.cost[ocean_mask]
        ocean_cost_with = cost_with_ais.cost[ocean_mask]
        # 至少某些格点的成本应该增加
        assert np.any(ocean_cost_with > ocean_cost_without), "AIS 未增加任何成本"
    
    print(f"✓ Step 3 通过：AIS 密度成功集成到成本模型")
    
    print("\n✅ AIS Phase 1 完整工作流程验证成功！")


def test_ais_phase1_with_real_data():
    """测试使用真实 AIS 数据的完整流程。"""
    ais_csv_path = get_real_ais_csv_path()
    
    if not Path(ais_csv_path).exists():
        pytest.skip(f"真实 AIS 数据不存在：{ais_csv_path}")
    
    # 加载网格
    grid, land_mask = make_demo_grid(ny=50, nx=50)
    
    # 构建 AIS 密度
    ais_result = build_ais_density_for_grid(
        ais_csv_path,
        grid.lat2d,
        grid.lon2d,
        max_rows=50000,
    )
    
    # 验证密度场有效
    assert ais_result.da.shape == (50, 50)
    assert ais_result.num_binned > 0
    
    # 创建环境
    ny, nx = grid.shape()
    env = RealEnvLayers(
        grid=grid,
        sic=np.ones((ny, nx), dtype=float) * 0.4,
        wave_swh=np.ones((ny, nx), dtype=float) * 2.0,
        ice_thickness_m=None,
        land_mask=land_mask,
    )
    
    # 构建成本（包含 AIS）
    cost_field = build_cost_from_real_env(
        grid, land_mask, env,
        ice_penalty=4.0,
        wave_penalty=2.0,
        ais_density=ais_result.da.values,
        ais_weight=2.0,
    )
    
    # 验证成本场有效
    assert cost_field.cost.shape == (ny, nx)
    assert "ais_density" in cost_field.components
    assert np.any(np.isfinite(cost_field.cost))
    
    print(f"✓ 真实数据测试通过：成功处理 {ais_result.num_points} 个 AIS 点")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])




