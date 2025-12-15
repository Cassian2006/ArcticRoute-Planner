"""
Step 2 测试：AIS 栅格化为密度场
"""

import numpy as np
import pytest
from pathlib import Path

from arcticroute.core.ais_ingest import (
    rasterize_ais_density_to_grid,
    build_ais_density_for_grid,
)


def get_test_ais_csv_path() -> str:
    """获取测试 AIS CSV 路径。"""
    return str(Path(__file__).parent / "data" / "ais_sample.csv")


def create_toy_grid(ny: int = 10, nx: int = 10) -> tuple:
    """创建一个小的玩具网格。"""
    # 创建简单的网格：纬度 75-76N，经度 20-22E
    lat2d = np.linspace(75.0, 76.0, ny)[:, np.newaxis] + np.zeros((1, nx))
    lon2d = np.linspace(20.0, 22.0, nx)[np.newaxis, :] + np.zeros((ny, 1))
    return lat2d, lon2d


def test_rasterize_ais_density_basic():
    """测试基础栅格化功能。"""
    lat2d, lon2d = create_toy_grid(10, 10)
    
    # 创建几个 AIS 点
    lat_points = np.array([75.5, 75.6, 75.7])
    lon_points = np.array([21.0, 21.1, 21.2])
    
    # 栅格化
    da = rasterize_ais_density_to_grid(lat_points, lon_points, lat2d, lon2d, normalize=False)
    
    # 检查形状
    assert da.shape == (10, 10)
    assert da.name == "ais_density"
    
    # 检查总和（应该等于点数）
    assert np.sum(da.values) == 3.0


def test_rasterize_ais_density_normalize():
    """测试归一化功能。"""
    lat2d, lon2d = create_toy_grid(10, 10)
    
    # 创建几个 AIS 点
    lat_points = np.array([75.5, 75.5, 75.5])  # 三个点在同一位置
    lon_points = np.array([21.0, 21.0, 21.0])
    
    # 栅格化（归一化）
    da = rasterize_ais_density_to_grid(lat_points, lon_points, lat2d, lon2d, normalize=True)
    
    # 检查最大值是 1.0
    assert np.max(da.values) <= 1.0
    
    # 检查最大值确实是 1.0（因为有 3 个点在同一位置）
    assert np.max(da.values) == 1.0


def test_rasterize_ais_density_no_crash_on_outliers():
    """测试处理越界坐标时不会崩溃。"""
    lat2d, lon2d = create_toy_grid(10, 10)
    
    # 创建包含越界点的 AIS 点
    lat_points = np.array([75.5, 200.0, 75.7])  # 200.0 是越界的
    lon_points = np.array([21.0, 21.1, 21.2])
    
    # 应该不会崩溃，而是找到最近的栅格
    da = rasterize_ais_density_to_grid(lat_points, lon_points, lat2d, lon2d, normalize=False)
    
    # 检查形状
    assert da.shape == (10, 10)
    
    # 检查总和（应该等于点数，即使有越界点）
    assert np.sum(da.values) == 3.0


def test_build_ais_density_for_grid_basic():
    """测试从 CSV 构建密度场。"""
    csv_path = get_test_ais_csv_path()
    lat2d, lon2d = create_toy_grid(10, 10)
    
    result = build_ais_density_for_grid(csv_path, lat2d, lon2d, max_rows=100)
    
    # 检查结果
    assert result.da.shape == (10, 10)
    assert result.num_points == 9  # 测试文件有 9 行
    assert result.num_binned == 9  # 全部有效
    assert result.frac_binned == 1.0


def test_build_ais_density_for_grid_nonexistent():
    """测试处理不存在的 CSV 文件。"""
    lat2d, lon2d = create_toy_grid(10, 10)
    
    result = build_ais_density_for_grid("/nonexistent/path.csv", lat2d, lon2d)
    
    # 应该返回空密度场
    assert result.da.shape == (10, 10)
    assert result.num_points == 0
    assert result.num_binned == 0
    assert result.frac_binned == 0.0


def test_build_ais_density_max_rows():
    """测试 max_rows 参数的效果。"""
    csv_path = get_test_ais_csv_path()
    lat2d, lon2d = create_toy_grid(10, 10)
    
    # 读取全部
    result_all = build_ais_density_for_grid(csv_path, lat2d, lon2d, max_rows=1000)
    
    # 读取前 5 行
    result_5 = build_ais_density_for_grid(csv_path, lat2d, lon2d, max_rows=5)
    
    # 两者都应该成功
    assert result_all.num_points == 9
    assert result_5.num_points == 5


def test_rasterize_ais_density_empty_points():
    """测试处理空点集。"""
    lat2d, lon2d = create_toy_grid(10, 10)
    
    # 空点集
    lat_points = np.array([])
    lon_points = np.array([])
    
    # 应该返回全零密度场
    da = rasterize_ais_density_to_grid(lat_points, lon_points, lat2d, lon2d, normalize=True)
    
    assert da.shape == (10, 10)
    assert np.sum(da.values) == 0.0


def test_rasterize_ais_density_single_point():
    """测试单个点的栅格化。"""
    lat2d, lon2d = create_toy_grid(10, 10)
    
    # 单个点
    lat_points = np.array([75.5])
    lon_points = np.array([21.0])
    
    # 栅格化
    da = rasterize_ais_density_to_grid(lat_points, lon_points, lat2d, lon2d, normalize=False)
    
    # 检查总和
    assert np.sum(da.values) == 1.0
    
    # 检查最大值
    assert np.max(da.values) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])










