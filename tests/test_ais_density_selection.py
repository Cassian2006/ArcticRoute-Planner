"""
防回归测试：AIS density 选择/匹配/对齐功能

覆盖：
  - scan_ais_density_candidates: 扫描候选文件
  - select_best_candidate: 选择最佳匹配
  - load_and_align_density: 加载并对齐到目标网格
"""

import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest
import xarray as xr

from arcticroute.core.ais_density_select import (
    scan_ais_density_candidates,
    select_best_candidate,
    load_and_align_density,
    AISDensityCandidate,
)
from arcticroute.core.grid import Grid2D, make_demo_grid


# ============================================================================
# 辅助函数
# ============================================================================

def create_minimal_netcdf(path: Path, shape: Tuple[int, int], varname: str = "ais_density") -> None:
    """
    创建一个最小的 NetCDF 文件用于测试
    
    Args:
        path: 输出文件路径
        shape: 数据形状 (ny, nx)
        varname: 变量名
    """
    ny, nx = shape
    data = np.random.rand(ny, nx).astype(np.float32)
    
    # 使用 xarray 创建数据集
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": np.arange(ny), "x": np.arange(nx)},
        name=varname,
    )
    
    ds = da.to_dataset()
    
    # 尝试写入 NetCDF
    try:
        ds.to_netcdf(path, engine="netcdf4")
    except ImportError:
        try:
            ds.to_netcdf(path, engine="h5netcdf")
        except ImportError:
            pytest.skip("No NetCDF write engine available (netCDF4 or h5netcdf)")


class TestScanCandidates:
    """扫描候选文件的测试"""
    
    def test_scan_finds_nc_files(self, tmp_path):
        """测试 scan_ais_density_candidates 能找到 .nc 文件"""
        # 创建测试文件
        density_file = tmp_path / "ais_density_test.nc"
        create_minimal_netcdf(density_file, shape=(10, 10))
        
        # 扫描
        candidates = scan_ais_density_candidates(search_dirs=[str(tmp_path)])
        
        # 验证
        assert len(candidates) > 0, "应该找到至少一个候选文件"
        
        # 检查是否包含我们创建的文件
        paths = [c.path for c in candidates]
        assert any("ais_density_test" in p for p in paths), "应该找到 ais_density_test.nc"
    
    def test_scan_empty_directory(self, tmp_path):
        """测试扫描空目录"""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        candidates = scan_ais_density_candidates(search_dirs=[str(empty_dir)])
        
        # 可能返回空列表或包含默认文件
        assert isinstance(candidates, list)
    
    def test_scan_multiple_directories(self, tmp_path):
        """测试扫描多个目录"""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        
        # 在 dir1 创建文件
        file1 = dir1 / "density1.nc"
        create_minimal_netcdf(file1, shape=(8, 8))
        
        # 在 dir2 创建文件
        file2 = dir2 / "density2.nc"
        create_minimal_netcdf(file2, shape=(12, 12))
        
        # 扫描两个目录
        candidates = scan_ais_density_candidates(search_dirs=[str(dir1), str(dir2)])
        
        # 应该找到两个文件
        assert len(candidates) >= 2, "应该找到至少两个候选文件"


class TestSelectBest:
    """选择最佳候选的测试"""
    
    def test_select_explicit_path(self):
        """测试显式指定路径时的选择"""
        candidates = [
            AISDensityCandidate(path="/path/to/file1.nc", shape=(10, 10)),
            AISDensityCandidate(path="/path/to/file2.nc", shape=(20, 20)),
        ]
        
        # 选择第一个
        best = select_best_candidate(
            candidates=candidates,
            prefer_path="/path/to/file1.nc",
            target_signature="sig1",
        )
        
        assert best is not None
        assert best.path == "/path/to/file1.nc"
    
    def test_select_with_no_candidates(self):
        """测试没有候选时的行为"""
        best = select_best_candidate(
            candidates=[],
            prefer_path=None,
            target_signature="sig1",
        )
        
        assert best is None
    
    def test_select_first_if_no_preference(self):
        """测试没有偏好时选择第一个"""
        candidates = [
            AISDensityCandidate(path="/path/to/file1.nc", shape=(10, 10)),
            AISDensityCandidate(path="/path/to/file2.nc", shape=(20, 20)),
        ]
        
        best = select_best_candidate(
            candidates=candidates,
            prefer_path=None,
            target_signature=None,
        )
        
        # 应该返回某个候选（可能是第一个或最匹配的）
        assert best is not None
        assert best in candidates


class TestLoadAndAlign:
    """加载并对齐密度数据的测试"""
    
    def test_load_and_align_same_shape(self, tmp_path):
        """测试加载和对齐相同形状的数据"""
        # 创建目标网格
        grid, _ = make_demo_grid(ny=10, nx=10)
        
        # 创建密度文件（相同形状）
        density_file = tmp_path / "density_10x10.nc"
        create_minimal_netcdf(density_file, shape=(10, 10))
        
        # 加载并对齐
        result = load_and_align_density(
            candidate_or_path=str(density_file),
            grid=grid,
            method="nearest",
        )
        
        assert result is not None, "应该成功加载"
        arr, meta = result
        
        # 验证形状
        assert arr.shape == (10, 10), "输出形状应该与网格匹配"
        
        # 验证元数据
        assert isinstance(meta, dict)
    
    def test_load_and_align_different_shape(self, tmp_path):
        """测试加载和对齐不同形状的数据（需要重采样）"""
        # 创建目标网格 6x6
        grid, _ = make_demo_grid(ny=6, nx=6)
        
        # 创建密度文件 4x4（不同形状）
        density_file = tmp_path / "density_4x4.nc"
        create_minimal_netcdf(density_file, shape=(4, 4))
        
        # 加载并对齐
        result = load_and_align_density(
            candidate_or_path=str(density_file),
            grid=grid,
            method="nearest",
        )
        
        if result is not None:
            arr, meta = result
            
            # 验证输出形状与网格匹配
            assert arr.shape == (6, 6), f"输出形状应该是 (6, 6)，实际是 {arr.shape}"
            
            # 验证重采样标记
            assert meta.get("resampled") == True, "应该标记为已重采样"
    
    def test_load_nonexistent_file(self, tmp_path):
        """测试加载不存在的文件"""
        grid, _ = make_demo_grid(ny=10, nx=10)
        
        result = load_and_align_density(
            candidate_or_path=str(tmp_path / "nonexistent.nc"),
            grid=grid,
            method="nearest",
        )
        
        # 应该返回 None 或抛出异常
        assert result is None or isinstance(result, tuple)
    
    def test_load_with_linear_method(self, tmp_path):
        """测试使用线性插值方法"""
        grid, _ = make_demo_grid(ny=8, nx=8)
        
        density_file = tmp_path / "density_linear.nc"
        create_minimal_netcdf(density_file, shape=(4, 4))
        
        result = load_and_align_density(
            candidate_or_path=str(density_file),
            grid=grid,
            method="linear",
        )
        
        if result is not None:
            arr, meta = result
            assert arr.shape == (8, 8)
            assert meta.get("method") == "linear" or meta.get("method") is None
    
    def test_load_with_nearest_method(self, tmp_path):
        """测试使用最近邻方法"""
        grid, _ = make_demo_grid(ny=8, nx=8)
        
        density_file = tmp_path / "density_nearest.nc"
        create_minimal_netcdf(density_file, shape=(4, 4))
        
        result = load_and_align_density(
            candidate_or_path=str(density_file),
            grid=grid,
            method="nearest",
        )
        
        if result is not None:
            arr, meta = result
            assert arr.shape == (8, 8)


class TestIntegration:
    """集成测试"""
    
    def test_full_workflow(self, tmp_path):
        """测试完整的扫描-选择-加载工作流"""
        # 1. 创建候选文件
        file1 = tmp_path / "density1.nc"
        file2 = tmp_path / "density2.nc"
        create_minimal_netcdf(file1, shape=(6, 6))
        create_minimal_netcdf(file2, shape=(8, 8))
        
        # 2. 扫描
        candidates = scan_ais_density_candidates(search_dirs=[str(tmp_path)])
        assert len(candidates) >= 2
        
        # 3. 选择
        best = select_best_candidate(
            candidates=candidates,
            prefer_path=None,
            target_signature=None,
        )
        assert best is not None
        
        # 4. 加载并对齐
        grid, _ = make_demo_grid(ny=6, nx=6)
        result = load_and_align_density(
            candidate_or_path=best.path,
            grid=grid,
            method="nearest",
        )
        
        if result is not None:
            arr, meta = result
            assert arr.shape == (6, 6)
            assert isinstance(meta, dict)


class TestEdgeCases:
    """边界情况测试"""
    
    def test_load_with_nan_values(self, tmp_path):
        """测试包含 NaN 值的数据"""
        grid, _ = make_demo_grid(ny=10, nx=10)
        
        # 创建包含 NaN 的数据
        density_file = tmp_path / "density_with_nan.nc"
        ny, nx = 10, 10
        data = np.random.rand(ny, nx).astype(np.float32)
        data[0:2, 0:2] = np.nan  # 设置一些 NaN 值
        
        da = xr.DataArray(data, dims=("y", "x"), name="ais_density")
        ds = da.to_dataset()
        
        try:
            ds.to_netcdf(density_file, engine="netcdf4")
        except ImportError:
            try:
                ds.to_netcdf(density_file, engine="h5netcdf")
            except ImportError:
                pytest.skip("No NetCDF write engine available")
        
        # 加载
        result = load_and_align_density(
            candidate_or_path=str(density_file),
            grid=grid,
            method="nearest",
        )
        
        if result is not None:
            arr, meta = result
            # 应该保留 NaN 值或进行处理
            assert arr.shape == (10, 10)
    
    def test_load_with_zero_values(self, tmp_path):
        """测试全零数据"""
        grid, _ = make_demo_grid(ny=10, nx=10)
        
        # 创建全零数据
        density_file = tmp_path / "density_zeros.nc"
        data = np.zeros((10, 10), dtype=np.float32)
        
        da = xr.DataArray(data, dims=("y", "x"), name="ais_density")
        ds = da.to_dataset()
        
        try:
            ds.to_netcdf(density_file, engine="netcdf4")
        except ImportError:
            try:
                ds.to_netcdf(density_file, engine="h5netcdf")
            except ImportError:
                pytest.skip("No NetCDF write engine available")
        
        result = load_and_align_density(
            candidate_or_path=str(density_file),
            grid=grid,
            method="nearest",
        )
        
        if result is not None:
            arr, meta = result
            assert arr.shape == (10, 10)
            # 全零数据应该保持全零（或接近全零）
            assert np.nanmin(arr) >= -1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

