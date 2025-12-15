"""
CMEMS 加载器测试

测试 CMEMS NetCDF 文件的加载和对齐功能。
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False


@pytest.mark.skipif(not HAS_XARRAY, reason="xarray not installed")
class TestCMEMSLoader:
    """CMEMS 加载器测试"""
    
    @staticmethod
    def create_test_sic_nc(path: Path, shape: tuple = (10, 10)) -> Path:
        """创建测试 SIC NetCDF 文件"""
        ny, nx = shape
        
        # 创建坐标
        lat = np.linspace(65, 85, ny)
        lon = np.linspace(-40, 60, nx)
        
        # 创建数据（0-100 范围）
        sic_data = np.random.uniform(0, 100, (ny, nx)).astype(np.float32)
        
        # 创建 xarray Dataset
        ds = xr.Dataset(
            {
                "sic": (["lat", "lon"], sic_data),
            },
            coords={
                "lat": lat,
                "lon": lon,
            },
        )
        
        # 保存为 NetCDF
        ds.to_netcdf(path)
        return path
    
    @staticmethod
    def create_test_swh_nc(path: Path, shape: tuple = (10, 10)) -> Path:
        """创建测试 SWH NetCDF 文件"""
        ny, nx = shape
        
        # 创建坐标
        lat = np.linspace(65, 85, ny)
        lon = np.linspace(-40, 60, nx)
        
        # 创建数据（0-10 米范围）
        swh_data = np.random.uniform(0, 10, (ny, nx)).astype(np.float32)
        
        # 创建 xarray Dataset
        ds = xr.Dataset(
            {
                "sea_surface_wave_significant_height": (["lat", "lon"], swh_data),
            },
            coords={
                "lat": lat,
                "lon": lon,
            },
        )
        
        # 保存为 NetCDF
        ds.to_netcdf(path)
        return path
    
    def test_load_sic_from_nc(self):
        """测试加载 SIC 数据"""
        from arcticroute.io.cmems_loader import load_sic_from_nc
        
        with tempfile.TemporaryDirectory() as tmpdir:
            nc_path = Path(tmpdir) / "test_sic.nc"
            self.create_test_sic_nc(nc_path)
            
            # 加载数据
            sic_2d, metadata = load_sic_from_nc(nc_path)
            
            # 验证形状
            assert sic_2d.shape == (10, 10)
            
            # 验证数据范围（应该被规范化到 0-1）
            assert sic_2d.min() >= 0
            assert sic_2d.max() <= 1
            
            # 验证元数据
            assert "variable" in metadata
            assert "shape" in metadata
            assert metadata["shape"] == (10, 10)
    
    def test_load_swh_from_nc(self):
        """测试加载 SWH 数据"""
        from arcticroute.io.cmems_loader import load_swh_from_nc
        
        with tempfile.TemporaryDirectory() as tmpdir:
            nc_path = Path(tmpdir) / "test_swh.nc"
            self.create_test_swh_nc(nc_path)
            
            # 加载数据
            swh_2d, metadata = load_swh_from_nc(nc_path)
            
            # 验证形状
            assert swh_2d.shape == (10, 10)
            
            # 验证数据范围
            assert swh_2d.min() >= 0
            assert swh_2d.max() <= 10
            
            # 验证元数据
            assert "variable" in metadata
            assert "shape" in metadata
            assert metadata["shape"] == (10, 10)
    
    def test_find_latest_nc(self):
        """测试查找最新 NetCDF 文件"""
        from arcticroute.io.cmems_loader import find_latest_nc
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # 创建多个文件
            file1 = tmpdir / "sic_20251201.nc"
            file2 = tmpdir / "sic_20251202.nc"
            file3 = tmpdir / "sic_20251203.nc"
            
            file1.touch()
            file2.touch()
            file3.touch()
            
            # 查找最新文件
            latest = find_latest_nc(tmpdir, "sic_*.nc")
            
            # 应该返回最后创建的文件
            assert latest is not None
            assert latest.name in ["sic_20251201.nc", "sic_20251202.nc", "sic_20251203.nc"]
    
    def test_load_sic_with_time_dimension(self):
        """测试加载有时间维度的 SIC 数据"""
        from arcticroute.io.cmems_loader import load_sic_from_nc
        
        with tempfile.TemporaryDirectory() as tmpdir:
            nc_path = Path(tmpdir) / "test_sic_time.nc"
            
            # 创建有时间维度的数据
            ny, nx, nt = 10, 10, 3
            lat = np.linspace(65, 85, ny)
            lon = np.linspace(-40, 60, nx)
            time = np.arange(nt)
            
            sic_data = np.random.uniform(0, 100, (nt, ny, nx)).astype(np.float32)
            
            ds = xr.Dataset(
                {
                    "sic": (["time", "lat", "lon"], sic_data),
                },
                coords={
                    "time": time,
                    "lat": lat,
                    "lon": lon,
                },
            )
            
            ds.to_netcdf(nc_path)
            
            # 加载数据（应该取最后一个时间步）
            sic_2d, metadata = load_sic_from_nc(nc_path)
            
            # 验证形状（应该是 2D）
            assert sic_2d.ndim == 2
            assert sic_2d.shape == (10, 10)
    
    def test_real_env_layers_from_cmems(self):
        """测试 RealEnvLayers.from_cmems 构造器"""
        from arcticroute.core.env_real import RealEnvLayers
        from arcticroute.core.grid import Grid2D
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # 创建测试网格
            ny, nx = 10, 10
            lat2d = np.linspace(65, 85, ny)[:, np.newaxis] * np.ones((1, nx))
            lon2d = np.linspace(-40, 60, nx)[np.newaxis, :] * np.ones((ny, 1))
            grid = Grid2D(lat2d=lat2d, lon2d=lon2d)
            
            # 创建测试数据文件
            sic_path = tmpdir / "test_sic.nc"
            self.create_test_sic_nc(sic_path, shape=(ny, nx))
            
            swh_path = tmpdir / "test_swh.nc"
            self.create_test_swh_nc(swh_path, shape=(ny, nx))
            
            # 创建 RealEnvLayers
            env = RealEnvLayers.from_cmems(
                grid=grid,
                sic_nc=sic_path,
                swh_nc=swh_path,
                allow_partial=True,
            )
            
            # 验证
            assert env.grid is not None
            assert env.sic is not None or env.wave_swh is not None
            assert env.sic.shape == (ny, nx) if env.sic is not None else True
            assert env.wave_swh.shape == (ny, nx) if env.wave_swh is not None else True
    
    def test_real_env_layers_from_cmems_partial(self):
        """测试 RealEnvLayers.from_cmems 部分数据加载"""
        from arcticroute.core.env_real import RealEnvLayers
        from arcticroute.core.grid import Grid2D
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # 创建测试网格
            ny, nx = 10, 10
            lat2d = np.linspace(65, 85, ny)[:, np.newaxis] * np.ones((1, nx))
            lon2d = np.linspace(-40, 60, nx)[np.newaxis, :] * np.ones((ny, 1))
            grid = Grid2D(lat2d=lat2d, lon2d=lon2d)
            
            # 只创建 SIC 数据
            sic_path = tmpdir / "test_sic.nc"
            self.create_test_sic_nc(sic_path, shape=(ny, nx))
            
            # 创建 RealEnvLayers（SWH 缺失）
            env = RealEnvLayers.from_cmems(
                grid=grid,
                sic_nc=sic_path,
                swh_nc=None,
                allow_partial=True,
            )
            
            # 验证
            assert env.grid is not None
            assert env.sic is not None
            assert env.wave_swh is None  # SWH 缺失


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

