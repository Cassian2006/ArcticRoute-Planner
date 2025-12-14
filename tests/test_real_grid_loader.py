"""
真实网格和 landmask 加载器的单元测试。

使用临时 NetCDF 文件进行测试，不依赖真实的 ArcticRoute_data_backup。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

# 尝试导入 xarray，如果不可用则跳过相关测试
try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False


@pytest.mark.skipif(not HAS_XARRAY, reason="xarray not installed")
class TestLoadRealGridFromNC:
    """测试 load_real_grid_from_nc 函数。"""

    def test_load_real_grid_from_nc_1d_coords(self, tmp_path: Path) -> None:
        """
        测试从 1D 坐标的 NetCDF 文件加载网格。

        创建一个简单的 10x20 网格，lat 和 lon 为 1D 坐标。
        """
        from arcticroute.core.grid import load_real_grid_from_nc

        # 创建临时 NetCDF 文件
        nc_path = tmp_path / "grid_1d.nc"
        ny, nx = 10, 20
        lat_1d = np.linspace(65.0, 80.0, ny)
        lon_1d = np.linspace(0.0, 160.0, nx)

        ds = xr.Dataset(
            coords={
                "lat": ("y", lat_1d),
                "lon": ("x", lon_1d),
            }
        )
        ds.to_netcdf(nc_path)

        # 加载网格
        grid = load_real_grid_from_nc(nc_path=nc_path)

        # 验证
        assert grid is not None
        assert grid.shape() == (ny, nx)
        assert grid.lat2d.shape == (ny, nx)
        assert grid.lon2d.shape == (ny, nx)

    def test_load_real_grid_from_nc_2d_coords(self, tmp_path: Path) -> None:
        """
        测试从 2D 坐标的 NetCDF 文件加载网格。

        创建一个简单的 10x20 网格，lat 和 lon 为 2D 坐标。
        """
        from arcticroute.core.grid import load_real_grid_from_nc

        # 创建临时 NetCDF 文件
        nc_path = tmp_path / "grid_2d.nc"
        ny, nx = 10, 20
        lat_1d = np.linspace(65.0, 80.0, ny)
        lon_1d = np.linspace(0.0, 160.0, nx)
        lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)

        ds = xr.Dataset(
            data_vars={
                "lat": (["y", "x"], lat2d),
                "lon": (["y", "x"], lon2d),
            }
        )
        ds.to_netcdf(nc_path)

        # 加载网格
        grid = load_real_grid_from_nc(nc_path=nc_path)

        # 验证
        assert grid is not None
        assert grid.shape() == (ny, nx)
        assert grid.lat2d.shape == (ny, nx)
        assert grid.lon2d.shape == (ny, nx)

    def test_load_real_grid_missing_file_returns_none(self) -> None:
        """测试加载不存在的文件时返回 None。"""
        from arcticroute.core.grid import load_real_grid_from_nc

        grid = load_real_grid_from_nc(nc_path=Path("/nonexistent/path/grid.nc"))
        assert grid is None

    def test_load_real_grid_missing_lat_lon_returns_none(
        self, tmp_path: Path
    ) -> None:
        """测试加载缺少 lat/lon 变量的文件时返回 None。"""
        from arcticroute.core.grid import load_real_grid_from_nc

        # 创建没有 lat/lon 的 NetCDF 文件
        nc_path = tmp_path / "no_coords.nc"
        ds = xr.Dataset(
            data_vars={
                "temperature": (["y", "x"], np.random.rand(10, 20)),
            }
        )
        ds.to_netcdf(nc_path)

        grid = load_real_grid_from_nc(nc_path=nc_path)
        assert grid is None


@pytest.mark.skipif(not HAS_XARRAY, reason="xarray not installed")
class TestLoadRealLandmaskFromNC:
    """测试 load_real_landmask_from_nc 函数。"""

    def test_load_real_landmask_from_nc_basic(self, tmp_path: Path) -> None:
        """
        测试从 NetCDF 文件加载 landmask。

        创建一个简单的 10x20 网格和对应的 landmask。
        """
        from arcticroute.core.grid import Grid2D, load_real_grid_from_nc
        from arcticroute.core.landmask import load_real_landmask_from_nc

        # 创建临时 NetCDF 文件
        nc_path = tmp_path / "grid_with_mask.nc"
        ny, nx = 10, 20
        lat_1d = np.linspace(65.0, 80.0, ny)
        lon_1d = np.linspace(0.0, 160.0, nx)
        land_mask_data = np.zeros((ny, nx), dtype=bool)
        land_mask_data[:, -5:] = True  # 右侧 5 列为陆地

        ds = xr.Dataset(
            coords={
                "lat": ("y", lat_1d),
                "lon": ("x", lon_1d),
            },
            data_vars={
                "land_mask": (["y", "x"], land_mask_data),
            },
        )
        ds.to_netcdf(nc_path)

        # 加载网格
        grid = load_real_grid_from_nc(nc_path=nc_path)
        assert grid is not None

        # 加载 landmask
        land_mask = load_real_landmask_from_nc(grid, nc_path=nc_path)

        # 验证
        assert land_mask is not None
        assert land_mask.shape == (ny, nx)
        assert land_mask.dtype == bool
        assert land_mask[:, -5:].all()  # 右侧 5 列应该全是 True
        assert not land_mask[:, :-5].any()  # 左侧应该全是 False

    def test_load_real_landmask_missing_file_returns_none(self) -> None:
        """测试加载不存在的文件时返回 None。"""
        from arcticroute.core.grid import Grid2D
        from arcticroute.core.landmask import load_real_landmask_from_nc

        # 创建一个虚拟网格
        lat2d = np.linspace(65.0, 80.0, 10).reshape(10, 1)
        lon2d = np.linspace(0.0, 160.0, 20).reshape(1, 20)
        lat2d = np.broadcast_to(lat2d, (10, 20))
        lon2d = np.broadcast_to(lon2d, (10, 20))
        grid = Grid2D(lat2d=lat2d, lon2d=lon2d)

        land_mask = load_real_landmask_from_nc(
            grid, nc_path=Path("/nonexistent/path/mask.nc")
        )
        assert land_mask is None

    def test_load_real_landmask_missing_var_returns_none(
        self, tmp_path: Path
    ) -> None:
        """测试加载缺少 landmask 变量的文件时返回 None。"""
        from arcticroute.core.grid import Grid2D, load_real_grid_from_nc
        from arcticroute.core.landmask import load_real_landmask_from_nc

        # 创建没有 land_mask 变量的 NetCDF 文件
        nc_path = tmp_path / "no_mask.nc"
        ny, nx = 10, 20
        lat_1d = np.linspace(65.0, 80.0, ny)
        lon_1d = np.linspace(0.0, 160.0, nx)

        ds = xr.Dataset(
            coords={
                "lat": ("y", lat_1d),
                "lon": ("x", lon_1d),
            },
            data_vars={
                "temperature": (["y", "x"], np.random.rand(ny, nx)),
            },
        )
        ds.to_netcdf(nc_path)

        # 加载网格
        grid = load_real_grid_from_nc(nc_path=nc_path)
        assert grid is not None

        # 尝试加载 landmask（应该失败）
        land_mask = load_real_landmask_from_nc(grid, nc_path=nc_path)
        assert land_mask is None

    def test_load_real_landmask_shape_mismatch_resamples(
        self, tmp_path: Path
    ) -> None:
        """
        测试当 landmask 形状与网格不匹配时进行重采样。

        创建一个 10x20 的网格，但 landmask 是 5x10，应该被重采样。
        """
        from arcticroute.core.grid import Grid2D
        from arcticroute.core.landmask import load_real_landmask_from_nc

        # 创建临时 NetCDF 文件
        nc_path = tmp_path / "shape_mismatch.nc"
        ny, nx = 10, 20
        lat_1d = np.linspace(65.0, 80.0, ny)
        lon_1d = np.linspace(0.0, 160.0, nx)

        # 创建一个较小的 landmask
        small_ny, small_nx = 5, 10
        small_land_mask = np.zeros((small_ny, small_nx), dtype=bool)
        small_land_mask[:, -3:] = True

        ds = xr.Dataset(
            coords={
                "lat": ("y", lat_1d),
                "lon": ("x", lon_1d),
            },
            data_vars={
                "land_mask": (["y_small", "x_small"], small_land_mask),
            },
        )
        ds.to_netcdf(nc_path)

        # 创建目标网格
        lat2d = np.linspace(65.0, 80.0, ny).reshape(ny, 1)
        lon2d = np.linspace(0.0, 160.0, nx).reshape(1, nx)
        lat2d = np.broadcast_to(lat2d, (ny, nx))
        lon2d = np.broadcast_to(lon2d, (ny, nx))
        grid = Grid2D(lat2d=lat2d, lon2d=lon2d)

        # 加载并重采样 landmask
        land_mask = load_real_landmask_from_nc(grid, nc_path=nc_path)

        # 验证
        assert land_mask is not None
        assert land_mask.shape == (ny, nx)
        assert land_mask.dtype == bool


@pytest.mark.skipif(not HAS_XARRAY, reason="xarray not installed")
class TestCheckGridAndLandmaskCLI:
    """测试 check_grid_and_landmask 脚本。"""

    def test_check_grid_and_landmask_cli_demo_fallback(self) -> None:
        """
        测试 CLI 脚本在没有真实数据时的 demo fallback 行为。

        使用 subprocess 运行脚本，检查返回码和输出。
        """
        import subprocess

        result = subprocess.run(
            ["python", "-m", "scripts.check_grid_and_landmask"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # 检查返回码
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # 检查输出中是否包含关键字
        output = result.stdout + result.stderr
        assert "source:" in output or "source" in output.lower()
        assert "shape:" in output or "shape" in output.lower()


class TestConfigPaths:
    """测试 config_paths 模块。"""

    def test_get_data_root_returns_path(self) -> None:
        """测试 get_data_root 返回一个 Path 对象。"""
        from arcticroute.core.config_paths import get_data_root

        root = get_data_root()
        assert isinstance(root, Path)
        assert root.is_absolute()

    def test_get_newenv_path_returns_path(self) -> None:
        """测试 get_newenv_path 返回一个 Path 对象。"""
        from arcticroute.core.config_paths import get_newenv_path

        newenv = get_newenv_path()
        assert isinstance(newenv, Path)
        assert newenv.is_absolute()
        assert newenv.name == "newenv"

    def test_get_newenv_path_is_subdir_of_data_root(self) -> None:
        """测试 get_newenv_path 是 get_data_root 的子目录。"""
        from arcticroute.core.config_paths import get_data_root, get_newenv_path

        root = get_data_root()
        newenv = get_newenv_path()

        # 检查 newenv 是否在 root 下
        assert str(newenv).startswith(str(root))













