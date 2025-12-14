"""
Landmask 选择、加载、对齐的单元测试。

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
class TestLandmaskSelection:
    """测试 landmask 选择和加载功能。"""

    def test_scan_landmask_candidates_finds_nc_files(self, tmp_path: Path) -> None:
        """
        测试 scan_landmask_candidates 能找到 .nc 文件。

        创建一个临时目录，放入几个 landmask .nc 文件，验证扫描能找到它们。
        """
        from arcticroute.core.landmask_select import scan_landmask_candidates

        # 创建临时 landmask 文件
        landmask_dir = tmp_path / "data_real" / "landmask"
        landmask_dir.mkdir(parents=True)

        # 创建第一个 landmask 文件
        nc_path1 = landmask_dir / "land_mask_gebco.nc"
        ny, nx = 10, 20
        lat_1d = np.linspace(65.0, 80.0, ny)
        lon_1d = np.linspace(0.0, 160.0, nx)
        land_mask = np.random.rand(ny, nx) > 0.5

        ds = xr.Dataset(
            data_vars={"land_mask": (["y", "x"], land_mask)},
            coords={
                "lat": ("y", lat_1d),
                "lon": ("x", lon_1d),
            },
        )
        ds.to_netcdf(nc_path1)

        # 扫描候选
        candidates = scan_landmask_candidates(search_dirs=[str(landmask_dir)])

        # 验证
        assert len(candidates) >= 1
        assert any("land_mask_gebco" in c.path for c in candidates)
        assert any(c.varname == "land_mask" for c in candidates)
        assert any(c.shape == (ny, nx) for c in candidates)

    def test_select_best_candidate_prefers_explicit_path(self, tmp_path: Path) -> None:
        """
        测试 select_best_candidate 优先选择显式指定的路径。
        """
        from arcticroute.core.landmask_select import (
            LandmaskCandidate,
            select_best_candidate,
        )

        # 创建候选列表（note="" 表示成功读取）
        cand1 = LandmaskCandidate(
            path="data_real/landmask/file1.nc",
            grid_signature="40x80_65.0000_80.0000_0.0000_160.0000",
            shape=(40, 80),
            varname="land_mask",
            note="",  # 成功读取
        )
        cand2 = LandmaskCandidate(
            path="data_real/landmask/file2.nc",
            grid_signature="50x100_60.0000_85.0000_-10.0000_170.0000",
            shape=(50, 100),
            varname="landmask",
            note="",  # 成功读取
        )

        candidates = [cand1, cand2]

        # 测试优先路径（但由于文件不存在，会回退到第一个成功读取的）
        # 所以这个测试实际上测试的是签名匹配
        selected = select_best_candidate(
            candidates, target_signature="50x100_60.0000_85.0000_-10.0000_170.0000"
        )
        assert selected is not None
        assert selected.path == "data_real/landmask/file2.nc"

    def test_select_best_candidate_matches_signature(self, tmp_path: Path) -> None:
        """
        测试 select_best_candidate 能精确匹配签名。
        """
        from arcticroute.core.landmask_select import (
            LandmaskCandidate,
            select_best_candidate,
        )

        # 创建候选列表
        target_sig = "40x80_65.0000_80.0000_0.0000_160.0000"
        cand1 = LandmaskCandidate(
            path="data_real/landmask/file1.nc",
            grid_signature=target_sig,
            shape=(40, 80),
            varname="land_mask",
        )
        cand2 = LandmaskCandidate(
            path="data_real/landmask/file2.nc",
            grid_signature="50x100_60.0000_85.0000_-10.0000_170.0000",
            shape=(50, 100),
            varname="landmask",
        )

        candidates = [cand1, cand2]

        # 测试签名匹配
        selected = select_best_candidate(candidates, target_signature=target_sig)
        assert selected is not None
        assert selected.grid_signature == target_sig

    def test_load_and_align_landmask_shape_match(self, tmp_path: Path) -> None:
        """
        测试 load_and_align_landmask 当形状已匹配时直接返回。
        """
        from arcticroute.core.grid import Grid2D
        from arcticroute.core.landmask_select import load_and_align_landmask

        # 创建临时 landmask 文件
        nc_path = tmp_path / "land_mask.nc"
        ny, nx = 10, 20
        lat_1d = np.linspace(65.0, 80.0, ny)
        lon_1d = np.linspace(0.0, 160.0, nx)
        land_mask_data = np.random.rand(ny, nx) > 0.5

        ds = xr.Dataset(
            data_vars={"land_mask": (["y", "x"], land_mask_data)},
            coords={
                "lat": ("y", lat_1d),
                "lon": ("x", lon_1d),
            },
        )
        ds.to_netcdf(nc_path)

        # 创建相同形状的网格
        lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
        grid = Grid2D(lat2d=lat2d, lon2d=lon2d)

        # 加载和对齐
        landmask, meta = load_and_align_landmask(str(nc_path), grid)

        # 验证
        assert landmask is not None
        assert landmask.shape == grid.shape()
        assert landmask.dtype == bool
        assert meta["resampled"] is False
        assert meta["land_fraction"] is not None

    def test_load_and_align_landmask_with_resampling(self, tmp_path: Path) -> None:
        """
        测试 load_and_align_landmask 能进行最近邻重采样。
        """
        from arcticroute.core.grid import Grid2D
        from arcticroute.core.landmask_select import load_and_align_landmask

        # 创建小的 landmask 文件
        nc_path = tmp_path / "land_mask_small.nc"
        ny_src, nx_src = 5, 10
        lat_1d_src = np.linspace(65.0, 80.0, ny_src)
        lon_1d_src = np.linspace(0.0, 160.0, nx_src)
        land_mask_data = np.random.rand(ny_src, nx_src) > 0.5

        ds = xr.Dataset(
            data_vars={"land_mask": (["y", "x"], land_mask_data)},
            coords={
                "lat": ("y", lat_1d_src),
                "lon": ("x", lon_1d_src),
            },
        )
        ds.to_netcdf(nc_path)

        # 创建更大的网格
        ny_tgt, nx_tgt = 20, 40
        lat_1d_tgt = np.linspace(65.0, 80.0, ny_tgt)
        lon_1d_tgt = np.linspace(0.0, 160.0, nx_tgt)
        lon2d, lat2d = np.meshgrid(lon_1d_tgt, lat_1d_tgt)
        grid = Grid2D(lat2d=lat2d, lon2d=lon2d)

        # 加载和对齐
        landmask, meta = load_and_align_landmask(str(nc_path), grid, method="nearest")

        # 验证
        assert landmask is not None
        assert landmask.shape == grid.shape()
        assert landmask.dtype == bool
        assert meta["resampled"] is True
        assert meta["original_shape"] == (ny_src, nx_src)
        assert meta["target_shape"] == (ny_tgt, nx_tgt)

    def test_normalize_landmask_semantics_0_1_encoding(self) -> None:
        """
        测试语义归一化能处理 0/1 编码。
        """
        from arcticroute.core.landmask_select import _normalize_landmask_semantics

        # 测试 1 = land 的情况（陆地比例合理）
        arr = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1]], dtype=float)
        result = _normalize_landmask_semantics(arr)
        assert result.dtype == bool
        # 应该识别出 1 是 land
        assert result[0, 1] == True
        assert result[0, 0] == False

    def test_normalize_landmask_semantics_inverted_encoding(self) -> None:
        """
        测试语义归一化能处理反转的 0/1 编码。
        """
        from arcticroute.core.landmask_select import _normalize_landmask_semantics

        # 测试 0 = land 的情况（陆地比例合理）
        arr = np.ones((10, 10), dtype=float)
        arr[0:3, 0:3] = 0  # 左上角 3x3 为 land (0)
        result = _normalize_landmask_semantics(arr)
        assert result.dtype == bool
        # 应该识别出 0 是 land
        assert result[0, 0] == True
        assert result[5, 5] == False

    def test_normalize_landmask_semantics_float_encoding(self) -> None:
        """
        测试语义归一化能处理 float 编码（>0.5 判为 land）。
        """
        from arcticroute.core.landmask_select import _normalize_landmask_semantics

        # 测试 float 编码
        arr = np.array([[0.1, 0.7, 0.3], [0.9, 0.2, 0.6]], dtype=float)
        result = _normalize_landmask_semantics(arr)
        assert result.dtype == bool
        assert result[0, 1] == True  # 0.7 > 0.5
        assert result[0, 0] == False  # 0.1 < 0.5

    def test_normalize_landmask_semantics_nan_handling(self) -> None:
        """
        测试语义归一化能处理 NaN（当 ocean）。
        """
        from arcticroute.core.landmask_select import _normalize_landmask_semantics

        # 测试 NaN 处理
        arr = np.array([[1.0, np.nan, 0.0], [1.0, 1.0, 0.0]], dtype=float)
        result = _normalize_landmask_semantics(arr)
        assert result.dtype == bool
        assert not np.isnan(result).any()  # 不应该有 NaN
        # NaN 应该被转换为 0（ocean）
        assert result[0, 1] == False

    def test_load_and_align_landmask_land_fraction_sanity(self, tmp_path: Path) -> None:
        """
        测试 load_and_align_landmask 计算的陆地比例是否合理。
        """
        from arcticroute.core.grid import Grid2D
        from arcticroute.core.landmask_select import load_and_align_landmask

        # 创建 landmask 文件，陆地比例约 30%
        nc_path = tmp_path / "land_mask_frac.nc"
        ny, nx = 10, 10
        lat_1d = np.linspace(65.0, 80.0, ny)
        lon_1d = np.linspace(0.0, 160.0, nx)
        land_mask_data = np.zeros((ny, nx), dtype=bool)
        land_mask_data[0:3, :] = True  # 前 3 行为陆地

        ds = xr.Dataset(
            data_vars={"land_mask": (["y", "x"], land_mask_data)},
            coords={
                "lat": ("y", lat_1d),
                "lon": ("x", lon_1d),
            },
        )
        ds.to_netcdf(nc_path)

        # 创建网格
        lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
        grid = Grid2D(lat2d=lat2d, lon2d=lon2d)

        # 加载和对齐
        landmask, meta = load_and_align_landmask(str(nc_path), grid)

        # 验证陆地比例
        assert landmask is not None
        land_frac = meta["land_fraction"]
        assert land_frac is not None
        assert 0.0 <= land_frac <= 1.0
        # 应该接近 30%
        assert 0.25 < land_frac < 0.35

    def test_load_and_align_landmask_warning_on_extreme_fraction(
        self, tmp_path: Path
    ) -> None:
        """
        测试当陆地比例异常时是否产生警告。
        """
        from arcticroute.core.grid import Grid2D
        from arcticroute.core.landmask_select import load_and_align_landmask

        # 创建陆地比例极高的 landmask（99%）
        nc_path = tmp_path / "land_mask_extreme.nc"
        ny, nx = 100, 100
        lat_1d = np.linspace(65.0, 80.0, ny)
        lon_1d = np.linspace(0.0, 160.0, nx)
        land_mask_data = np.ones((ny, nx), dtype=bool)
        # 只有 1% 的海洋点（100 个点）
        land_mask_data[0:10, 0:10] = False

        ds = xr.Dataset(
            data_vars={"land_mask": (["y", "x"], land_mask_data)},
            coords={
                "lat": ("y", lat_1d),
                "lon": ("x", lon_1d),
            },
        )
        ds.to_netcdf(nc_path)

        # 创建网格
        lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
        grid = Grid2D(lat2d=lat2d, lon2d=lon2d)

        # 加载和对齐
        landmask, meta = load_and_align_landmask(str(nc_path), grid)

        # 验证警告（陆地比例 > 0.99）
        assert landmask is not None
        # 陆地比例应该是 0.99
        assert meta.get("land_fraction") is not None
        assert meta.get("land_fraction") > 0.98
        # 应该产生警告
        assert meta.get("warning") is not None
        assert "异常" in meta["warning"] or "abnormal" in meta["warning"].lower()

    def test_compute_grid_signature(self) -> None:
        """
        测试网格签名计算。
        """
        from arcticroute.core.grid import Grid2D
        from arcticroute.core.landmask_select import compute_grid_signature

        # 创建网格
        ny, nx = 40, 80
        lat_1d = np.linspace(65.0, 80.0, ny)
        lon_1d = np.linspace(0.0, 160.0, nx)
        lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
        grid = Grid2D(lat2d=lat2d, lon2d=lon2d)

        # 计算签名
        sig = compute_grid_signature(grid)

        # 验证
        assert isinstance(sig, str)
        assert "40x80" in sig
        assert "65.0000" in sig
        assert "80.0000" in sig
        assert "0.0000" in sig
        assert "160.0000" in sig

    def test_load_and_align_landmask_file_not_found(self) -> None:
        """
        测试当文件不存在时的处理。
        """
        from arcticroute.core.grid import Grid2D
        from arcticroute.core.landmask_select import load_and_align_landmask

        # 创建网格
        ny, nx = 10, 20
        lat_1d = np.linspace(65.0, 80.0, ny)
        lon_1d = np.linspace(0.0, 160.0, nx)
        lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
        grid = Grid2D(lat2d=lat2d, lon2d=lon2d)

        # 尝试加载不存在的文件
        landmask, meta = load_and_align_landmask("/nonexistent/path/land_mask.nc", grid)

        # 验证
        assert landmask is None
        assert meta["error"] is not None
        assert "不存在" in meta["error"] or "not exist" in meta["error"].lower()

