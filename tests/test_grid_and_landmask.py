"""
网格与陆地掩码的测试模块。
"""

import numpy as np

from arcticroute.core.grid import make_demo_grid, load_grid_with_landmask
from arcticroute.core import landmask as lm


def test_demo_grid_shape_and_range():
    """
    测试 demo grid 的形状和范围。

    - 调用 make_demo_grid()；
    - 断言纬度/经度单调递增，形状合理（ny>10, nx>10）；
    - land_mask 里既有 True 也有 False。
    """
    grid, land_mask = make_demo_grid()

    ny, nx = grid.shape()
    assert ny > 10, f"ny={ny} should be > 10"
    assert nx > 10, f"nx={nx} should be > 10"

    # 检查纬度单调递增
    lat_first_col = grid.lat2d[:, 0]
    assert np.all(np.diff(lat_first_col) > 0), "latitude should be monotonically increasing"

    # 检查经度单调递增
    lon_first_row = grid.lon2d[0, :]
    assert np.all(np.diff(lon_first_row) > 0), "longitude should be monotonically increasing"

    # 检查 land_mask 既有 True 也有 False
    assert land_mask.dtype == bool, f"land_mask dtype should be bool, got {land_mask.dtype}"
    assert np.any(land_mask), "land_mask should have at least one True"
    assert np.any(~land_mask), "land_mask should have at least one False"


def test_load_grid_with_landmask_demo():
    """
    测试 load_grid_with_landmask 的 demo 模式。

    - 调用 load_grid_with_landmask(prefer_real=False)；
    - 断言 meta["source"] == "demo"；
    - 断言 land_mask.shape == grid.shape()。
    """
    grid, land_mask, meta = load_grid_with_landmask(prefer_real=False)

    assert meta["source"] == "demo", f"source should be 'demo', got {meta['source']}"
    assert land_mask.shape == grid.shape(), (
        f"land_mask shape {land_mask.shape} != grid shape {grid.shape()}"
    )


def test_landmask_info_basic():
    """
    测试 LandMaskInfo 的基本属性。

    - 调用 landmask.load_landmask(prefer_real=False)；
    - 断言 0 < frac_land < 1；
    - 断言 land_mask.dtype == bool。
    """
    info = lm.load_landmask(prefer_real=False)

    assert 0 < info.frac_land < 1, (
        f"frac_land should be between 0 and 1, got {info.frac_land}"
    )
    assert info.frac_ocean == 1.0 - info.frac_land, (
        f"frac_ocean should equal 1 - frac_land"
    )
    assert info.land_mask.dtype == bool, (
        f"land_mask dtype should be bool, got {info.land_mask.dtype}"
    )
    assert info.source == "demo", f"source should be 'demo', got {info.source}"











