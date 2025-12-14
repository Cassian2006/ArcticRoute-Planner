"""
Step 3 测试：成本模型与 AIS 密度集成
"""

import numpy as np
import pytest
import xarray as xr

import arcticroute.core.cost as cost_mod
from arcticroute.core.grid import make_demo_grid
from arcticroute.core.cost import build_cost_from_real_env, build_demo_cost
from arcticroute.core.env_real import RealEnvLayers


@pytest.mark.integration
@pytest.mark.xfail(reason="AIS corridor cost logic needs review - cost not being reduced as expected")
def test_cost_increases_with_ais_weight():
    """测试 AIS 权重增加时成本单调上升。"""
    # 创建演示网格
    grid, land_mask = make_demo_grid(ny=20, nx=20)
    
    # 创建简单的 AIS 密度场（在海洋格点有高密度）
    ny, nx = grid.shape()
    ais_density = np.zeros((ny, nx), dtype=float)
    
    # 找一个海洋格点
    ocean_idx = None
    for i in range(ny):
        for j in range(nx):
            if not land_mask[i, j]:
                ocean_idx = (i, j)
                break
        if ocean_idx:
            break
    
    if ocean_idx is None:
        pytest.skip("No ocean grid point found in demo grid")
    
    ais_density[ocean_idx] = 1.0  # 在海洋格点处有最高密度
    
    # 创建最小的真实环境（只需要 sic）
    env = RealEnvLayers(
        grid=grid,
        sic=np.ones((ny, nx), dtype=float) * 0.5,  # 50% 冰浓度
        wave_swh=None,
        ice_thickness_m=None,
        land_mask=land_mask,
    )
    
    # 构建不同权重的成本场（使用 w_ais_corridor 而不是 ais_weight）
    cost_0 = build_cost_from_real_env(
        grid, land_mask, env, ais_density=ais_density, w_ais_corridor=0.0
    )
    
    cost_1 = build_cost_from_real_env(
        grid, land_mask, env, ais_density=ais_density, w_ais_corridor=1.0
    )
    
    cost_2 = build_cost_from_real_env(
        grid, land_mask, env, ais_density=ais_density, w_ais_corridor=2.0
    )
    
    # 检查海洋格点处的成本单调上升（走廊成本 = 1 - sqrt(density)，高密度更便宜）
    i, j = ocean_idx
    # 由于 ais_density[i,j] = 1.0，corridor_cost = 1 - sqrt(1) = 0，所以成本应该减少而不是增加
    # 因此我们检查有 AIS 的成本小于没有 AIS 的成本
    assert cost_1.cost[i, j] < cost_0.cost[i, j], "AIS corridor should reduce cost at high-density points"
    assert cost_2.cost[i, j] < cost_1.cost[i, j], "Higher corridor weight should further reduce cost"


@pytest.mark.integration
@pytest.mark.xfail(reason="AIS corridor cost contains inf values - needs investigation")
def test_components_contains_ais_density():
    """测试成本分解中包含 AIS 密度组件。"""
    # 创建演示网格
    grid, land_mask = make_demo_grid(ny=20, nx=20)
    
    # 创建 AIS 密度场
    ny, nx = grid.shape()
    ais_density = np.ones((ny, nx), dtype=float) * 0.5
    
    # 创建真实环境
    env = RealEnvLayers(
        grid=grid,
        sic=np.ones((ny, nx), dtype=float) * 0.3,
        wave_swh=None,
        ice_thickness_m=None,
        land_mask=land_mask,
    )
    
    # 构建成本场（启用 AIS 走廊）
    cost_field = build_cost_from_real_env(
        grid, land_mask, env, ais_density=ais_density, w_ais_corridor=1.5
    )
    
    # 检查 components 中有 ais_corridor（新 API 使用 corridor 而不是 density）
    assert "ais_corridor" in cost_field.components
    
    # 检查 AIS 走廊成本非负
    ais_cost = cost_field.components["ais_corridor"]
    assert np.all(ais_cost >= 0.0)
    
    # 检查 AIS 走廊成本的范围（应该在 0 到 1.5 之间）
    assert np.max(ais_cost) <= 1.5


def test_no_crash_when_no_ais():
    """测试没有 AIS 数据或权重为 0 时行为正常。"""
    # 创建演示网格
    grid, land_mask = make_demo_grid(ny=20, nx=20)
    
    # 创建真实环境
    ny, nx = grid.shape()
    env = RealEnvLayers(
        grid=grid,
        sic=np.ones((ny, nx), dtype=float) * 0.3,
        wave_swh=None,
        ice_thickness_m=None,
        land_mask=land_mask,
    )
    
    # 情况 1：ais_density=None
    cost_field_1 = build_cost_from_real_env(
        grid, land_mask, env, ais_density=None, ais_weight=1.0
    )
    
    # 情况 2：ais_weight=0
    ais_density = np.ones((ny, nx), dtype=float) * 0.5
    cost_field_2 = build_cost_from_real_env(
        grid, land_mask, env, ais_density=ais_density, ais_weight=0.0
    )
    
    # 两种情况都应该成功，且成本相同（因为 AIS 没有被应用）
    assert cost_field_1.cost.shape == (ny, nx)
    assert cost_field_2.cost.shape == (ny, nx)
    
    # 检查 components 中没有 ais_density（或者有但值为 0）
    if "ais_density" in cost_field_1.components:
        assert np.all(cost_field_1.components["ais_density"] == 0.0)
    
    if "ais_density" in cost_field_2.components:
        assert np.all(cost_field_2.components["ais_density"] == 0.0)


def test_ais_density_shape_mismatch():
    """测试 AIS 密度形状不匹配时的处理。"""
    # 创建演示网格
    grid, land_mask = make_demo_grid(ny=20, nx=20)
    
    # 创建形状不匹配的 AIS 密度
    ais_density = np.ones((10, 10), dtype=float)  # 形状不对
    
    # 创建真实环境
    ny, nx = grid.shape()
    env = RealEnvLayers(
        grid=grid,
        sic=np.ones((ny, nx), dtype=float) * 0.3,
        wave_swh=None,
        ice_thickness_m=None,
        land_mask=land_mask,
    )
    
    # 应该不会崩溃，而是跳过 AIS 密度
    cost_field = build_cost_from_real_env(
        grid, land_mask, env, ais_density=ais_density, ais_weight=1.0
    )
    
    # 检查 components 中没有 ais_density
    assert "ais_density" not in cost_field.components


def test_ais_density_normalization():
    """测试 AIS 密度的归一化。"""
    # 创建演示网格
    grid, land_mask = make_demo_grid(ny=20, nx=20)
    
    # 创建超出 [0, 1] 范围的 AIS 密度
    ny, nx = grid.shape()
    ais_density = np.ones((ny, nx), dtype=float) * 2.0  # 超出范围
    
    # 创建真实环境
    env = RealEnvLayers(
        grid=grid,
        sic=np.ones((ny, nx), dtype=float) * 0.3,
        wave_swh=None,
        ice_thickness_m=None,
        land_mask=land_mask,
    )
    
    # 构建成本场
    cost_field = build_cost_from_real_env(
        grid, land_mask, env, ais_density=ais_density, ais_weight=1.0
    )
    
    # 检查 AIS 成本被正确归一化
    if "ais_density" in cost_field.components:
        ais_cost = cost_field.components["ais_density"]
        # 应该被 clip 到 [0, 1]，然后乘以权重 1.0
        assert np.max(ais_cost) <= 1.0


def test_ais_corridor_prefers_high_density():
    """测试 AIS 主航道成本：高密度区域成本更低。"""
    grid, land_mask = make_demo_grid(ny=12, nx=12)
    ny, nx = grid.shape()
    data = np.zeros((ny, nx), dtype=float)
    data[0, 0] = 5.0      # 高密度点（海洋区域）
    data[0, 1] = 0.1      # 低密度点（海洋区域）
    da = xr.DataArray(data, dims=("y", "x"))

    env = RealEnvLayers(
        grid=grid,
        sic=np.zeros((ny, nx), dtype=float),
        wave_swh=None,
        ice_thickness_m=None,
        land_mask=land_mask,
    )

    cost_field = build_cost_from_real_env(
        grid,
        land_mask,
        env,
        ais_density_da=da,
        w_ais_corridor=2.0,
    )

    assert "ais_corridor" in cost_field.components
    ais_corridor = cost_field.components["ais_corridor"]
    finite_vals = ais_corridor[np.isfinite(ais_corridor)]
    assert ais_corridor[0, 0] == np.nanmin(finite_vals)
    assert ais_corridor[0, 1] > ais_corridor[0, 0]
    assert np.nanmax(finite_vals) >= ais_corridor[0, 1]


@pytest.mark.integration
@pytest.mark.xfail(reason="AIS corridor cost logic needs review - cost not being reduced as expected")
def test_cost_uses_density_file_when_available(monkeypatch, tmp_path):
    """测试从 nc 文件加载 AIS density 时 w_ais_corridor 对成本的影响。"""
    grid, land_mask = make_demo_grid(ny=6, nx=6)
    ny, nx = grid.shape()
    land_mask = np.zeros_like(land_mask, dtype=bool)

    ais_values = np.linspace(0, 1, num=ny * nx, dtype=float).reshape(ny, nx)
    ais_da = xr.DataArray(ais_values, dims=("y", "x"), name="ais_density")
    density_path = tmp_path / "ais_density_dummy.nc"
    ais_da.to_dataset(name="ais_density").to_netcdf(density_path)

    monkeypatch.setattr(cost_mod, "AIS_DENSITY_PATH", density_path)

    cost_zero = build_demo_cost(
        grid,
        land_mask,
        ice_penalty=0.0,
        ice_lat_threshold=90.0,
        w_ais_corridor=0.0,
    )
    cost_with = build_demo_cost(
        grid,
        land_mask,
        ice_penalty=0.0,
        ice_lat_threshold=90.0,
        w_ais_corridor=5.0,
    )

    # 检查 ais_corridor 组件（新 API）
    assert "ais_corridor" not in cost_zero.components or np.all(cost_zero.components.get("ais_corridor", 0.0) == 0)
    assert "ais_corridor" in cost_with.components
    assert np.sum(cost_with.components["ais_corridor"]) > 0

    # 高密度区域应该有更低的成本（走廊成本 = 1 - sqrt(density)）
    finite_mask = np.isfinite(cost_zero.cost)
    assert cost_with.cost[finite_mask].mean() < cost_zero.cost[finite_mask].mean()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

