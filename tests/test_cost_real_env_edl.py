"""
EDL 风险与成本集成的单元测试。

测试项：
  1. test_build_cost_with_edl_disabled_equals_prev_behavior: EDL 禁用时行为不变
  2. test_build_cost_with_edl_enabled_adds_component: EDL 启用时添加成本组件
  3. test_build_cost_with_edl_and_no_torch_does_not_crash: 无 torch 时不报错
"""

from __future__ import annotations

import numpy as np
import pytest

from arcticroute.core.grid import make_demo_grid
from arcticroute.core.cost import build_cost_from_real_env
from arcticroute.core.env_real import RealEnvLayers
from arcticroute.core.eco.vessel_profiles import get_default_profiles
from arcticroute.core.edl_backend_miles import has_miles_guess


def _has_torch() -> bool:
    """检测当前环境是否有 PyTorch。"""
    try:
        import torch  # type: ignore
        return True
    except Exception:
        return False


def _has_edl_backend() -> bool:
    """检测当前环境是否有任何 EDL 后端（torch 或 miles-guess）。"""
    return _has_torch() or has_miles_guess()


class TestBuildCostWithEDLDisabled:
    """测试 EDL 禁用时的成本构建行为。"""

    def test_build_cost_with_edl_disabled_equals_prev_behavior(self):
        """
        调用 build_cost_from_real_env(..., use_edl=False, w_edl=0.0)，
        与旧版行为一致。

        预期：
          - components 中不包含 "edl_risk"
          - 总成本矩阵与不传 EDL 参数时相同
        """
        grid, land_mask = make_demo_grid(ny=10, nx=15)
        ny, nx = grid.shape()

        # 构造简单的环境数据
        sic = np.random.rand(ny, nx) * 0.5  # 0..0.5
        wave_swh = np.random.rand(ny, nx) * 3.0  # 0..3m
        ice_thickness = np.random.rand(ny, nx) * 0.5  # 0..0.5m

        env = RealEnvLayers(
            sic=sic,
            wave_swh=wave_swh,
            ice_thickness_m=ice_thickness,
        )

        # 调用两次：一次不传 EDL 参数，一次显式禁用
        cost_field_1 = build_cost_from_real_env(
            grid, land_mask, env,
            ice_penalty=4.0,
            wave_penalty=2.0,
        )

        cost_field_2 = build_cost_from_real_env(
            grid, land_mask, env,
            ice_penalty=4.0,
            wave_penalty=2.0,
            use_edl=False,
            w_edl=0.0,
        )

        # 验证成本矩阵相同
        assert np.allclose(cost_field_1.cost, cost_field_2.cost, equal_nan=True)

        # 验证 components 中都不包含 "edl_risk"
        assert "edl_risk" not in cost_field_1.components
        assert "edl_risk" not in cost_field_2.components

    def test_build_cost_with_edl_disabled_has_base_components(self):
        """验证 EDL 禁用时，基础成本组件仍然存在。"""
        grid, land_mask = make_demo_grid(ny=10, nx=15)
        ny, nx = grid.shape()

        sic = np.random.rand(ny, nx) * 0.5
        env = RealEnvLayers(sic=sic)

        cost_field = build_cost_from_real_env(
            grid, land_mask, env,
            use_edl=False,
            w_edl=0.0,
        )

        # 验证基础组件存在
        assert "base_distance" in cost_field.components
        assert "ice_risk" in cost_field.components


class TestBuildCostWithEDLEnabled:
    """测试 EDL 启用时的成本构建行为。"""

    def test_build_cost_with_edl_enabled_adds_component(self):
        """
        在 use_edl=True, w_edl>0 下，检查：
          - cost_field.components 中包含 "edl_risk"
          - 总成本矩阵与 w_edl=0 时不同
        """
        grid, land_mask = make_demo_grid(ny=10, nx=15)
        ny, nx = grid.shape()

        sic = np.random.rand(ny, nx) * 0.5
        wave_swh = np.random.rand(ny, nx) * 3.0
        ice_thickness = np.random.rand(ny, nx) * 0.5

        env = RealEnvLayers(
            sic=sic,
            wave_swh=wave_swh,
            ice_thickness_m=ice_thickness,
        )

        # 构建两个成本场：一个禁用 EDL，一个启用
        cost_field_no_edl = build_cost_from_real_env(
            grid, land_mask, env,
            ice_penalty=4.0,
            wave_penalty=2.0,
            use_edl=False,
            w_edl=0.0,
        )

        cost_field_with_edl = build_cost_from_real_env(
            grid, land_mask, env,
            ice_penalty=4.0,
            wave_penalty=2.0,
            use_edl=True,
            w_edl=1.0,
        )

        # 验证 EDL 启用时包含 "edl_risk" 组件
        assert "edl_risk" in cost_field_with_edl.components

        # 验证 EDL 风险组件的形状和数值范围
        edl_risk = cost_field_with_edl.components["edl_risk"]
        assert edl_risk.shape == (ny, nx)
        assert np.all(edl_risk >= 0.0)  # 成本应该非负

        # 验证总成本不同（至少在某些位置）
        # 注意：可能在陆地上相同（都是 inf），所以检查海洋部分
        ocean_mask = ~land_mask
        if np.any(ocean_mask):
            ocean_cost_no_edl = cost_field_no_edl.cost[ocean_mask]
            ocean_cost_with_edl = cost_field_with_edl.cost[ocean_mask]
            # 至少有一些位置的成本应该不同
            assert not np.allclose(ocean_cost_no_edl, ocean_cost_with_edl)

    def test_build_cost_with_edl_different_weights(self):
        """验证不同的 EDL 权重会产生不同的成本。"""
        grid, land_mask = make_demo_grid(ny=10, nx=15)
        ny, nx = grid.shape()

        sic = np.random.rand(ny, nx) * 0.5
        env = RealEnvLayers(sic=sic)

        # 构建两个成本场：不同的 w_edl
        cost_field_w1 = build_cost_from_real_env(
            grid, land_mask, env,
            use_edl=True,
            w_edl=0.5,
        )

        cost_field_w2 = build_cost_from_real_env(
            grid, land_mask, env,
            use_edl=True,
            w_edl=2.0,
        )

        # 验证两个成本场都包含 "edl_risk"
        assert "edl_risk" in cost_field_w1.components
        assert "edl_risk" in cost_field_w2.components

        # 验证 edl_risk 组件都是有限的正数
        edl_risk_1 = cost_field_w1.components["edl_risk"]
        edl_risk_2 = cost_field_w2.components["edl_risk"]

        # 验证两个权重的成本都是有限的
        assert np.all(np.isfinite(edl_risk_1[~land_mask]))
        assert np.all(np.isfinite(edl_risk_2[~land_mask]))

        # 验证权重为 2.0 的成本大于权重为 0.5 的成本（在海洋部分）
        ocean_mask = ~land_mask
        assert np.mean(edl_risk_2[ocean_mask]) > np.mean(edl_risk_1[ocean_mask])


class TestBuildCostWithEDLAndNoTorch:
    """测试 EDL 在无 PyTorch 时的行为。

    注意：这个测试类中的测试用例专门用于验证"当环境中没有 EDL 后端时"的降级行为。
    如果当前环境已经有 EDL 后端（torch 或 miles-guess），这些测试会被跳过。
    """

    @pytest.mark.skipif(
        _has_edl_backend(),
        reason="当前环境已有 EDL 后端（torch/miles-guess），此测试仅在无 EDL 后端环境中有效"
    )
    def test_build_cost_with_edl_and_no_torch_does_not_crash(self, monkeypatch):
        """
        在测试中模拟 TORCH_AVAILABLE=False，
        调用 build_cost_from_real_env 确保不会抛异常，
        并且 edl_risk 组件存在（哪怕是占位值）。
        """
        import arcticroute.ml.edl_core as edl_module

        original_torch_available = edl_module.TORCH_AVAILABLE
        edl_module.TORCH_AVAILABLE = False

        try:
            grid, land_mask = make_demo_grid(ny=10, nx=15)
            ny, nx = grid.shape()

            sic = np.random.rand(ny, nx) * 0.5
            env = RealEnvLayers(sic=sic)

            # 调用 build_cost_from_real_env，应该不报错
            cost_field = build_cost_from_real_env(
                grid, land_mask, env,
                use_edl=True,
                w_edl=1.0,
            )

            # 验证 edl_risk 组件存在
            assert "edl_risk" in cost_field.components

            # 验证 edl_risk 的形状和数值
            edl_risk = cost_field.components["edl_risk"]
            assert edl_risk.shape == (ny, nx)
            # 在 fallback 模式下，risk_mean 应该是 0，所以 edl_risk = 1.0 * 0 = 0
            assert np.allclose(edl_risk, 0.0)

        finally:
            edl_module.TORCH_AVAILABLE = original_torch_available

    @pytest.mark.skipif(
        _has_edl_backend(),
        reason="当前环境已有 EDL 后端（torch/miles-guess），此测试仅在无 EDL 后端环境中有效"
    )
    def test_build_cost_with_edl_fallback_no_exception(self, monkeypatch):
        """验证 EDL fallback 时不会抛异常。"""
        import arcticroute.ml.edl_core as edl_module

        original_torch_available = edl_module.TORCH_AVAILABLE
        edl_module.TORCH_AVAILABLE = False

        try:
            grid, land_mask = make_demo_grid(ny=5, nx=5)
            env = RealEnvLayers(sic=np.ones((5, 5)) * 0.3)

            # 应该不报错
            cost_field = build_cost_from_real_env(
                grid, land_mask, env,
                use_edl=True,
                w_edl=2.0,
            )

            assert cost_field is not None
            assert "edl_risk" in cost_field.components

        finally:
            edl_module.TORCH_AVAILABLE = original_torch_available


class TestBuildCostWithEDLAndVessel:
    """测试 EDL 与船舶冰级约束的组合。"""

    def test_build_cost_with_edl_and_ice_class_constraints(self):
        """验证 EDL 与冰级约束可以同时工作。"""
        grid, land_mask = make_demo_grid(ny=10, nx=15)
        ny, nx = grid.shape()

        sic = np.random.rand(ny, nx) * 0.5
        ice_thickness = np.random.rand(ny, nx) * 1.0  # 0..1m
        env = RealEnvLayers(sic=sic, ice_thickness_m=ice_thickness)

        vessel_profiles = get_default_profiles()
        vessel = vessel_profiles["panamax"]

        # 构建成本场，同时启用 EDL 和冰级约束
        cost_field = build_cost_from_real_env(
            grid, land_mask, env,
            vessel_profile=vessel,
            use_edl=True,
            w_edl=1.0,
        )

        # 验证所有相关组件都存在
        assert "base_distance" in cost_field.components
        assert "ice_risk" in cost_field.components
        assert "ice_class_soft" in cost_field.components
        assert "ice_class_hard" in cost_field.components
        assert "edl_risk" in cost_field.components

    def test_build_cost_with_edl_zero_weight_no_component(self):
        """验证 w_edl=0 时，即使 use_edl=True，也不添加 edl_risk 组件。"""
        grid, land_mask = make_demo_grid(ny=10, nx=15)
        ny, nx = grid.shape()

        sic = np.random.rand(ny, nx) * 0.5
        env = RealEnvLayers(sic=sic)

        cost_field = build_cost_from_real_env(
            grid, land_mask, env,
            use_edl=True,
            w_edl=0.0,  # 权重为 0
        )

        # 当 w_edl=0 时，不应该添加 edl_risk 组件
        assert "edl_risk" not in cost_field.components


class TestBuildCostWithEDLFeatures:
    """测试 EDL 特征构造的正确性。"""

    def test_build_cost_with_edl_feature_normalization(self):
        """验证 EDL 特征的归一化范围。"""
        grid, land_mask = make_demo_grid(ny=10, nx=15)
        ny, nx = grid.shape()

        # 构造极端的环境数据
        sic = np.ones((ny, nx)) * 1.0  # 全 1
        wave_swh = np.ones((ny, nx)) * 10.0  # 全 10m
        ice_thickness = np.ones((ny, nx)) * 2.0  # 全 2m

        env = RealEnvLayers(
            sic=sic,
            wave_swh=wave_swh,
            ice_thickness_m=ice_thickness,
        )

        # 构建成本场
        cost_field = build_cost_from_real_env(
            grid, land_mask, env,
            use_edl=True,
            w_edl=1.0,
        )

        # 验证 edl_risk 存在且数值合理
        assert "edl_risk" in cost_field.components
        edl_risk = cost_field.components["edl_risk"]
        assert np.all(np.isfinite(edl_risk[~land_mask]))  # 海洋部分应该有限

    def test_build_cost_with_edl_missing_features(self):
        """验证 EDL 在某些特征缺失时的处理。"""
        grid, land_mask = make_demo_grid(ny=10, nx=15)
        ny, nx = grid.shape()

        # 仅提供 sic，其他为 None
        env = RealEnvLayers(sic=np.random.rand(ny, nx) * 0.5)

        # 应该不报错
        cost_field = build_cost_from_real_env(
            grid, land_mask, env,
            use_edl=True,
            w_edl=1.0,
        )

        assert "edl_risk" in cost_field.components

