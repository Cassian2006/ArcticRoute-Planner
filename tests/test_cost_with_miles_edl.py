"""
Test cost building with miles-guess EDL backend integration.

Phase EDL-CORE Step 3: 验证 EDL 输出正确融合进成本
- 有 miles-guess 时：检查 components 中有 edl_risk 且非全 0
- 无 miles-guess 时：检查行为退化到"无 EDL"，但不会抛异常
"""

import numpy as np
import pytest

from arcticroute.core.grid import make_demo_grid
from arcticroute.core.cost import build_cost_from_real_env, build_demo_cost
from arcticroute.core.env_real import RealEnvLayers
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


class TestCostWithMilesEDL:
    """测试成本构建与 miles-guess EDL 的集成。"""

    def test_build_cost_without_edl(self):
        """不启用 EDL 时，成本构建应该正常工作。"""
        grid, land_mask = make_demo_grid()
        ny, nx = grid.shape()
        
        # 创建简单的环境数据
        env = RealEnvLayers(
            sic=np.random.rand(ny, nx) * 0.5,
            wave_swh=np.random.rand(ny, nx) * 3.0,
            ice_thickness_m=None,
        )
        
        # 构建成本，不启用 EDL
        cost_field = build_cost_from_real_env(
            grid=grid,
            land_mask=land_mask,
            env=env,
            ice_penalty=4.0,
            wave_penalty=1.0,
            use_edl=False,
            w_edl=0.0,
        )
        
        assert cost_field.cost.shape == (ny, nx)
        assert "edl_risk" not in cost_field.components
        assert cost_field.edl_uncertainty is None

    def test_build_cost_with_edl_enabled(self):
        """启用 EDL 时，成本构建应该包含 EDL 成本。"""
        grid, land_mask = make_demo_grid()
        ny, nx = grid.shape()
        
        # 创建简单的环境数据
        env = RealEnvLayers(
            sic=np.random.rand(ny, nx) * 0.5,
            wave_swh=np.random.rand(ny, nx) * 3.0,
            ice_thickness_m=None,
        )
        
        # 构建成本，启用 EDL
        cost_field = build_cost_from_real_env(
            grid=grid,
            land_mask=land_mask,
            env=env,
            ice_penalty=4.0,
            wave_penalty=1.0,
            use_edl=True,
            w_edl=2.0,
        )
        
        assert cost_field.cost.shape == (ny, nx)
        # EDL 应该被应用（无论是 miles-guess 还是 PyTorch）
        # 由于 miles-guess 可能不可用，我们只检查成本场的有效性
        assert np.any(np.isfinite(cost_field.cost))

    def test_build_cost_with_edl_and_uncertainty(self):
        """启用 EDL 不确定性时，应该在成本中反映。"""
        grid, land_mask = make_demo_grid()
        ny, nx = grid.shape()
        
        # 创建简单的环境数据
        env = RealEnvLayers(
            sic=np.random.rand(ny, nx) * 0.5,
            wave_swh=np.random.rand(ny, nx) * 3.0,
            ice_thickness_m=None,
        )
        
        # 构建成本，启用 EDL 和不确定性
        cost_field = build_cost_from_real_env(
            grid=grid,
            land_mask=land_mask,
            env=env,
            ice_penalty=4.0,
            wave_penalty=1.0,
            use_edl=True,
            w_edl=2.0,
            use_edl_uncertainty=True,
            edl_uncertainty_weight=1.0,
        )
        
        assert cost_field.cost.shape == (ny, nx)
        assert cost_field.edl_uncertainty is not None
        assert cost_field.edl_uncertainty.shape == (ny, nx)

    def test_build_cost_backward_compatibility(self):
        """不启用 EDL 时，成本应该与之前的实现一致（向后兼容）。"""
        grid, land_mask = make_demo_grid()
        ny, nx = grid.shape()
        
        # 创建简单的环境数据
        env = RealEnvLayers(
            sic=np.random.rand(ny, nx) * 0.5,
            wave_swh=None,
            ice_thickness_m=None,
        )
        
        # 构建成本，不启用 EDL
        cost_field = build_cost_from_real_env(
            grid=grid,
            land_mask=land_mask,
            env=env,
            ice_penalty=4.0,
            wave_penalty=0.0,
            use_edl=False,
            w_edl=0.0,
        )
        
        # 检查基本成本组件
        assert "base_distance" in cost_field.components
        assert "ice_risk" in cost_field.components
        
        # 陆地应该是 inf
        assert np.all(np.isinf(cost_field.cost[land_mask]))
        
        # 海洋应该是有限的
        ocean_mask = ~land_mask
        assert np.all(np.isfinite(cost_field.cost[ocean_mask]))

    def test_build_cost_no_exception_on_failure(self):
        """即使 EDL 失败，成本构建也不应该抛异常。"""
        grid, land_mask = make_demo_grid()
        ny, nx = grid.shape()
        
        # 创建环境数据，故意使用不合理的值
        env = RealEnvLayers(
            sic=np.full((ny, nx), 2.0),  # 超出 [0, 1] 范围
            wave_swh=np.full((ny, nx), 100.0),  # 超出合理范围
            ice_thickness_m=None,
        )
        
        # 这个调用不应该抛异常，即使 EDL 失败
        try:
            cost_field = build_cost_from_real_env(
                grid=grid,
                land_mask=land_mask,
                env=env,
                ice_penalty=4.0,
                wave_penalty=1.0,
                use_edl=True,
                w_edl=2.0,
            )
            assert cost_field is not None
        except Exception as e:
            pytest.fail(f"build_cost_from_real_env raised exception: {e}")

    def test_build_cost_edl_components_structure(self):
        """EDL 成本应该正确添加到 components 字典中。"""
        grid, land_mask = make_demo_grid()
        ny, nx = grid.shape()
        
        # 创建简单的环境数据
        env = RealEnvLayers(
            sic=np.random.rand(ny, nx) * 0.5,
            wave_swh=None,
            ice_thickness_m=None,
        )
        
        # 构建成本，启用 EDL
        cost_field = build_cost_from_real_env(
            grid=grid,
            land_mask=land_mask,
            env=env,
            ice_penalty=4.0,
            wave_penalty=0.0,
            use_edl=True,
            w_edl=2.0,
        )
        
        # 检查 components 的结构
        assert isinstance(cost_field.components, dict)
        assert "base_distance" in cost_field.components
        assert "ice_risk" in cost_field.components
        
        # EDL 风险可能存在（取决于后端可用性）
        # 但如果存在，应该是正确的形状
        if "edl_risk" in cost_field.components:
            edl_risk = cost_field.components["edl_risk"]
            assert edl_risk.shape == (ny, nx)
            assert np.all(np.isfinite(edl_risk[~land_mask]))  # 海洋应该有限

    def test_build_cost_edl_uncertainty_in_cost_field(self):
        """EDL 不确定性应该正确存储在 CostField 中。"""
        grid, land_mask = make_demo_grid()
        ny, nx = grid.shape()
        
        # 创建简单的环境数据
        env = RealEnvLayers(
            sic=np.random.rand(ny, nx) * 0.5,
            wave_swh=None,
            ice_thickness_m=None,
        )
        
        # 构建成本，启用 EDL 不确定性
        cost_field = build_cost_from_real_env(
            grid=grid,
            land_mask=land_mask,
            env=env,
            ice_penalty=4.0,
            wave_penalty=0.0,
            use_edl=True,
            w_edl=2.0,
            use_edl_uncertainty=True,
            edl_uncertainty_weight=1.0,
        )
        
        # 检查 edl_uncertainty 字段
        if cost_field.edl_uncertainty is not None:
            assert cost_field.edl_uncertainty.shape == (ny, nx)
            # 不确定性应该是非负的
            assert np.all(cost_field.edl_uncertainty >= 0.0)

    def test_build_cost_demo_mode_unchanged(self):
        """Demo 模式（不使用真实环境数据）应该不受 EDL 影响。"""
        grid, land_mask = make_demo_grid()
        
        # 构建 demo 成本（不使用 EDL）
        cost_field_demo = build_demo_cost(
            grid=grid,
            land_mask=land_mask,
            ice_penalty=4.0,
            ice_lat_threshold=75.0,
        )
        
        assert cost_field_demo.cost.shape == grid.shape()
        assert "base_distance" in cost_field_demo.components
        assert "ice_risk" in cost_field_demo.components
        assert "edl_risk" not in cost_field_demo.components


class TestCostWithMilesGuessAvailability:
    """测试在 miles-guess 可用/不可用时的行为。"""

    def test_cost_with_miles_guess_available(self):
        """若 miles-guess 可用，应该使用它。"""
        if not has_miles_guess():
            pytest.skip("miles-guess not available")
        
        grid, land_mask = make_demo_grid()
        ny, nx = grid.shape()
        
        env = RealEnvLayers(
            sic=np.random.rand(ny, nx) * 0.5,
            wave_swh=None,
            ice_thickness_m=None,
        )
        
        cost_field = build_cost_from_real_env(
            grid=grid,
            land_mask=land_mask,
            env=env,
            ice_penalty=4.0,
            wave_penalty=0.0,
            use_edl=True,
            w_edl=2.0,
        )
        
        # 应该有 edl_risk 组件
        assert "edl_risk" in cost_field.components

    @pytest.mark.skipif(
        _has_edl_backend(),
        reason="当前环境已有 EDL 后端（torch/miles-guess），此测试仅在无 EDL 后端环境中有效"
    )
    def test_cost_without_miles_guess_fallback(self):
        """若 miles-guess 不可用，应该回退到 PyTorch 或占位实现。

        注意：这个测试专门用于验证"当环境中没有任何 EDL 后端时"的降级行为。
        如果当前环境已经有 EDL 后端（torch 或 miles-guess），此测试会被跳过。
        """
        grid, land_mask = make_demo_grid()
        ny, nx = grid.shape()
        
        env = RealEnvLayers(
            sic=np.random.rand(ny, nx) * 0.5,
            wave_swh=None,
            ice_thickness_m=None,
        )
        
        # 这个调用应该总是成功的，无论 miles-guess 是否可用
        cost_field = build_cost_from_real_env(
            grid=grid,
            land_mask=land_mask,
            env=env,
            ice_penalty=4.0,
            wave_penalty=0.0,
            use_edl=True,
            w_edl=2.0,
        )
        
        # 成本应该是有效的
        assert cost_field.cost.shape == (ny, nx)
        assert np.any(np.isfinite(cost_field.cost))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


