"""
Smoke test for miles-guess EDL backend.

Phase EDL-CORE Step 2: 验证 run_miles_edl_on_grid() 的基本功能
- 检测 miles-guess 库的可用性
- 若可用，验证推理输出的形状和数值范围
- 若不可用，验证占位实现的行为
"""

import numpy as np
import pytest

from arcticroute.core.edl_backend_miles import (
    has_miles_guess,
    edl_dummy_on_grid,
    run_miles_edl_on_grid,
)


class TestMilesGuessDetection:
    """测试 miles-guess 库的检测。"""

    def test_has_miles_guess_returns_bool(self):
        """has_miles_guess() 应该返回布尔值。"""
        result = has_miles_guess()
        assert isinstance(result, bool)


class TestDummyImplementation:
    """测试占位实现。"""

    def test_edl_dummy_on_grid_shape(self):
        """占位实现应该返回正确形状的数组。"""
        H, W = 10, 20
        output = edl_dummy_on_grid((H, W))

        assert output.risk.shape == (H, W)
        assert output.uncertainty.shape == (H, W)

    def test_edl_dummy_on_grid_values(self):
        """占位实现的数值应该在合理范围内。"""
        H, W = 10, 20
        output = edl_dummy_on_grid((H, W))

        # risk 应该全为 0
        assert np.allclose(output.risk, 0.0)

        # uncertainty 应该全为 0.5
        assert np.allclose(output.uncertainty, 0.5)

    def test_edl_dummy_on_grid_meta(self):
        """占位实现应该包含元数据。"""
        H, W = 10, 20
        output = edl_dummy_on_grid((H, W))

        assert "source" in output.meta
        assert output.meta["source"] == "placeholder"


class TestRunMilesEdlOnGrid:
    """测试 run_miles_edl_on_grid() 函数。"""

    def test_run_miles_edl_on_grid_basic_shape(self):
        """推理输出应该与输入网格形状一致。"""
        H, W = 10, 20
        sic = np.random.rand(H, W)

        output = run_miles_edl_on_grid(sic)

        assert output.risk.shape == (H, W)
        assert output.uncertainty.shape == (H, W)

    def test_run_miles_edl_on_grid_with_optional_inputs(self):
        """推理应该支持可选的输入（swh, ice_thickness, lat, lon）。"""
        H, W = 10, 20
        sic = np.random.rand(H, W)
        swh = np.random.rand(H, W) * 5.0  # 0-5 m
        ice_thickness = np.random.rand(H, W) * 2.0  # 0-2 m
        lat = np.linspace(60, 85, H)[:, np.newaxis] * np.ones((1, W))
        lon = np.linspace(-180, 180, W)[np.newaxis, :] * np.ones((H, 1))

        output = run_miles_edl_on_grid(
            sic,
            swh=swh,
            ice_thickness=ice_thickness,
            grid_lat=lat,
            grid_lon=lon,
        )

        assert output.risk.shape == (H, W)
        assert output.uncertainty.shape == (H, W)

    def test_run_miles_edl_on_grid_values_in_range(self):
        """推理输出的数值应该在合理范围内。"""
        H, W = 10, 20
        sic = np.random.rand(H, W)

        output = run_miles_edl_on_grid(sic)

        # risk 应该在 [0, 1] 范围内
        assert np.all(output.risk >= 0.0) and np.all(output.risk <= 1.0)

        # uncertainty 应该非负
        assert np.all(output.uncertainty >= 0.0)

    def test_run_miles_edl_on_grid_meta(self):
        """推理输出应该包含元数据。"""
        H, W = 10, 20
        sic = np.random.rand(H, W)

        output = run_miles_edl_on_grid(sic)

        assert "source" in output.meta
        # source 应该是 "miles-guess" 或 "placeholder"
        assert output.meta["source"] in ["miles-guess", "placeholder"]

    def test_run_miles_edl_on_grid_no_exception(self):
        """推理不应该抛出异常，即使 miles-guess 不可用。"""
        H, W = 10, 20
        sic = np.random.rand(H, W)

        # 这个调用不应该抛出异常
        try:
            output = run_miles_edl_on_grid(sic)
            assert output is not None
        except Exception as e:
            pytest.fail(f"run_miles_edl_on_grid raised exception: {e}")

    def test_run_miles_edl_on_grid_with_all_zeros(self):
        """推理应该处理全零输入。"""
        H, W = 10, 20
        sic = np.zeros((H, W))

        output = run_miles_edl_on_grid(sic)

        assert output.risk.shape == (H, W)
        assert output.uncertainty.shape == (H, W)
        # 全零输入应该产生较低的风险
        assert np.mean(output.risk) < 0.5

    def test_run_miles_edl_on_grid_with_all_ones(self):
        """推理应该处理全 1 输入。"""
        H, W = 10, 20
        sic = np.ones((H, W))

        output = run_miles_edl_on_grid(sic)

        assert output.risk.shape == (H, W)
        assert output.uncertainty.shape == (H, W)
        # 若 miles-guess 不可用，返回占位实现（risk=0）
        # 若 miles-guess 可用，全 1 输入应该产生较高的风险
        if output.meta["source"] == "miles-guess":
            assert np.mean(output.risk) > 0.3
        else:
            # 占位实现返回 risk=0
            assert np.allclose(output.risk, 0.0)

    def test_run_miles_edl_on_grid_deterministic_without_miles_guess(self):
        """
        若 miles-guess 不可用，推理应该返回占位实现（确定性）。
        
        注意：如果 miles-guess 可用，此测试可能不适用。
        """
        if has_miles_guess():
            pytest.skip("miles-guess is available, skipping placeholder test")

        H, W = 10, 20
        sic = np.random.rand(H, W)

        output1 = run_miles_edl_on_grid(sic)
        output2 = run_miles_edl_on_grid(sic)

        # 两次调用应该返回相同的占位结果
        assert np.allclose(output1.risk, output2.risk)
        assert np.allclose(output1.uncertainty, output2.uncertainty)
        assert output1.meta["source"] == output2.meta["source"]


class TestEdlBackendIntegration:
    """测试 EDL 后端与其他模块的集成。"""

    def test_edl_output_compatible_with_cost_module(self):
        """
        EDLGridOutput 应该与 cost.py 中的使用兼容。
        
        cost.py 期望的字段：
        - risk: 风险分数，用于 edl_cost = w_edl * risk
        - uncertainty: 不确定性，用于 unc_cost = weight * uncertainty
        """
        H, W = 10, 20
        sic = np.random.rand(H, W)

        output = run_miles_edl_on_grid(sic)

        # 应该能够进行标量乘法
        w_edl = 2.0
        edl_cost = w_edl * output.risk
        assert edl_cost.shape == (H, W)

        # 应该能够进行加法
        unc_weight = 1.0
        unc_cost = unc_weight * output.uncertainty
        total_edl_cost = edl_cost + unc_cost
        assert total_edl_cost.shape == (H, W)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

