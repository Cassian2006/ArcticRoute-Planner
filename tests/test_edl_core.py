"""
EDL 核心模块的单元测试。

测试项：
  1. test_edl_fallback_without_torch: 模拟 TORCH_AVAILABLE=False 时的 fallback 行为
  2. test_edl_with_torch_shapes_match: 有 torch 时，验证输出形状和数值范围
  3. test_edl_config_num_classes_effect: 改变 num_classes 后的稳定性
"""

from __future__ import annotations

import numpy as np
import pytest

from arcticroute.ml.edl_core import (
    EDLConfig,
    EDLGridOutput,
    run_edl_on_features,
    TORCH_AVAILABLE,
)


class TestEDLFallback:
    """测试 EDL fallback 行为（无 torch）。"""

    def test_edl_fallback_without_torch(self, monkeypatch):
        """
        模拟 TORCH_AVAILABLE=False 时，调用 run_edl_on_features 能返回形状正确的输出。

        预期：
          - risk_mean shape = (H, W)，值为 0
          - uncertainty shape = (H, W)，值为 1
        """
        # monkeypatch TORCH_AVAILABLE 为 False
        import arcticroute.ml.edl_core as edl_module

        original_torch_available = edl_module.TORCH_AVAILABLE
        edl_module.TORCH_AVAILABLE = False

        try:
            # 构造简单的特征数组
            H, W, F = 4, 5, 3
            features = np.random.randn(H, W, F).astype(float)

            # 调用 run_edl_on_features
            output = run_edl_on_features(features)

            # 验证输出类型和形状
            assert isinstance(output, EDLGridOutput)
            assert output.risk_mean.shape == (H, W)
            assert output.uncertainty.shape == (H, W)

            # 验证数值（fallback 时应为占位符）
            assert np.allclose(output.risk_mean, 0.0)
            assert np.allclose(output.uncertainty, 1.0)

        finally:
            # 恢复原始值
            edl_module.TORCH_AVAILABLE = original_torch_available

    def test_edl_fallback_returns_numpy(self, monkeypatch):
        """验证 fallback 时返回的是 numpy 数组。"""
        import arcticroute.ml.edl_core as edl_module

        original_torch_available = edl_module.TORCH_AVAILABLE
        edl_module.TORCH_AVAILABLE = False

        try:
            H, W, F = 3, 3, 2
            features = np.random.randn(H, W, F).astype(float)

            output = run_edl_on_features(features)

            assert isinstance(output.risk_mean, np.ndarray)
            assert isinstance(output.uncertainty, np.ndarray)
            assert output.risk_mean.dtype in [np.float32, np.float64]
            assert output.uncertainty.dtype in [np.float32, np.float64]

        finally:
            edl_module.TORCH_AVAILABLE = original_torch_available


class TestEDLWithTorch:
    """测试 EDL 在有 torch 时的行为。"""

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_edl_with_torch_shapes_match(self):
        """
        在有 torch 的环境下，验证输出形状和数值范围。

        预期：
          - risk_mean shape = (H, W)，值在 [0, 1]
          - uncertainty shape = (H, W)，值 >= 0
        """
        H, W, F = 4, 5, 3
        features = np.random.randn(H, W, F).astype(float)

        output = run_edl_on_features(features)

        # 验证形状
        assert output.risk_mean.shape == (H, W)
        assert output.uncertainty.shape == (H, W)

        # 验证数值范围
        assert np.all(output.risk_mean >= 0.0) and np.all(output.risk_mean <= 1.0), \
            f"risk_mean 超出 [0, 1] 范围: min={output.risk_mean.min()}, max={output.risk_mean.max()}"
        assert np.all(output.uncertainty >= 0.0), \
            f"uncertainty 包含负值: min={output.uncertainty.min()}"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_edl_with_torch_output_types(self):
        """验证 EDL 输出的类型和形状。"""
        H, W, F = 3, 3, 2
        features = np.random.randn(H, W, F).astype(float)

        output1 = run_edl_on_features(features)
        output2 = run_edl_on_features(features)

        # 验证输出类型
        assert isinstance(output1.risk_mean, np.ndarray)
        assert isinstance(output1.uncertainty, np.ndarray)
        assert isinstance(output2.risk_mean, np.ndarray)
        assert isinstance(output2.uncertainty, np.ndarray)

        # 验证形状一致
        assert output1.risk_mean.shape == output2.risk_mean.shape == (H, W)
        assert output1.uncertainty.shape == output2.uncertainty.shape == (H, W)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_edl_with_torch_different_inputs(self):
        """验证不同输入会产生不同的输出。"""
        H, W, F = 3, 3, 2

        features1 = np.zeros((H, W, F), dtype=float)
        features2 = np.ones((H, W, F), dtype=float)

        output1 = run_edl_on_features(features1)
        output2 = run_edl_on_features(features2)

        # 不同的输入应该产生不同的输出（至少在某些位置）
        assert not np.allclose(output1.risk_mean, output2.risk_mean)


class TestEDLConfig:
    """测试 EDL 配置的影响。"""

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_edl_config_num_classes_effect(self):
        """
        改变 num_classes 后，内部计算不报错。

        预期：
          - num_classes=2, 3, 4 都能正常运行
          - 输出形状和范围仍然正确
        """
        H, W, F = 3, 3, 2
        features = np.random.randn(H, W, F).astype(float)

        for num_classes in [2, 3, 4, 5]:
            config = EDLConfig(num_classes=num_classes)
            output = run_edl_on_features(features, config=config)

            # 验证输出形状
            assert output.risk_mean.shape == (H, W)
            assert output.uncertainty.shape == (H, W)

            # 验证数值范围
            assert np.all(output.risk_mean >= 0.0) and np.all(output.risk_mean <= 1.0)
            assert np.all(output.uncertainty >= 0.0)

    def test_edl_config_default_values(self):
        """验证 EDLConfig 的默认值。"""
        config = EDLConfig()
        assert config.num_classes == 3


class TestEDLGridOutput:
    """测试 EDLGridOutput 数据类。"""

    def test_edl_grid_output_creation(self):
        """验证 EDLGridOutput 可以正确创建。"""
        risk_mean = np.random.rand(4, 5)
        uncertainty = np.random.rand(4, 5)

        output = EDLGridOutput(risk_mean=risk_mean, uncertainty=uncertainty)

        assert output.risk_mean.shape == (4, 5)
        assert output.uncertainty.shape == (4, 5)
        assert np.array_equal(output.risk_mean, risk_mean)
        assert np.array_equal(output.uncertainty, uncertainty)


class TestEDLFeatureProcessing:
    """测试 EDL 的特征处理。"""

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_edl_with_different_feature_dims(self):
        """验证不同特征维度的处理。"""
        H, W = 3, 3

        for F in [1, 3, 5, 10]:
            features = np.random.randn(H, W, F).astype(float)
            output = run_edl_on_features(features)

            assert output.risk_mean.shape == (H, W)
            assert output.uncertainty.shape == (H, W)

    def test_edl_with_large_grid(self):
        """验证大网格的处理（性能和内存）。"""
        H, W, F = 100, 100, 5
        features = np.random.randn(H, W, F).astype(float)

        output = run_edl_on_features(features)

        assert output.risk_mean.shape == (H, W)
        assert output.uncertainty.shape == (H, W)

    def test_edl_with_nan_features(self):
        """验证包含 NaN 的特征处理（应该不报错）。"""
        H, W, F = 3, 3, 2
        features = np.random.randn(H, W, F).astype(float)
        features[0, 0, 0] = np.nan

        # 不应该报错
        output = run_edl_on_features(features)
        assert output.risk_mean.shape == (H, W)
        assert output.uncertainty.shape == (H, W)

