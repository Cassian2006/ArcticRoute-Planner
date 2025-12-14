"""
单元测试：用合成数据验证拟合脚本能找回真实指数。

测试不依赖真实大文件，使用合成数据验证拟合算法的正确性。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 导入拟合脚本中的函数
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fit_speed_exponents import (
    compute_speed_ratios,
    grid_search_fit,
    _fit_linear_model,
)


class TestSpeedRatioComputation:
    """测试速度比计算。"""
    
    def test_speed_ratio_basic(self):
        """测试基本的速度比计算。"""
        records = [
            {'timestamp': '2024-12-01T00:00:00', 'lat': 70.0, 'lon': -30.0, 'sog': 10.0, 'mmsi': '123456'},
            {'timestamp': '2024-12-01T01:00:00', 'lat': 70.1, 'lon': -29.9, 'sog': 12.0, 'mmsi': '123456'},
            {'timestamp': '2024-12-01T02:00:00', 'lat': 70.2, 'lon': -29.8, 'sog': 8.0, 'mmsi': '123456'},
        ]
        
        df, mmsi_baselines = compute_speed_ratios(records, baseline_percentile=80.0)
        
        assert len(df) == 3
        assert '123456' in mmsi_baselines
        assert mmsi_baselines['123456'] > 0
        assert 'speed_ratio' in df.columns
        assert 'y' in df.columns
        
        # 验证 speed_ratio 在有效范围内
        assert (df['speed_ratio'] >= 0.05).all()
        assert (df['speed_ratio'] <= 1.2).all()
    
    def test_speed_ratio_multiple_mmsi(self):
        """测试多个 MMSI 的速度比计算。"""
        records = [
            {'timestamp': '2024-12-01T00:00:00', 'lat': 70.0, 'lon': -30.0, 'sog': 10.0, 'mmsi': '111111'},
            {'timestamp': '2024-12-01T01:00:00', 'lat': 70.1, 'lon': -29.9, 'sog': 15.0, 'mmsi': '111111'},
            {'timestamp': '2024-12-01T02:00:00', 'lat': 70.2, 'lon': -29.8, 'sog': 20.0, 'mmsi': '222222'},
            {'timestamp': '2024-12-01T03:00:00', 'lat': 70.3, 'lon': -29.7, 'sog': 25.0, 'mmsi': '222222'},
        ]
        
        df, mmsi_baselines = compute_speed_ratios(records)
        
        assert len(mmsi_baselines) == 2
        assert '111111' in mmsi_baselines
        assert '222222' in mmsi_baselines
        assert mmsi_baselines['111111'] != mmsi_baselines['222222']


class TestLinearModelFitting:
    """测试线性模型拟合。"""
    
    def test_fit_linear_model_basic(self):
        """测试基本的线性模型拟合。"""
        # 生成合成数据：y = 1.0 - 0.5*x1 - 0.3*x2
        np.random.seed(42)
        n_train = 100
        n_holdout = 30
        
        x1_train = np.random.uniform(0, 1, n_train)
        x2_train = np.random.uniform(0, 1, n_train)
        y_train = 1.0 - 0.5 * x1_train - 0.3 * x2_train + np.random.normal(0, 0.01, n_train)
        
        x1_holdout = np.random.uniform(0, 1, n_holdout)
        x2_holdout = np.random.uniform(0, 1, n_holdout)
        y_holdout = 1.0 - 0.5 * x1_holdout - 0.3 * x2_holdout + np.random.normal(0, 0.01, n_holdout)
        
        df_train = pd.DataFrame({
            'sic': x1_train,
            'wave_swh': x2_train,
            'y': y_train,
        })
        
        df_holdout = pd.DataFrame({
            'sic': x1_holdout,
            'wave_swh': x2_holdout,
            'y': y_holdout,
        })
        
        b0, b1, b2, rmse_train, rmse_holdout, r2_holdout = _fit_linear_model(
            df_train, df_holdout, p=1.0, q=1.0
        )
        
        # 验证系数接近真实值
        assert abs(b0 - 1.0) < 0.1
        assert abs(b1 - (-0.5)) < 0.1
        assert abs(b2 - (-0.3)) < 0.1
        
        # 验证 RMSE 较小
        assert rmse_train < 0.05
        assert rmse_holdout < 0.05
        
        # 验证 R2 较高
        assert r2_holdout > 0.8


class TestGridSearchFitting:
    """测试网格搜索拟合。"""
    
    def test_grid_search_recovery(self):
        """
        测试网格搜索能否找回真实指数。
        
        生成合成数据，使用已知的 p0, q0，然后验证拟合能否恢复这些值。
        """
        np.random.seed(42)
        
        # 真实参数
        p0 = 1.5
        q0 = 2.0
        b1_true = -0.5
        b2_true = -0.3
        b0_true = 0.5
        
        # 生成合成数据
        n_samples = 500
        sic = np.random.uniform(0.1, 1.0, n_samples)
        wave_swh = np.random.uniform(0.5, 4.0, n_samples)
        
        # 生成目标变量
        x1 = sic ** p0
        x2 = wave_swh ** q0
        y = b0_true + b1_true * x1 + b2_true * x2 + np.random.normal(0, 0.05, n_samples)
        
        df = pd.DataFrame({
            'sic': sic,
            'wave_swh': wave_swh,
            'y': y,
        })
        
        # 运行网格搜索
        p_fit, q_fit, b0_fit, b1_fit, b2_fit, rmse_train, rmse_holdout, r2_holdout = grid_search_fit(
            df,
            p_min=1.0,
            p_max=2.5,
            q_min=1.5,
            q_max=2.5,
            coarse_step=0.2,
            fine_step=0.05,
            holdout_ratio=0.2,
            random_seed=42,
        )
        
        # 验证拟合的指数接近真实值
        # 允许误差 ±0.2（相对宽松的容差）
        assert abs(p_fit - p0) < 0.2, f"p_fit={p_fit}, p0={p0}"
        assert abs(q_fit - q0) < 0.2, f"q_fit={q_fit}, q0={q0}"
        
        # 验证系数接近真实值
        assert abs(b0_fit - b0_true) < 0.2
        assert abs(b1_fit - b1_true) < 0.2
        assert abs(b2_fit - b2_true) < 0.2
        
        # 验证 R2 较高
        assert r2_holdout > 0.7, f"r2_holdout={r2_holdout}"
    
    def test_grid_search_with_different_ranges(self):
        """测试不同搜索范围的网格搜索。"""
        np.random.seed(42)
        
        # 生成简单的合成数据
        n_samples = 200
        sic = np.random.uniform(0.1, 1.0, n_samples)
        wave_swh = np.random.uniform(0.5, 4.0, n_samples)
        
        x1 = sic ** 1.2
        x2 = wave_swh ** 1.8
        y = 0.5 - 0.4 * x1 - 0.2 * x2 + np.random.normal(0, 0.05, n_samples)
        
        df = pd.DataFrame({
            'sic': sic,
            'wave_swh': wave_swh,
            'y': y,
        })
        
        # 运行网格搜索
        p_fit, q_fit, _, _, _, _, _, r2_holdout = grid_search_fit(
            df,
            p_min=0.8,
            p_max=2.0,
            q_min=1.0,
            q_max=2.5,
            coarse_step=0.2,
            fine_step=0.1,
            holdout_ratio=0.2,
        )
        
        # 验证拟合结果在搜索范围内
        assert 0.8 <= p_fit <= 2.0
        assert 1.0 <= q_fit <= 2.5
        
        # 验证 R2 有效
        assert 0 <= r2_holdout <= 1


class TestEdgeCases:
    """测试边界情况。"""
    
    def test_empty_dataframe(self):
        """测试空 DataFrame。"""
        df = pd.DataFrame({
            'sic': [],
            'wave_swh': [],
            'y': [],
        })
        
        # 应该处理空数据而不崩溃
        try:
            p_fit, q_fit, _, _, _, _, _, _ = grid_search_fit(
                df,
                p_min=1.0,
                p_max=2.0,
                q_min=1.0,
                q_max=2.0,
                coarse_step=0.2,
                fine_step=0.05,
            )
        except Exception as e:
            # 允许抛出异常，但应该是有意义的
            assert "empty" in str(e).lower() or "no data" in str(e).lower()
    
    def test_nan_values(self):
        """测试包含 NaN 值的数据。"""
        np.random.seed(42)
        
        sic = np.array([0.5, 0.6, np.nan, 0.8, 0.9])
        wave_swh = np.array([1.0, np.nan, 1.5, 2.0, 2.5])
        y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        df = pd.DataFrame({
            'sic': sic,
            'wave_swh': wave_swh,
            'y': y,
        })
        
        # 应该自动移除 NaN 行
        p_fit, q_fit, _, _, _, _, _, _ = grid_search_fit(
            df,
            p_min=1.0,
            p_max=2.0,
            q_min=1.0,
            q_max=2.0,
            coarse_step=0.2,
            fine_step=0.05,
        )
        
        # 应该成功拟合（使用非 NaN 行）
        assert not np.isnan(p_fit)
        assert not np.isnan(q_fit)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

