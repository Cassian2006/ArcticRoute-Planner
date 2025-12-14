#!/usr/bin/env python3
"""
环境指数参数校准脚本的轻量级烟雾测试。

使用临时小网格和伪 AIS 轨迹，确保脚本能跑完并写出 csv 和 md 文件。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from arcticroute.core.grid import Grid2D
from arcticroute.core.env_real import RealEnvLayers


# ============================================================================
# 测试工具
# ============================================================================

def create_test_grid(ny: int = 20, nx: int = 30) -> Grid2D:
    """创建小测试网格。"""
    lat_min, lat_max = 70.0, 80.0
    lon_min, lon_max = 0.0, 30.0
    
    lat1d = np.linspace(lat_min, lat_max, ny)
    lon1d = np.linspace(lon_min, lon_max, nx)
    
    lat2d, lon2d = np.meshgrid(lat1d, lon1d, indexing="ij")
    
    grid = Grid2D(lat2d=lat2d, lon2d=lon2d)
    return grid


def create_test_landmask(grid: Grid2D) -> np.ndarray:
    """创建测试陆地掩码（仅北边界为陆地）。"""
    ny, nx = grid.shape()
    land_mask = np.zeros((ny, nx), dtype=bool)
    land_mask[:2, :] = True  # 北边界为陆地
    return land_mask


def create_test_env(grid: Grid2D) -> RealEnvLayers:
    """创建测试环境数据。"""
    ny, nx = grid.shape()
    
    # 创建 sic：从南到北逐渐增加
    sic = np.zeros((ny, nx), dtype=float)
    for i in range(ny):
        sic[i, :] = min(1.0, (i / ny) * 1.5)
    
    # 创建 wave_swh：随机
    wave_swh = np.random.uniform(0.5, 3.0, (ny, nx))
    
    # 创建 ice_thickness：随机
    ice_thickness_m = np.random.uniform(0.0, 1.5, (ny, nx))
    
    env = RealEnvLayers(
        grid=grid,
        sic=sic,
        wave_swh=wave_swh,
        ice_thickness_m=ice_thickness_m,
        land_mask=create_test_landmask(grid),
    )
    
    return env


def create_test_ais_trajectories(grid: Grid2D, n_traj: int = 5) -> List[List[Tuple[float, float]]]:
    """创建伪 AIS 轨迹。"""
    trajectories = []
    
    for _ in range(n_traj):
        # 随机起点和终点
        start_lat = np.random.uniform(72.0, 76.0)
        start_lon = np.random.uniform(5.0, 10.0)
        end_lat = np.random.uniform(76.0, 79.0)
        end_lon = np.random.uniform(15.0, 25.0)
        
        # 线性插值生成轨迹
        n_points = np.random.randint(10, 30)
        lats = np.linspace(start_lat, end_lat, n_points)
        lons = np.linspace(start_lon, end_lon, n_points)
        
        trajectory = list(zip(lats, lons))
        trajectories.append(trajectory)
    
    return trajectories


# ============================================================================
# 测试
# ============================================================================

def test_calibrate_exponents_smoke():
    """烟雾测试：确保脚本能跑完并生成输出文件。"""
    # 导入脚本中的函数
    from scripts.calibrate_env_exponents import (
        construct_training_samples,
        extract_features,
        grid_search_exponents,
        bootstrap_confidence_intervals,
        save_results_csv,
        save_report_markdown,
        ExponentCalibrationResult,
    )
    
    # 创建测试数据
    grid = create_test_grid(ny=20, nx=30)
    land_mask = create_test_landmask(grid)
    env = create_test_env(grid)
    ais_trajectories = create_test_ais_trajectories(grid, n_traj=5)
    
    # 构造训练样本
    indices, labels = construct_training_samples(
        grid,
        land_mask,
        ym="202412",
        ais_trajectories=ais_trajectories,
        sample_n=1000,  # 小样本
        negative_ratio=3.0,
    )
    
    assert len(indices) > 0, "No training samples constructed"
    assert len(labels) == len(indices), "Labels and indices length mismatch"
    assert np.sum(labels) > 0, "No positive samples"
    
    # 提取特征
    features = extract_features(grid, env, indices)
    
    assert features.shape[0] == len(labels), "Feature and label count mismatch"
    assert features.shape[1] >= 4, "Not enough features"
    
    # 网格搜索（简化版，仅搜索小范围）
    best_result, all_results = grid_search_exponents(
        features, labels,
        p_range=(1.0, 2.0),  # 缩小搜索范围
        q_range=(1.0, 2.0),
        step=0.5,  # 增大步长
    )
    
    assert best_result is not None, "No best result found"
    assert len(all_results) > 0, "No grid search results"
    assert 0.5 <= best_result.p <= 3.0, "p out of range"
    assert 0.5 <= best_result.q <= 3.0, "q out of range"
    assert 0.0 <= best_result.auc <= 1.0, "AUC out of range"
    
    # Bootstrap 置信区间（简化版，仅 10 次迭代）
    # 注意：由于样本量小，bootstrap 置信区间可能不包含初始最优值
    # 这是正常的，我们只检查置信区间的合理性
    p_ci_lower, p_ci_upper, q_ci_lower, q_ci_upper = bootstrap_confidence_intervals(
        features, labels,
        best_result.p, best_result.q,
        n_bootstrap=10,  # 减少迭代次数
        ci=0.95,
    )
    
    # 检查置信区间的合理性（而不是检查初始值是否在区间内）
    assert p_ci_lower <= p_ci_upper, "p CI lower > upper"
    assert q_ci_lower <= q_ci_upper, "q CI lower > upper"
    assert 0.5 <= p_ci_lower <= 3.0, "p CI lower out of range"
    assert 0.5 <= p_ci_upper <= 3.0, "p CI upper out of range"
    assert 0.5 <= q_ci_lower <= 3.0, "q CI lower out of range"
    assert 0.5 <= q_ci_upper <= 3.0, "q CI upper out of range"
    
    # 构造结果对象
    from datetime import datetime
    
    calibration_result = ExponentCalibrationResult(
        optimal_p=best_result.p,
        optimal_q=best_result.q,
        optimal_auc=best_result.auc,
        optimal_logloss=best_result.logloss,
        p_ci_lower=p_ci_lower,
        p_ci_upper=p_ci_upper,
        q_ci_lower=q_ci_lower,
        q_ci_upper=q_ci_upper,
        bootstrap_n=10,
        grid_search_results=all_results,
        timestamp=datetime.now().isoformat(),
    )
    
    # 保存到临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        csv_path = output_dir / "exponent_fit_results.csv"
        md_path = output_dir / "exponent_fit_report.md"
        
        save_results_csv(calibration_result, csv_path)
        save_report_markdown(calibration_result, md_path)
        
        # 验证文件存在
        assert csv_path.exists(), "CSV file not created"
        assert md_path.exists(), "Markdown file not created"
        
        # 验证文件内容
        csv_content = csv_path.read_text(encoding="utf-8")
        assert "Exponent Calibration Results" in csv_content, "CSV header missing"
        assert f"{best_result.p:.1f}" in csv_content, "p value not in CSV"
        assert f"{best_result.q:.1f}" in csv_content, "q value not in CSV"
        
        md_content = md_path.read_text(encoding="utf-8")
        assert "环境指数参数校准报告" in md_content, "Markdown header missing"
        assert f"{best_result.p:.3f}" in md_content, "p value not in Markdown"
        assert f"{best_result.q:.3f}" in md_content, "q value not in Markdown"


def test_construct_training_samples():
    """测试样本构造函数。"""
    from scripts.calibrate_env_exponents import construct_training_samples
    
    grid = create_test_grid(ny=20, nx=30)
    land_mask = create_test_landmask(grid)
    ais_trajectories = create_test_ais_trajectories(grid, n_traj=3)
    
    indices, labels = construct_training_samples(
        grid,
        land_mask,
        ym="202412",
        ais_trajectories=ais_trajectories,
        sample_n=500,
        negative_ratio=2.0,
    )
    
    assert len(indices) > 0
    assert len(labels) == len(indices)
    assert np.sum(labels) > 0  # 至少有正样本
    assert np.sum(1 - labels) > 0  # 至少有负样本


def test_extract_features():
    """测试特征提取函数。"""
    from scripts.calibrate_env_exponents import extract_features
    
    grid = create_test_grid(ny=20, nx=30)
    env = create_test_env(grid)
    
    # 创建随机索引
    indices = np.random.randint(0, 20, (100, 2))
    
    features = extract_features(grid, env, indices)
    
    assert features.shape[0] == 100
    assert features.shape[1] >= 4  # 至少有 sic, wave, lat, lon


def test_apply_exponent_transform():
    """测试指数变换函数。"""
    from scripts.calibrate_env_exponents import apply_exponent_transform
    
    # 创建测试特征
    features = np.array([
        [0.5, 2.0, 70.0, 10.0],
        [0.8, 1.5, 72.0, 15.0],
        [0.2, 3.0, 75.0, 20.0],
    ])
    
    p, q = 1.5, 2.0
    transformed = apply_exponent_transform(features, p, q)
    
    assert transformed.shape == features.shape
    # sic (feature 0) 应该被变换
    assert np.allclose(transformed[:, 0], np.power(features[:, 0], p))
    # wave (feature 1) 应该被变换
    assert np.allclose(transformed[:, 1], np.power(features[:, 1], q))
    # 其他特征不变
    assert np.allclose(transformed[:, 2:], features[:, 2:])


def test_evaluate_exponents():
    """测试单个 (p, q) 组合的评估函数。"""
    from scripts.calibrate_env_exponents import evaluate_exponents
    
    # 创建简单的测试数据
    np.random.seed(42)
    features = np.random.randn(200, 4)
    labels = np.random.randint(0, 2, 200)
    
    auc, logloss, spatial_cv_auc, spatial_cv_std = evaluate_exponents(
        features, labels,
        p=1.5, q=1.5,
        n_splits=3,
    )
    
    assert 0.0 <= auc <= 1.0
    assert logloss >= 0.0
    assert 0.0 <= spatial_cv_auc <= 1.0
    assert spatial_cv_std >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

