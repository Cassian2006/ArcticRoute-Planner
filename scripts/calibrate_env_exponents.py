#!/usr/bin/env python3
"""
环境指数参数校准脚本。

目标：通过网格搜索和 logistic 回归，为海冰浓度 (sic) 和波浪高度 (wave_swh) 的指数参数 (p, q)
找到最优值，并估计置信区间。

流程：
  1. 构造训练样本（cell-level 二分类）
     - 正样本：AIS 轨迹经过的格点
     - 负样本：同月份同区域随机采样海上格点（数量=正样本的 2~5 倍）
  
  2. 特征工程
     - sic（海冰浓度）
     - wave_swh（波浪高度）
     - ice_thickness（可选）
     - ais_density（可选）
     - lat, lon（地理位置）
  
  3. 网格搜索
     - p ∈ [0.5, 3.0] 步长 0.1（sic 指数）
     - q ∈ [0.5, 3.0] 步长 0.1（wave_swh 指数）
     - 对每组 (p,q) 拟合 logistic 回归
     - 评价指标：AUC、logloss、分区稳定性（空间 CV）
  
  4. Bootstrap 置信区间
     - 对最优 (p,q) 进行 200 次重采样
     - 估计 95% 置信区间
  
  5. 输出报告
     - reports/exponent_fit_results.csv：最优参数与置信区间
     - reports/exponent_fit_report.md：详细分析报告

用法：
  python scripts/calibrate_env_exponents.py --ym 202412 --grid-mode real --sample-n 200000
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, log_loss
    from sklearn.model_selection import KFold
except ImportError:
    raise ImportError("scikit-learn is required for calibrate_env_exponents.py")

# 项目导入
try:
    from arcticroute.core.grid import Grid2D, make_demo_grid
    from arcticroute.core.landmask import load_landmask_for_grid
    from arcticroute.core.env_real import load_real_env_for_grid
except ImportError:
    pass

try:
    from arcticroute.data.ais_io import load_ais_trajectories_for_month
except ImportError:
    load_ais_trajectories_for_month = None

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class ExponentGridSearchResult:
    """单个 (p, q) 组合的搜索结果。"""
    p: float
    q: float
    auc: float
    logloss: float
    spatial_cv_auc: float  # 空间分块 CV 的 AUC 均值
    spatial_cv_std: float  # 空间分块 CV 的 AUC 标准差
    n_samples: int
    n_positive: int
    n_negative: int


@dataclass
class ExponentCalibrationResult:
    """校准结果汇总。"""
    optimal_p: float
    optimal_q: float
    optimal_auc: float
    optimal_logloss: float
    p_ci_lower: float
    p_ci_upper: float
    q_ci_lower: float
    q_ci_upper: float
    bootstrap_n: int
    grid_search_results: List[ExponentGridSearchResult]
    timestamp: str


# ============================================================================
# 样本构造
# ============================================================================

def construct_training_samples(
    grid: Grid2D,
    land_mask: np.ndarray,
    ym: str,
    ais_trajectories: Optional[List[List[Tuple[float, float]]]] = None,
    sample_n: int = 200000,
    negative_ratio: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造 cell-level 二分类训练样本。
    
    正样本：AIS 轨迹经过的格点
    负样本：同月份同区域随机采样海上格点
    
    Args:
        grid: Grid2D 对象
        land_mask: bool 数组，True = 陆地
        ym: 年月字符串（例如 "202412"）
        ais_trajectories: 可选的 AIS 轨迹列表（每条轨迹是 [(lat, lon), ...] 列表）
        sample_n: 目标样本数量
        negative_ratio: 负样本与正样本的比例（默认 3.0）
    
    Returns:
        (indices, labels) 元组
        - indices: shape (n_samples, 2)，每行是 (i, j) 网格索引
        - labels: shape (n_samples,)，0/1 标签
    """
    ny, nx = grid.shape()
    
    # 获取海洋格点索引
    ocean_mask = ~land_mask
    ocean_indices = np.argwhere(ocean_mask)  # shape (n_ocean, 2)
    
    logger.info(f"Ocean cells: {len(ocean_indices)} / {ny * nx}")
    
    # 构造正样本：AIS 轨迹经过的格点
    positive_indices = set()
    
    if ais_trajectories is not None and len(ais_trajectories) > 0:
        for traj in ais_trajectories:
            for lat, lon in traj:
                # 找到最近的网格点
                lat_idx = np.argmin(np.abs(grid.lat2d[:, 0] - lat))
                lon_idx = np.argmin(np.abs(grid.lon2d[0, :] - lon))
                
                # 检查是否是海洋格点
                if ocean_mask[lat_idx, lon_idx]:
                    positive_indices.add((lat_idx, lon_idx))
    
    positive_indices = np.array(list(positive_indices), dtype=int)
    n_positive = len(positive_indices)
    
    logger.info(f"Positive samples (AIS-visited cells): {n_positive}")
    
    # 构造负样本：随机采样海上格点
    n_negative = max(int(n_positive * negative_ratio), 100)
    n_negative = min(n_negative, len(ocean_indices) - n_positive)
    
    # 从海洋格点中排除正样本，再随机采样负样本
    ocean_set = set(map(tuple, ocean_indices))
    positive_set = set(map(tuple, positive_indices))
    negative_candidates = list(ocean_set - positive_set)
    
    if len(negative_candidates) < n_negative:
        logger.warning(
            f"Not enough negative candidates: {len(negative_candidates)} < {n_negative}, "
            f"using all available"
        )
        n_negative = len(negative_candidates)
    
    negative_indices = np.array(
        list(np.random.choice(len(negative_candidates), n_negative, replace=False)),
        dtype=int
    )
    negative_indices = np.array([negative_candidates[i] for i in negative_indices], dtype=int)
    
    logger.info(f"Negative samples (random ocean cells): {n_negative}")
    
    # 合并样本
    all_indices = np.vstack([positive_indices, negative_indices])
    all_labels = np.hstack([
        np.ones(n_positive, dtype=int),
        np.zeros(n_negative, dtype=int),
    ])
    
    # 打乱顺序
    perm = np.random.permutation(len(all_labels))
    all_indices = all_indices[perm]
    all_labels = all_labels[perm]
    
    logger.info(f"Total training samples: {len(all_labels)} (positive: {n_positive}, negative: {n_negative})")
    
    return all_indices, all_labels


# ============================================================================
# 特征工程
# ============================================================================

def extract_features(
    grid: Grid2D,
    env,
    indices: np.ndarray,
    include_ice_thickness: bool = False,
    include_ais_density: bool = False,
) -> np.ndarray:
    """
    从网格和环境数据中提取特征。
    
    特征列表：
      0. sic（海冰浓度，0..1）
      1. wave_swh（波浪高度，m）
      2. ice_thickness（可选，m）
      3. ais_density（可选，0..1）
      4. lat（纬度，度）
      5. lon（经度，度）
    
    Args:
        grid: Grid2D 对象
        env: RealEnvLayers 对象
        indices: shape (n_samples, 2)，网格索引
        include_ice_thickness: 是否包含冰厚特征
        include_ais_density: 是否包含 AIS 密度特征
    
    Returns:
        特征数组，shape (n_samples, n_features)
    """
    n_samples = len(indices)
    features_list = []
    
    # 特征 0: sic
    if env.sic is not None:
        sic_vals = env.sic[indices[:, 0], indices[:, 1]]
    else:
        sic_vals = np.zeros(n_samples)
    features_list.append(sic_vals)
    
    # 特征 1: wave_swh
    if env.wave_swh is not None:
        wave_vals = env.wave_swh[indices[:, 0], indices[:, 1]]
    else:
        wave_vals = np.zeros(n_samples)
    features_list.append(wave_vals)
    
    # 特征 2: ice_thickness（可选）
    if include_ice_thickness and env.ice_thickness_m is not None:
        ice_thick_vals = env.ice_thickness_m[indices[:, 0], indices[:, 1]]
        features_list.append(ice_thick_vals)
    
    # 特征 3: ais_density（可选）
    if include_ais_density:
        # TODO: 加载 AIS 密度
        ais_density_vals = np.zeros(n_samples)
        features_list.append(ais_density_vals)
    
    # 特征 4: lat
    lat_vals = grid.lat2d[indices[:, 0], indices[:, 1]]
    features_list.append(lat_vals)
    
    # 特征 5: lon
    lon_vals = grid.lon2d[indices[:, 0], indices[:, 1]]
    features_list.append(lon_vals)
    
    features = np.column_stack(features_list)
    
    logger.info(f"Extracted features: shape {features.shape}")
    
    return features


def apply_exponent_transform(
    features: np.ndarray,
    p: float,
    q: float,
) -> np.ndarray:
    """
    对特征应用指数变换。
    
    变换规则：
      - sic (feature 0) -> sic^p
      - wave_swh (feature 1) -> wave_swh^q
      - 其他特征保持不变
    
    Args:
        features: 原始特征数组，shape (n_samples, n_features)
        p: sic 指数
        q: wave_swh 指数
    
    Returns:
        变换后的特征数组，shape (n_samples, n_features)
    """
    transformed = features.copy()
    
    # 应用 sic 指数
    transformed[:, 0] = np.power(np.clip(features[:, 0], 0.0, 1.0), p)
    
    # 应用 wave_swh 指数
    transformed[:, 1] = np.power(np.clip(features[:, 1], 0.0, 10.0), q)
    
    return transformed


# ============================================================================
# 网格搜索与评估
# ============================================================================

def evaluate_exponents(
    features: np.ndarray,
    labels: np.ndarray,
    p: float,
    q: float,
    n_splits: int = 5,
) -> Tuple[float, float, float, float]:
    """
    评估单个 (p, q) 组合的性能。
    
    使用 K-Fold 交叉验证（空间分块 CV）计算：
      - AUC（全局）
      - LogLoss（全局）
      - 空间 CV AUC 均值
      - 空间 CV AUC 标准差
    
    Args:
        features: 特征数组，shape (n_samples, n_features)
        labels: 标签数组，shape (n_samples,)
        p: sic 指数
        q: wave_swh 指数
        n_splits: K-Fold 分割数
    
    Returns:
        (auc, logloss, spatial_cv_auc_mean, spatial_cv_auc_std) 元组
    """
    # 应用指数变换
    transformed_features = apply_exponent_transform(features, p, q)
    
    # 全局评估：用全部数据训练和评估
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(transformed_features, labels)
    
    y_pred_proba = lr.predict_proba(transformed_features)[:, 1]
    auc = roc_auc_score(labels, y_pred_proba)
    logloss = log_loss(labels, y_pred_proba)
    
    # 空间分块 CV：按纬度分块，避免空间泄漏
    lat_vals = features[:, -2]  # 倒数第二列是 lat
    lat_quantiles = np.quantile(lat_vals, np.linspace(0, 1, n_splits + 1))
    
    cv_aucs = []
    for i in range(n_splits):
        lat_min = lat_quantiles[i]
        lat_max = lat_quantiles[i + 1]
        
        # 训练集：其他分块
        train_mask = (lat_vals < lat_min) | (lat_vals > lat_max)
        # 测试集：当前分块
        test_mask = (lat_vals >= lat_min) & (lat_vals <= lat_max)
        
        if np.sum(test_mask) == 0:
            continue
        
        X_train = transformed_features[train_mask]
        y_train = labels[train_mask]
        X_test = transformed_features[test_mask]
        y_test = labels[test_mask]
        
        # 训练 logistic 回归
        lr_cv = LogisticRegression(max_iter=1000, random_state=42)
        lr_cv.fit(X_train, y_train)
        
        # 评估
        y_pred_proba_cv = lr_cv.predict_proba(X_test)[:, 1]
        auc_cv = roc_auc_score(y_test, y_pred_proba_cv)
        cv_aucs.append(auc_cv)
    
    spatial_cv_auc_mean = np.mean(cv_aucs) if cv_aucs else auc
    spatial_cv_auc_std = np.std(cv_aucs) if cv_aucs else 0.0
    
    return auc, logloss, spatial_cv_auc_mean, spatial_cv_auc_std


def grid_search_exponents(
    features: np.ndarray,
    labels: np.ndarray,
    p_range: Tuple[float, float] = (0.5, 3.0),
    q_range: Tuple[float, float] = (0.5, 3.0),
    step: float = 0.1,
) -> Tuple[ExponentGridSearchResult, List[ExponentGridSearchResult]]:
    """
    对 (p, q) 进行网格搜索。
    
    Args:
        features: 特征数组，shape (n_samples, n_features)
        labels: 标签数组，shape (n_samples,)
        p_range: p 的搜索范围
        q_range: q 的搜索范围
        step: 搜索步长
    
    Returns:
        (best_result, all_results) 元组
    """
    p_values = np.arange(p_range[0], p_range[1] + step / 2, step)
    q_values = np.arange(q_range[0], q_range[1] + step / 2, step)
    
    logger.info(f"Grid search: p ∈ [{p_range[0]}, {p_range[1]}], q ∈ [{q_range[0]}, {q_range[1]}]")
    logger.info(f"Grid size: {len(p_values)} × {len(q_values)} = {len(p_values) * len(q_values)}")
    
    all_results = []
    best_result = None
    best_score = -np.inf
    
    for i, p in enumerate(p_values):
        for j, q in enumerate(q_values):
            logger.info(f"Evaluating ({p:.1f}, {q:.1f}) [{i*len(q_values)+j+1}/{len(p_values)*len(q_values)}]")
            
            auc, logloss, spatial_cv_auc, spatial_cv_std = evaluate_exponents(
                features, labels, p, q
            )
            
            result = ExponentGridSearchResult(
                p=p,
                q=q,
                auc=auc,
                logloss=logloss,
                spatial_cv_auc=spatial_cv_auc,
                spatial_cv_std=spatial_cv_std,
                n_samples=len(labels),
                n_positive=int(np.sum(labels)),
                n_negative=int(np.sum(1 - labels)),
            )
            all_results.append(result)
            
            # 使用 AUC 作为主要评价指标
            score = auc
            if score > best_score:
                best_score = score
                best_result = result
            
            logger.info(
                f"  AUC={auc:.4f}, LogLoss={logloss:.4f}, "
                f"Spatial CV AUC={spatial_cv_auc:.4f}±{spatial_cv_std:.4f}"
            )
    
    logger.info(f"Best result: p={best_result.p:.1f}, q={best_result.q:.1f}, AUC={best_result.auc:.4f}")
    
    return best_result, all_results


# ============================================================================
# Bootstrap 置信区间
# ============================================================================

def bootstrap_confidence_intervals(
    features: np.ndarray,
    labels: np.ndarray,
    p_init: float,
    q_init: float,
    n_bootstrap: int = 200,
    ci: float = 0.95,
) -> Tuple[float, float, float, float]:
    """
    通过 bootstrap 重采样估计 (p, q) 的置信区间。
    
    流程：
      1. 对原始样本进行 n_bootstrap 次有放回重采样
      2. 对每次重采样进行网格搜索（简化版，仅搜索 p, q 周围的小范围）
      3. 收集最优 p 和 q 的分布
      4. 计算置信区间
    
    Args:
        features: 特征数组，shape (n_samples, n_features)
        labels: 标签数组，shape (n_samples,)
        p_init: 初始 p 值
        q_init: 初始 q 值
        n_bootstrap: bootstrap 重采样次数
        ci: 置信水平（默认 0.95）
    
    Returns:
        (p_ci_lower, p_ci_upper, q_ci_lower, q_ci_upper) 元组
    """
    logger.info(f"Bootstrap confidence intervals: n_bootstrap={n_bootstrap}, ci={ci}")
    
    p_values = []
    q_values = []
    
    n_samples = len(labels)
    
    for boot_idx in range(n_bootstrap):
        # 有放回重采样
        indices = np.random.choice(n_samples, n_samples, replace=True)
        features_boot = features[indices]
        labels_boot = labels[indices]
        
        # 简化的网格搜索：仅搜索初始值周围的小范围
        p_range = (max(0.5, p_init - 0.3), min(3.0, p_init + 0.3))
        q_range = (max(0.5, q_init - 0.3), min(3.0, q_init + 0.3))
        
        best_result, _ = grid_search_exponents(
            features_boot, labels_boot,
            p_range=p_range,
            q_range=q_range,
            step=0.1,
        )
        
        p_values.append(best_result.p)
        q_values.append(best_result.q)
        
        if (boot_idx + 1) % 50 == 0:
            logger.info(f"Bootstrap iteration {boot_idx + 1}/{n_bootstrap}")
    
    # 计算置信区间
    alpha = 1.0 - ci
    p_ci_lower = np.percentile(p_values, alpha / 2 * 100)
    p_ci_upper = np.percentile(p_values, (1 - alpha / 2) * 100)
    q_ci_lower = np.percentile(q_values, alpha / 2 * 100)
    q_ci_upper = np.percentile(q_values, (1 - alpha / 2) * 100)
    
    logger.info(f"p CI: [{p_ci_lower:.3f}, {p_ci_upper:.3f}]")
    logger.info(f"q CI: [{q_ci_lower:.3f}, {q_ci_upper:.3f}]")
    
    return p_ci_lower, p_ci_upper, q_ci_lower, q_ci_upper


# ============================================================================
# 报告生成
# ============================================================================

def save_results_csv(
    result: ExponentCalibrationResult,
    output_path: Path,
) -> None:
    """保存结果到 CSV 文件。"""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        # 写入汇总信息
        writer.writerow(["Exponent Calibration Results"])
        writer.writerow(["Timestamp", result.timestamp])
        writer.writerow([])
        
        # 写入最优参数
        writer.writerow(["Optimal Parameters"])
        writer.writerow(["Parameter", "Value", "95% CI Lower", "95% CI Upper"])
        writer.writerow(["p (sic exponent)", f"{result.optimal_p:.3f}", f"{result.p_ci_lower:.3f}", f"{result.p_ci_upper:.3f}"])
        writer.writerow(["q (wave_swh exponent)", f"{result.optimal_q:.3f}", f"{result.q_ci_lower:.3f}", f"{result.q_ci_upper:.3f}"])
        writer.writerow([])
        
        # 写入性能指标
        writer.writerow(["Performance Metrics"])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["AUC", f"{result.optimal_auc:.4f}"])
        writer.writerow(["LogLoss", f"{result.optimal_logloss:.4f}"])
        writer.writerow(["Bootstrap Iterations", result.bootstrap_n])
        writer.writerow([])
        
        # 写入网格搜索结果
        writer.writerow(["Grid Search Results (Top 20)"])
        writer.writerow(["p", "q", "AUC", "LogLoss", "Spatial CV AUC", "Spatial CV Std"])
        
        # 按 AUC 排序
        sorted_results = sorted(result.grid_search_results, key=lambda x: x.auc, reverse=True)
        for res in sorted_results[:20]:
            writer.writerow([
                f"{res.p:.1f}",
                f"{res.q:.1f}",
                f"{res.auc:.4f}",
                f"{res.logloss:.4f}",
                f"{res.spatial_cv_auc:.4f}",
                f"{res.spatial_cv_std:.4f}",
            ])
    
    logger.info(f"Results saved to {output_path}")


def save_report_markdown(
    result: ExponentCalibrationResult,
    output_path: Path,
) -> None:
    """保存详细报告到 Markdown 文件。"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 环境指数参数校准报告\n\n")
        
        f.write(f"**生成时间**: {result.timestamp}\n\n")
        
        # 摘要
        f.write("## 摘要\n\n")
        f.write(
            f"通过网格搜索和 logistic 回归，为海冰浓度 (sic) 和波浪高度 (wave_swh) 的指数参数找到最优值。\n\n"
        )
        
        # 最优参数
        f.write("## 最优参数\n\n")
        f.write(f"| 参数 | 最优值 | 95% 置信区间 |\n")
        f.write(f"|------|--------|---------------|\n")
        f.write(f"| p (sic 指数) | {result.optimal_p:.3f} | [{result.p_ci_lower:.3f}, {result.p_ci_upper:.3f}] |\n")
        f.write(f"| q (wave_swh 指数) | {result.optimal_q:.3f} | [{result.q_ci_lower:.3f}, {result.q_ci_upper:.3f}] |\n")
        f.write("\n")
        
        # 性能指标
        f.write("## 性能指标\n\n")
        f.write(f"- **AUC**: {result.optimal_auc:.4f}\n")
        f.write(f"- **LogLoss**: {result.optimal_logloss:.4f}\n")
        f.write(f"- **Bootstrap 迭代数**: {result.bootstrap_n}\n")
        f.write("\n")
        
        # 网格搜索结果
        f.write("## 网格搜索结果（Top 10）\n\n")
        f.write(f"| p | q | AUC | LogLoss | Spatial CV AUC | Spatial CV Std |\n")
        f.write(f"|---|---|-----|---------|----------------|----------------|\n")
        
        sorted_results = sorted(result.grid_search_results, key=lambda x: x.auc, reverse=True)
        for res in sorted_results[:10]:
            f.write(
                f"| {res.p:.1f} | {res.q:.1f} | {res.auc:.4f} | {res.logloss:.4f} | "
                f"{res.spatial_cv_auc:.4f} | {res.spatial_cv_std:.4f} |\n"
            )
        f.write("\n")
        
        # 方法说明
        f.write("## 方法\n\n")
        f.write(
            "1. **样本构造**：正样本为 AIS 轨迹经过的格点，负样本为随机采样的海上格点（比例 3:1）。\n"
        )
        f.write(
            "2. **特征工程**：提取 sic、wave_swh、ice_thickness、lat、lon 等特征。\n"
        )
        f.write(
            "3. **网格搜索**：对 p ∈ [0.5, 3.0]、q ∈ [0.5, 3.0]（步长 0.1）进行搜索，"
            "对每组 (p,q) 拟合 logistic 回归。\n"
        )
        f.write(
            "4. **评价指标**：使用 AUC 和 LogLoss 评估模型性能，使用空间分块 CV 避免空间泄漏。\n"
        )
        f.write(
            "5. **Bootstrap 置信区间**：对最优 (p,q) 进行 200 次重采样，估计 95% 置信区间。\n"
        )
        f.write("\n")
        
        # 建议
        f.write("## 建议\n\n")
        f.write(
            f"建议在配置中使用以下参数：\n\n"
            f"```yaml\n"
            f"sic_exp: {result.optimal_p:.3f}  # 海冰浓度指数\n"
            f"wave_exp: {result.optimal_q:.3f}  # 波浪高度指数\n"
            f"```\n\n"
        )
    
    logger.info(f"Report saved to {output_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数。"""
    parser = argparse.ArgumentParser(
        description="环境指数参数校准脚本"
    )
    parser.add_argument(
        "--ym",
        type=str,
        default="202412",
        help="年月，格式 YYYYMM（默认 202412）",
    )
    parser.add_argument(
        "--grid-mode",
        type=str,
        choices=["demo", "real"],
        default="real",
        help="网格模式（默认 real）",
    )
    parser.add_argument(
        "--ais-density",
        type=str,
        default="auto",
        help="AIS 密度文件路径或 'auto'（默认 auto）",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=200000,
        help="目标样本数量（默认 200000）",
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=200,
        help="Bootstrap 重采样次数（默认 200）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="输出目录（默认 reports）",
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting exponent calibration: ym={args.ym}, grid_mode={args.grid_mode}")
    
    # 加载网格和陆地掩码
    if args.grid_mode == "demo":
        grid, _ = make_demo_grid()
    else:
        # TODO: 加载真实网格
        grid, _ = make_demo_grid()
    
    land_mask = load_landmask_for_grid(grid)
    
    logger.info(f"Grid shape: {grid.shape()}")
    
    # 加载环境数据
    env = load_real_env_for_grid(ym=args.ym)
    if env is None:
        logger.error(f"Failed to load environment data for {args.ym}")
        return
    
    logger.info(f"Environment data loaded: sic={env.sic is not None}, wave={env.wave_swh is not None}")
    
    # 加载 AIS 轨迹
    ais_trajectories = None
    try:
        ais_trajectories = load_ais_trajectories_for_month(args.ym)
        logger.info(f"Loaded {len(ais_trajectories) if ais_trajectories else 0} AIS trajectories")
    except Exception as e:
        logger.warning(f"Failed to load AIS trajectories: {e}")
    
    # 构造训练样本
    indices, labels = construct_training_samples(
        grid,
        land_mask,
        args.ym,
        ais_trajectories=ais_trajectories,
        sample_n=args.sample_n,
    )
    
    # 提取特征
    features = extract_features(grid, env, indices)
    
    # 网格搜索
    best_result, all_results = grid_search_exponents(features, labels)
    
    # Bootstrap 置信区间
    p_ci_lower, p_ci_upper, q_ci_lower, q_ci_upper = bootstrap_confidence_intervals(
        features, labels,
        best_result.p, best_result.q,
        n_bootstrap=args.bootstrap_n,
    )
    
    # 构造结果对象
    calibration_result = ExponentCalibrationResult(
        optimal_p=best_result.p,
        optimal_q=best_result.q,
        optimal_auc=best_result.auc,
        optimal_logloss=best_result.logloss,
        p_ci_lower=p_ci_lower,
        p_ci_upper=p_ci_upper,
        q_ci_lower=q_ci_lower,
        q_ci_upper=q_ci_upper,
        bootstrap_n=args.bootstrap_n,
        grid_search_results=all_results,
        timestamp=datetime.now().isoformat(),
    )
    
    # 保存结果
    csv_path = args.output_dir / "exponent_fit_results.csv"
    md_path = args.output_dir / "exponent_fit_report.md"
    
    save_results_csv(calibration_result, csv_path)
    save_report_markdown(calibration_result, md_path)
    
    logger.info(f"Calibration complete!")
    logger.info(f"Results: p={calibration_result.optimal_p:.3f}, q={calibration_result.optimal_q:.3f}")
    logger.info(f"AUC={calibration_result.optimal_auc:.4f}, LogLoss={calibration_result.optimal_logloss:.4f}")


if __name__ == "__main__":
    main()

