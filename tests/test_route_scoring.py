"""
路线评分与推荐系统的测试。

测试 compute_route_scores 函数的功能，包括：
- 归一化指标的计算
- 综合分数的加权计算
- 推荐路线的选择
"""

from __future__ import annotations

import numpy as np

from arcticroute.core.analysis import (
    RouteCostBreakdown,
    compute_route_scores,
)


def test_compute_route_scores_basic():
    """测试基本的路线评分功能。"""
    # 构造三条假路线的成本分解
    breakdowns = {
        "efficient": RouteCostBreakdown(
            total_cost=100.0,
            component_totals={
                "base_distance": 50.0,
                "ice_risk": 30.0,
                "edl_risk": 10.0,
                "edl_uncertainty_penalty": 10.0,
            },
            component_fractions={},
            s_km=[0.0, 50.0, 100.0],
            component_along_path={},
        ),
        "edl_safe": RouteCostBreakdown(
            total_cost=120.0,
            component_totals={
                "base_distance": 50.0,
                "ice_risk": 40.0,
                "edl_risk": 20.0,
                "edl_uncertainty_penalty": 10.0,
            },
            component_fractions={},
            s_km=[0.0, 50.0, 100.0],
            component_along_path={},
        ),
        "edl_robust": RouteCostBreakdown(
            total_cost=140.0,
            component_totals={
                "base_distance": 50.0,
                "ice_risk": 40.0,
                "edl_risk": 20.0,
                "edl_uncertainty_penalty": 30.0,
            },
            component_fractions={},
            s_km=[0.0, 50.0, 100.0],
            component_along_path={},
        ),
    }
    
    # 构造 ECO 数据
    eco_by_key = {
        "efficient": {"fuel_total_t": 10.0, "co2_total_t": 30.0},
        "edl_safe": {"fuel_total_t": 12.0, "co2_total_t": 36.0},
        "edl_robust": {"fuel_total_t": 12.0, "co2_total_t": 36.0},
    }
    
    # 调用 compute_route_scores
    scores = compute_route_scores(
        breakdowns=breakdowns,
        eco_by_key=eco_by_key,
        weight_risk=0.33,
        weight_uncertainty=0.33,
        weight_fuel=0.34,
    )
    
    # 验证返回了 3 个 RouteScore
    assert len(scores) == 3
    assert "efficient" in scores
    assert "edl_safe" in scores
    assert "edl_robust" in scores
    
    # 验证每个 RouteScore 都有必要的字段
    for key, score in scores.items():
        assert score.distance_km >= 0
        assert score.total_cost >= 0
        assert 0 <= score.norm_fuel <= 1
        assert 0 <= score.norm_edl_risk <= 1
        assert 0 <= score.norm_edl_uncertainty <= 1
        assert score.composite_score >= 0


def test_compute_route_scores_fuel_weight():
    """测试当燃油权重很大时，燃油最少的路线分数最低。"""
    # 构造三条路线，燃油消耗明显不同
    breakdowns = {
        "efficient": RouteCostBreakdown(
            total_cost=100.0,
            component_totals={
                "base_distance": 50.0,
                "ice_risk": 50.0,
                "edl_risk": 0.0,
                "edl_uncertainty_penalty": 0.0,
            },
            component_fractions={},
            s_km=[0.0, 100.0],
            component_along_path={},
        ),
        "edl_safe": RouteCostBreakdown(
            total_cost=100.0,
            component_totals={
                "base_distance": 50.0,
                "ice_risk": 50.0,
                "edl_risk": 0.0,
                "edl_uncertainty_penalty": 0.0,
            },
            component_fractions={},
            s_km=[0.0, 100.0],
            component_along_path={},
        ),
        "edl_robust": RouteCostBreakdown(
            total_cost=100.0,
            component_totals={
                "base_distance": 50.0,
                "ice_risk": 50.0,
                "edl_risk": 0.0,
                "edl_uncertainty_penalty": 0.0,
            },
            component_fractions={},
            s_km=[0.0, 100.0],
            component_along_path={},
        ),
    }
    
    eco_by_key = {
        "efficient": {"fuel_total_t": 5.0, "co2_total_t": 15.0},  # 最少燃油
        "edl_safe": {"fuel_total_t": 10.0, "co2_total_t": 30.0},
        "edl_robust": {"fuel_total_t": 15.0, "co2_total_t": 45.0},  # 最多燃油
    }
    
    # 设置很高的燃油权重
    scores = compute_route_scores(
        breakdowns=breakdowns,
        eco_by_key=eco_by_key,
        weight_risk=0.0,
        weight_uncertainty=0.0,
        weight_fuel=1.0,
    )
    
    # efficient 应该有最低的综合分数（燃油最少）
    efficient_score = scores["efficient"].composite_score
    edl_safe_score = scores["edl_safe"].composite_score
    edl_robust_score = scores["edl_robust"].composite_score
    
    assert efficient_score < edl_safe_score
    assert efficient_score < edl_robust_score


def test_compute_route_scores_risk_weight():
    """测试当风险权重很大时，EDL 风险最小的路线分数最低。"""
    # 构造三条路线，EDL 风险明显不同
    breakdowns = {
        "efficient": RouteCostBreakdown(
            total_cost=100.0,
            component_totals={
                "base_distance": 50.0,
                "ice_risk": 50.0,
                "edl_risk": 20.0,  # 高风险
                "edl_uncertainty_penalty": 0.0,
            },
            component_fractions={},
            s_km=[0.0, 100.0],
            component_along_path={},
        ),
        "edl_safe": RouteCostBreakdown(
            total_cost=100.0,
            component_totals={
                "base_distance": 50.0,
                "ice_risk": 50.0,
                "edl_risk": 10.0,  # 中等风险
                "edl_uncertainty_penalty": 0.0,
            },
            component_fractions={},
            s_km=[0.0, 100.0],
            component_along_path={},
        ),
        "edl_robust": RouteCostBreakdown(
            total_cost=100.0,
            component_totals={
                "base_distance": 50.0,
                "ice_risk": 50.0,
                "edl_risk": 0.0,  # 低风险
                "edl_uncertainty_penalty": 0.0,
            },
            component_fractions={},
            s_km=[0.0, 100.0],
            component_along_path={},
        ),
    }
    
    eco_by_key = {
        "efficient": {"fuel_total_t": 10.0, "co2_total_t": 30.0},
        "edl_safe": {"fuel_total_t": 10.0, "co2_total_t": 30.0},
        "edl_robust": {"fuel_total_t": 10.0, "co2_total_t": 30.0},
    }
    
    # 设置很高的风险权重
    scores = compute_route_scores(
        breakdowns=breakdowns,
        eco_by_key=eco_by_key,
        weight_risk=1.0,
        weight_uncertainty=0.0,
        weight_fuel=0.0,
    )
    
    # edl_robust 应该有最低的综合分数（风险最低）
    efficient_score = scores["efficient"].composite_score
    edl_safe_score = scores["edl_safe"].composite_score
    edl_robust_score = scores["edl_robust"].composite_score
    
    assert edl_robust_score < edl_safe_score
    assert edl_robust_score < efficient_score


def test_compute_route_scores_uncertainty_weight():
    """测试当不确定性权重很大时，不确定性最小的路线分数最低。"""
    # 构造三条路线，不确定性明显不同
    breakdowns = {
        "efficient": RouteCostBreakdown(
            total_cost=100.0,
            component_totals={
                "base_distance": 50.0,
                "ice_risk": 50.0,
                "edl_risk": 0.0,
                "edl_uncertainty_penalty": 20.0,  # 高不确定性
            },
            component_fractions={},
            s_km=[0.0, 100.0],
            component_along_path={},
        ),
        "edl_safe": RouteCostBreakdown(
            total_cost=100.0,
            component_totals={
                "base_distance": 50.0,
                "ice_risk": 50.0,
                "edl_risk": 0.0,
                "edl_uncertainty_penalty": 10.0,  # 中等不确定性
            },
            component_fractions={},
            s_km=[0.0, 100.0],
            component_along_path={},
        ),
        "edl_robust": RouteCostBreakdown(
            total_cost=100.0,
            component_totals={
                "base_distance": 50.0,
                "ice_risk": 50.0,
                "edl_risk": 0.0,
                "edl_uncertainty_penalty": 0.0,  # 低不确定性
            },
            component_fractions={},
            s_km=[0.0, 100.0],
            component_along_path={},
        ),
    }
    
    eco_by_key = {
        "efficient": {"fuel_total_t": 10.0, "co2_total_t": 30.0},
        "edl_safe": {"fuel_total_t": 10.0, "co2_total_t": 30.0},
        "edl_robust": {"fuel_total_t": 10.0, "co2_total_t": 30.0},
    }
    
    # 设置很高的不确定性权重
    scores = compute_route_scores(
        breakdowns=breakdowns,
        eco_by_key=eco_by_key,
        weight_risk=0.0,
        weight_uncertainty=1.0,
        weight_fuel=0.0,
    )
    
    # edl_robust 应该有最低的综合分数（不确定性最低）
    efficient_score = scores["efficient"].composite_score
    edl_safe_score = scores["edl_safe"].composite_score
    edl_robust_score = scores["edl_robust"].composite_score
    
    assert edl_robust_score < edl_safe_score
    assert edl_robust_score < efficient_score


def test_compute_route_scores_normalization():
    """测试归一化指标的范围和性质。"""
    breakdowns = {
        "efficient": RouteCostBreakdown(
            total_cost=100.0,
            component_totals={
                "base_distance": 50.0,
                "ice_risk": 50.0,
                "edl_risk": 5.0,
                "edl_uncertainty_penalty": 5.0,
            },
            component_fractions={},
            s_km=[0.0, 100.0],
            component_along_path={},
        ),
        "edl_safe": RouteCostBreakdown(
            total_cost=100.0,
            component_totals={
                "base_distance": 50.0,
                "ice_risk": 50.0,
                "edl_risk": 10.0,
                "edl_uncertainty_penalty": 10.0,
            },
            component_fractions={},
            s_km=[0.0, 100.0],
            component_along_path={},
        ),
        "edl_robust": RouteCostBreakdown(
            total_cost=100.0,
            component_totals={
                "base_distance": 50.0,
                "ice_risk": 50.0,
                "edl_risk": 15.0,
                "edl_uncertainty_penalty": 15.0,
            },
            component_fractions={},
            s_km=[0.0, 100.0],
            component_along_path={},
        ),
    }
    
    eco_by_key = {
        "efficient": {"fuel_total_t": 5.0, "co2_total_t": 15.0},
        "edl_safe": {"fuel_total_t": 10.0, "co2_total_t": 30.0},
        "edl_robust": {"fuel_total_t": 15.0, "co2_total_t": 45.0},
    }
    
    scores = compute_route_scores(
        breakdowns=breakdowns,
        eco_by_key=eco_by_key,
        weight_risk=0.33,
        weight_uncertainty=0.33,
        weight_fuel=0.34,
    )
    
    # 验证归一化指标都在 [0, 1] 范围内
    for key, score in scores.items():
        assert 0 <= score.norm_distance <= 1, f"{key}: norm_distance={score.norm_distance}"
        assert 0 <= score.norm_fuel <= 1, f"{key}: norm_fuel={score.norm_fuel}"
        assert 0 <= score.norm_edl_risk <= 1, f"{key}: norm_edl_risk={score.norm_edl_risk}"
        assert 0 <= score.norm_edl_uncertainty <= 1, f"{key}: norm_edl_uncertainty={score.norm_edl_uncertainty}"
    
    # 验证至少有一条路线的每个指标为 0（最优）
    # 例如，efficient 应该在燃油上为 0（燃油最少）
    assert scores["efficient"].norm_fuel == 0.0
    
    # efficient 应该在 EDL 风险上为 0（风险最低）
    assert scores["efficient"].norm_edl_risk == 0.0
    
    # efficient 应该在不确定性上为 0（不确定性最低）
    assert scores["efficient"].norm_edl_uncertainty == 0.0


def test_compute_route_scores_with_none_eco():
    """测试当 ECO 数据为 None 时的处理。"""
    breakdowns = {
        "efficient": RouteCostBreakdown(
            total_cost=100.0,
            component_totals={
                "base_distance": 50.0,
                "ice_risk": 50.0,
                "edl_risk": 0.0,
                "edl_uncertainty_penalty": 0.0,
            },
            component_fractions={},
            s_km=[0.0, 100.0],
            component_along_path={},
        ),
        "edl_safe": RouteCostBreakdown(
            total_cost=100.0,
            component_totals={
                "base_distance": 50.0,
                "ice_risk": 50.0,
                "edl_risk": 0.0,
                "edl_uncertainty_penalty": 0.0,
            },
            component_fractions={},
            s_km=[0.0, 100.0],
            component_along_path={},
        ),
        "edl_robust": RouteCostBreakdown(
            total_cost=100.0,
            component_totals={
                "base_distance": 50.0,
                "ice_risk": 50.0,
                "edl_risk": 0.0,
                "edl_uncertainty_penalty": 0.0,
            },
            component_fractions={},
            s_km=[0.0, 100.0],
            component_along_path={},
        ),
    }
    
    # 某些路线的 ECO 数据为 None
    eco_by_key = {
        "efficient": {"fuel_total_t": 10.0, "co2_total_t": 30.0},
        "edl_safe": None,  # 不可达的路线
        "edl_robust": {"fuel_total_t": 12.0, "co2_total_t": 36.0},
    }
    
    # 应该不报错
    scores = compute_route_scores(
        breakdowns=breakdowns,
        eco_by_key=eco_by_key,
        weight_risk=0.33,
        weight_uncertainty=0.33,
        weight_fuel=0.34,
    )
    
    # 验证返回了 3 个 RouteScore
    assert len(scores) == 3
    
    # edl_safe 的 fuel_t 应该是 None
    assert scores["edl_safe"].fuel_t is None


def test_compute_route_scores_equal_values():
    """测试当所有路线的某个指标相同时的处理。"""
    breakdowns = {
        "efficient": RouteCostBreakdown(
            total_cost=100.0,
            component_totals={
                "base_distance": 50.0,
                "ice_risk": 50.0,
                "edl_risk": 0.0,
                "edl_uncertainty_penalty": 0.0,
            },
            component_fractions={},
            s_km=[0.0, 100.0],
            component_along_path={},
        ),
        "edl_safe": RouteCostBreakdown(
            total_cost=100.0,
            component_totals={
                "base_distance": 50.0,
                "ice_risk": 50.0,
                "edl_risk": 0.0,
                "edl_uncertainty_penalty": 0.0,
            },
            component_fractions={},
            s_km=[0.0, 100.0],
            component_along_path={},
        ),
        "edl_robust": RouteCostBreakdown(
            total_cost=100.0,
            component_totals={
                "base_distance": 50.0,
                "ice_risk": 50.0,
                "edl_risk": 0.0,
                "edl_uncertainty_penalty": 0.0,
            },
            component_fractions={},
            s_km=[0.0, 100.0],
            component_along_path={},
        ),
    }
    
    # 所有路线的燃油消耗相同
    eco_by_key = {
        "efficient": {"fuel_total_t": 10.0, "co2_total_t": 30.0},
        "edl_safe": {"fuel_total_t": 10.0, "co2_total_t": 30.0},
        "edl_robust": {"fuel_total_t": 10.0, "co2_total_t": 30.0},
    }
    
    scores = compute_route_scores(
        breakdowns=breakdowns,
        eco_by_key=eco_by_key,
        weight_risk=0.33,
        weight_uncertainty=0.33,
        weight_fuel=0.34,
    )
    
    # 当所有值相同时，归一化后应该都是 0
    for key, score in scores.items():
        assert score.norm_fuel == 0.0
        assert score.norm_edl_risk == 0.0
        assert score.norm_edl_uncertainty == 0.0

