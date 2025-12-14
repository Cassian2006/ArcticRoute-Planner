"""
Pareto 多目标前沿分析模块。

提供候选解的定义、支配关系判断、Pareto 前沿计算和数据框导出功能。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd

from .analysis import RouteCostBreakdown


@dataclass
class CandidateSolution:
    """候选路线解的数据结构。
    
    Attributes:
        key: 候选 ID（例如 "efficient", "edl_safe", "edl_robust", "random_001" 等）
        route: 路线坐标列表 [(lat, lon), ...]
        breakdown: 成本分解结果（RouteCostBreakdown 对象）
        eco: 生态经济指标字典（可选），例如 {"fuel_total_t": 123.4, "co2_total_t": 456.7}
        meta: 元数据字典，保存权重配置等，例如 {"w_ice": 1.0, "w_wave": 0.5, ...}
    """
    
    key: str
    route: List[Tuple[float, float]]
    breakdown: RouteCostBreakdown
    eco: Optional[Dict[str, float]] = None
    meta: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """初始化默认值。"""
        if self.eco is None:
            self.eco = {}
        if self.meta is None:
            self.meta = {}


def dominates(
    a: CandidateSolution,
    b: CandidateSolution,
    minimize_fields: Tuple[str, ...] = ("distance_km", "total_cost", "edl_risk", "edl_uncertainty"),
) -> bool:
    """
    判断候选解 a 是否支配候选解 b。
    
    支配定义：a 在所有目标上都不劣于 b，且至少在一个目标上严格优于 b。
    
    Args:
        a: 候选解 A
        b: 候选解 B
        minimize_fields: 要最小化的目标字段名称元组
    
    Returns:
        True 如果 a 支配 b，否则 False
    """
    
    def get_value(sol: CandidateSolution, field: str) -> float:
        """从候选解中获取指标值，缺失则返回 0。"""
        if field == "distance_km":
            # 从 breakdown.s_km 的最后一个值获取距离
            return sol.breakdown.s_km[-1] if sol.breakdown.s_km else 0.0
        elif field == "total_cost":
            return sol.breakdown.total_cost
        elif field == "edl_risk":
            return sol.breakdown.component_totals.get("edl_risk", 0.0)
        elif field == "edl_uncertainty":
            return sol.breakdown.component_totals.get("edl_uncertainty_penalty", 0.0)
        elif field == "ice_risk":
            return sol.breakdown.component_totals.get("ice_risk", 0.0)
        elif field == "wave_risk":
            return sol.breakdown.component_totals.get("wave_risk", 0.0)
        elif field == "ais_risk":
            return sol.breakdown.component_totals.get("ais_density", 0.0)
        else:
            # 尝试从 eco 字典中获取
            if sol.eco and field in sol.eco:
                return sol.eco[field]
            return 0.0
    
    # 检查 a 是否在所有目标上都不劣于 b
    better_or_equal = True
    strictly_better = False
    
    for field in minimize_fields:
        val_a = get_value(a, field)
        val_b = get_value(b, field)
        
        if val_a > val_b:
            # a 在该目标上劣于 b
            better_or_equal = False
            break
        elif val_a < val_b:
            # a 在该目标上严格优于 b
            strictly_better = True
    
    return better_or_equal and strictly_better


def pareto_front(
    candidates: List[CandidateSolution],
    minimize_fields: Tuple[str, ...] = ("distance_km", "total_cost", "edl_risk", "edl_uncertainty"),
) -> List[CandidateSolution]:
    """
    计算 Pareto 前沿。
    
    使用 O(n^2) 算法：对每个候选解，检查是否被其他候选解支配。
    
    Args:
        candidates: 候选解列表
        minimize_fields: 要最小化的目标字段名称元组
    
    Returns:
        Pareto 前沿上的候选解列表
    """
    if not candidates:
        return []
    
    front = []
    
    for i, candidate_a in enumerate(candidates):
        dominated = False
        
        for j, candidate_b in enumerate(candidates):
            if i == j:
                continue
            
            # 检查 candidate_b 是否支配 candidate_a
            if dominates(candidate_b, candidate_a, minimize_fields):
                dominated = True
                break
        
        if not dominated:
            front.append(candidate_a)
    
    return front


def solutions_to_dataframe(
    solutions: List[CandidateSolution],
    include_fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    将候选解列表转换为 DataFrame。
    
    Args:
        solutions: 候选解列表
        include_fields: 要包含的字段列表。如果为 None，则包含默认字段。
                       默认字段：key, distance_km, total_cost, edl_risk, edl_uncertainty, 
                                ice_risk, wave_risk, ais_risk
    
    Returns:
        DataFrame，其中每行代表一个候选解
    """
    
    if include_fields is None:
        include_fields = [
            "key",
            "distance_km",
            "total_cost",
            "edl_risk",
            "edl_uncertainty",
            "ice_risk",
            "wave_risk",
            "ais_risk",
        ]
    
    rows = []
    
    for sol in solutions:
        row = {}
        
        for field in include_fields:
            if field == "key":
                row[field] = sol.key
            elif field == "distance_km":
                row[field] = sol.breakdown.s_km[-1] if sol.breakdown.s_km else 0.0
            elif field == "total_cost":
                row[field] = sol.breakdown.total_cost
            elif field == "edl_risk":
                row[field] = sol.breakdown.component_totals.get("edl_risk", 0.0)
            elif field == "edl_uncertainty":
                row[field] = sol.breakdown.component_totals.get("edl_uncertainty_penalty", 0.0)
            elif field == "ice_risk":
                row[field] = sol.breakdown.component_totals.get("ice_risk", 0.0)
            elif field == "wave_risk":
                row[field] = sol.breakdown.component_totals.get("wave_risk", 0.0)
            elif field == "ais_risk":
                row[field] = sol.breakdown.component_totals.get("ais_density", 0.0)
            elif field in sol.eco:
                row[field] = sol.eco[field]
            else:
                row[field] = 0.0
        
        rows.append(row)
    
    # 如果没有行，创建一个空 DataFrame，但保留列
    if not rows:
        return pd.DataFrame(columns=include_fields)
    
    return pd.DataFrame(rows)


__all__ = [
    "CandidateSolution",
    "dominates",
    "pareto_front",
    "solutions_to_dataframe",
]

