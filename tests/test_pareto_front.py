"""
Pareto 前沿算法单元测试。

测试支配关系、Pareto 前沿计算和数据框导出。
"""

import pytest
from arcticroute.core.pareto import (
    CandidateSolution,
    dominates,
    pareto_front,
    solutions_to_dataframe,
)
from arcticroute.core.analysis import RouteCostBreakdown


def make_test_solution(
    key: str,
    distance_km: float,
    total_cost: float,
    edl_risk: float = 0.0,
    edl_uncertainty: float = 0.0,
) -> CandidateSolution:
    """创建测试用的候选解。"""
    breakdown = RouteCostBreakdown(
        total_cost=total_cost,
        component_totals={
            "edl_risk": edl_risk,
            "edl_uncertainty_penalty": edl_uncertainty,
        },
        component_fractions={},
        s_km=[0.0, distance_km],  # 简化：假设只有两个点
        component_along_path={},
    )
    
    return CandidateSolution(
        key=key,
        route=[(75.0, 20.0), (75.0, 140.0)],
        breakdown=breakdown,
        eco=None,
        meta=None,
    )


class TestDominates:
    """测试支配关系判断。"""
    
    def test_dominates_all_better(self):
        """A 在所有目标上都优于 B，则 A 支配 B。"""
        a = make_test_solution("a", distance_km=100.0, total_cost=50.0)
        b = make_test_solution("b", distance_km=150.0, total_cost=70.0)
        
        assert dominates(a, b)
        assert not dominates(b, a)
    
    def test_dominates_partial_better(self):
        """A 在某些目标上优于 B，在某些目标上劣于 B，则 A 不支配 B。"""
        a = make_test_solution("a", distance_km=100.0, total_cost=70.0)
        b = make_test_solution("b", distance_km=150.0, total_cost=50.0)
        
        assert not dominates(a, b)
        assert not dominates(b, a)
    
    def test_dominates_equal(self):
        """A 和 B 相同，则互不支配。"""
        a = make_test_solution("a", distance_km=100.0, total_cost=50.0)
        b = make_test_solution("b", distance_km=100.0, total_cost=50.0)
        
        assert not dominates(a, b)
        assert not dominates(b, a)
    
    def test_dominates_with_risk(self):
        """考虑 EDL 风险的支配关系。"""
        a = make_test_solution("a", distance_km=100.0, total_cost=50.0, edl_risk=10.0)
        b = make_test_solution("b", distance_km=100.0, total_cost=50.0, edl_risk=20.0)
        
        assert dominates(a, b)
        assert not dominates(b, a)


class TestParetoFront:
    """测试 Pareto 前沿计算。"""
    
    def test_pareto_front_empty(self):
        """空列表应返回空列表。"""
        front = pareto_front([])
        assert front == []
    
    def test_pareto_front_single(self):
        """单个候选解应该在前沿上。"""
        a = make_test_solution("a", distance_km=100.0, total_cost=50.0)
        front = pareto_front([a])
        
        assert len(front) == 1
        assert front[0].key == "a"
    
    def test_pareto_front_all_dominated(self):
        """如果所有解都被某个解支配，则只有该解在前沿上。"""
        a = make_test_solution("a", distance_km=100.0, total_cost=50.0)
        b = make_test_solution("b", distance_km=150.0, total_cost=70.0)
        c = make_test_solution("c", distance_km=200.0, total_cost=100.0)
        
        front = pareto_front([a, b, c])
        
        assert len(front) == 1
        assert front[0].key == "a"
    
    def test_pareto_front_multiple(self):
        """多个非支配解应该都在前沿上。"""
        # 创建一个三角形的 Pareto 前沿
        a = make_test_solution("a", distance_km=100.0, total_cost=100.0)  # 短距离，高成本
        b = make_test_solution("b", distance_km=150.0, total_cost=50.0)   # 中距离，低成本
        c = make_test_solution("c", distance_km=200.0, total_cost=30.0)   # 长距离，最低成本
        
        front = pareto_front([a, b, c])
        
        assert len(front) == 3
        keys = {sol.key for sol in front}
        assert keys == {"a", "b", "c"}
    
    def test_pareto_front_with_dominated(self):
        """混合支配和非支配解。"""
        # 非支配解
        a = make_test_solution("a", distance_km=100.0, total_cost=100.0)
        b = make_test_solution("b", distance_km=200.0, total_cost=50.0)
        
        # 被 a 支配的解
        c = make_test_solution("c", distance_km=150.0, total_cost=120.0)
        
        # 被 b 支配的解
        d = make_test_solution("d", distance_km=250.0, total_cost=70.0)
        
        front = pareto_front([a, b, c, d])
        
        assert len(front) == 2
        keys = {sol.key for sol in front}
        assert keys == {"a", "b"}


class TestSolutionsToDataframe:
    """测试数据框导出。"""
    
    def test_solutions_to_dataframe_empty(self):
        """空列表应返回空 DataFrame。"""
        df = solutions_to_dataframe([])
        
        assert len(df) == 0
        assert "key" in df.columns
    
    def test_solutions_to_dataframe_single(self):
        """单个解应该导出为一行。"""
        a = make_test_solution("a", distance_km=100.0, total_cost=50.0, edl_risk=10.0)
        df = solutions_to_dataframe([a])
        
        assert len(df) == 1
        assert df.loc[0, "key"] == "a"
        assert df.loc[0, "distance_km"] == 100.0
        assert df.loc[0, "total_cost"] == 50.0
        assert df.loc[0, "edl_risk"] == 10.0
    
    def test_solutions_to_dataframe_multiple(self):
        """多个解应该导出为多行。"""
        a = make_test_solution("a", distance_km=100.0, total_cost=50.0)
        b = make_test_solution("b", distance_km=150.0, total_cost=70.0)
        c = make_test_solution("c", distance_km=200.0, total_cost=100.0)
        
        df = solutions_to_dataframe([a, b, c])
        
        assert len(df) == 3
        assert list(df["key"]) == ["a", "b", "c"]
        assert list(df["distance_km"]) == [100.0, 150.0, 200.0]
    
    def test_solutions_to_dataframe_custom_fields(self):
        """指定自定义字段。"""
        a = make_test_solution("a", distance_km=100.0, total_cost=50.0)
        df = solutions_to_dataframe([a], include_fields=["key", "distance_km"])
        
        assert set(df.columns) == {"key", "distance_km"}
        assert df.loc[0, "key"] == "a"
        assert df.loc[0, "distance_km"] == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



