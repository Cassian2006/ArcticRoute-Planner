"""
Pareto 多目标前沿演示脚本。

生成一批候选权重组合，跑规划，计算 Pareto 前沿，输出 CSV 报告。
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from arcticroute.core.pareto import (
    CandidateSolution,
    pareto_front,
    solutions_to_dataframe,
)
from arcticroute.core.grid import make_demo_grid
from arcticroute.core.cost import build_demo_cost  # 直接从 cost.py 导入
from arcticroute.core.astar import plan_route_latlon
from arcticroute.core.analysis import compute_route_cost_breakdown

# 尝试导入 eco 模块
try:
    from arcticroute.core.eco.eco_model import estimate_route_eco
except ImportError:
    # 如果 eco 模块不可用，定义一个占位符
    def estimate_route_eco(route_latlon, vessel_type="icebreaker"):
        """占位符 eco 估算函数。"""
        return None


# 默认的三个预设 profile
PRESET_PROFILES = {
    "efficient": {
        "w_ice": 1.0,
        "w_wave": 0.0,
        "w_ais": 0.0,
        "w_edl": 0.0,
        "use_edl_uncertainty": False,
    },
    "edl_safe": {
        "w_ice": 1.0,
        "w_wave": 0.5,
        "w_ais": 0.3,
        "w_edl": 2.0,
        "use_edl_uncertainty": False,
    },
    "edl_robust": {
        "w_ice": 1.0,
        "w_wave": 0.5,
        "w_ais": 0.3,
        "w_edl": 2.0,
        "use_edl_uncertainty": True,
    },
}

# 演示网格的起点和终点
DEMO_START = (75.0, 20.0)  # (lat, lon)
DEMO_END = (75.0, 140.0)


def generate_random_profiles(
    n: int = 20,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    生成 N 个随机权重组合。
    
    Args:
        n: 生成的随机 profile 数量
        seed: 随机种子，保证可复现
    
    Returns:
        {profile_key: {weight_dict}} 字典
    """
    random.seed(seed)
    np.random.seed(seed)
    
    profiles = {}
    
    for i in range(n):
        key = f"random_{i:03d}"
        
        # 随机生成权重（0..2 范围）
        w_ice = random.uniform(0.5, 2.0)
        w_wave = random.uniform(0.0, 1.5)
        w_ais = random.uniform(0.0, 1.0)
        w_edl = random.uniform(0.0, 3.0)
        use_edl_uncertainty = random.choice([True, False])
        
        profiles[key] = {
            "w_ice": w_ice,
            "w_wave": w_wave,
            "w_ais": w_ais,
            "w_edl": w_edl,
            "use_edl_uncertainty": use_edl_uncertainty,
        }
    
    return profiles


def plan_single_route(
    grid,
    cost_field,
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> Tuple[List[Tuple[float, float]], bool]:
    """
    规划单条路线。
    
    Args:
        grid: Grid2D 对象
        cost_field: CostField 对象
        start: 起点 (lat, lon)
        end: 终点 (lat, lon)
    
    Returns:
        (route_latlon, reachable) 元组
    """
    try:
        route_latlon = plan_route_latlon(grid, cost_field, start, end)
        reachable = len(route_latlon) > 0
        return route_latlon, reachable
    except Exception as e:
        print(f"Warning: 规划失败: {e}")
        return [], False


def run_pareto_suite(
    n_random: int = 20,
    seed: int = 42,
    output_dir: str = "reports",
) -> Tuple[List[CandidateSolution], List[CandidateSolution]]:
    """
    运行 Pareto 多目标前沿演示。
    
    Args:
        n_random: 随机 profile 数量
        seed: 随机种子
        output_dir: 输出目录
    
    Returns:
        (all_solutions, front_solutions) 元组
    """
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[Pareto Suite] 初始化演示网格...")
    grid = make_demo_grid()
    
    # 生成所有 profile（预设 + 随机）
    all_profiles = dict(PRESET_PROFILES)
    random_profiles = generate_random_profiles(n_random, seed)
    all_profiles.update(random_profiles)
    
    print(f"[Pareto Suite] 生成 {len(all_profiles)} 个候选 profile")
    print(f"  - 预设: {list(PRESET_PROFILES.keys())}")
    print(f"  - 随机: {n_random} 个")
    
    # 规划所有候选
    all_solutions: List[CandidateSolution] = []
    
    for profile_key, weights in all_profiles.items():
        print(f"\n[Pareto Suite] 规划 {profile_key}...")
        
        # 构建成本场
        try:
            cost_field = build_demo_cost(
                grid,
                w_ice=weights.get("w_ice", 1.0),
                w_wave=weights.get("w_wave", 0.0),
                w_ais=weights.get("w_ais", 0.0),
                w_edl=weights.get("w_edl", 0.0),
                use_edl_uncertainty=weights.get("use_edl_uncertainty", False),
            )
        except Exception as e:
            print(f"  Warning: 构建成本场失败: {e}")
            continue
        
        # 规划路线
        route_latlon, reachable = plan_single_route(
            grid, cost_field, DEMO_START, DEMO_END
        )
        
        if not reachable:
            print(f"  Warning: 无法到达终点")
            continue
        
        # 计算成本分解
        try:
            breakdown = compute_route_cost_breakdown(grid, cost_field, route_latlon)
        except Exception as e:
            print(f"  Warning: 成本分解失败: {e}")
            continue
        
        # 计算生态经济指标
        eco = None
        try:
            eco = estimate_route_eco(route_latlon, vessel_type="icebreaker")
        except Exception:
            eco = None
        
        # 创建候选解
        solution = CandidateSolution(
            key=profile_key,
            route=route_latlon,
            breakdown=breakdown,
            eco=eco,
            meta=weights,
        )
        
        all_solutions.append(solution)
        print(f"  ✓ 规划成功: distance={breakdown.s_km[-1]:.1f}km, cost={breakdown.total_cost:.2f}")
    
    print(f"\n[Pareto Suite] 共规划 {len(all_solutions)} 条路线")
    
    # 计算 Pareto 前沿
    print(f"[Pareto Suite] 计算 Pareto 前沿...")
    front_solutions = pareto_front(
        all_solutions,
        minimize_fields=("distance_km", "total_cost", "edl_risk", "edl_uncertainty"),
    )
    
    print(f"[Pareto Suite] Pareto 前沿包含 {len(front_solutions)} 条路线")
    
    # 输出 CSV
    print(f"[Pareto Suite] 输出 CSV 报告...")
    
    # 全部候选
    all_df = solutions_to_dataframe(all_solutions)
    all_csv = output_path / "pareto_solutions.csv"
    all_df.to_csv(all_csv, index=False)
    print(f"  ✓ {all_csv}")
    
    # Pareto 前沿
    front_df = solutions_to_dataframe(front_solutions)
    front_csv = output_path / "pareto_front.csv"
    front_df.to_csv(front_csv, index=False)
    print(f"  ✓ {front_csv}")
    
    return all_solutions, front_solutions


def main():
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="Pareto 多目标前沿演示脚本"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="随机候选数量（默认 20）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="输出目录（默认 reports）",
    )
    
    args = parser.parse_args()
    
    all_solutions, front_solutions = run_pareto_suite(
        n_random=args.n,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    
    print(f"\n[Pareto Suite] 完成！")
    print(f"  全部候选: {len(all_solutions)}")
    print(f"  Pareto 前沿: {len(front_solutions)}")


if __name__ == "__main__":
    main()

