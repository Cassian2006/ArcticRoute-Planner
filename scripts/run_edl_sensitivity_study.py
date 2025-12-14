"""
EDL 灵敏度分析脚本。

对一组标准场景，分别以三种模式运行路线规划：
  - efficient（无 EDL）
  - edl_safe（有 EDL 风险）
  - edl_robust（风险 + 不确定性）

输出结果到 CSV，并生成简单图表。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 导入项目模块
from arcticroute.core.grid import make_demo_grid, load_real_grid_from_nc
from arcticroute.core.landmask import load_real_landmask_from_nc
from arcticroute.core.cost import build_demo_cost, build_cost_from_real_env
from arcticroute.core.env_real import load_real_env_for_grid
from arcticroute.core.astar import plan_route_latlon
from arcticroute.core.analysis import compute_route_cost_breakdown
from arcticroute.core.eco.vessel_profiles import get_default_profiles

# 导入共享配置
from arcticroute.config import EDL_MODES, SCENARIOS
from arcticroute.config.scenarios import Scenario


# ============================================================================
# 配置常数
# ============================================================================

# 从共享配置模块导入 EDL 模式定义
# 这确保 CLI 和 UI 使用相同的参数
MODES = EDL_MODES


# ============================================================================
# 数据类
# ============================================================================

class SensitivityResult:
    """单个场景 + 模式的结果。"""
    
    def __init__(self, scenario_name: str, mode: str):
        self.scenario_name = scenario_name
        self.mode = mode
        
        # 基本信息
        self.reachable = False
        self.distance_km = 0.0
        self.total_cost = 0.0
        
        # 成本分解
        self.components: Dict[str, float] = {}
        
        # EDL 相关
        self.edl_risk_cost = 0.0
        self.edl_uncertainty_cost = 0.0
        self.mean_uncertainty = 0.0
        self.max_uncertainty = 0.0
        
        # 错误信息
        self.error_message = ""
    
    def to_dict(self) -> Dict:
        """转换为字典，用于 CSV 输出。"""
        result = {
            "scenario": self.scenario_name,
            "mode": self.mode,
            "reachable": "yes" if self.reachable else "no",
            "distance_km": f"{self.distance_km:.2f}",
            "total_cost": f"{self.total_cost:.4f}",
            "edl_risk_cost": f"{self.edl_risk_cost:.4f}",
            "edl_uncertainty_cost": f"{self.edl_uncertainty_cost:.4f}",
            "mean_uncertainty": f"{self.mean_uncertainty:.4f}",
            "max_uncertainty": f"{self.max_uncertainty:.4f}",
        }
        
        # 添加各成本分量
        for comp_name, comp_value in self.components.items():
            result[f"comp_{comp_name}"] = f"{comp_value:.4f}"
        
        # 添加错误信息
        if self.error_message:
            result["error"] = self.error_message
        
        return result


# ============================================================================
# 核心函数
# ============================================================================

def run_single_scenario_mode(
    scenario: Scenario,
    mode: str,
    use_real_data: bool = False,
) -> SensitivityResult:
    """
    对单个场景 + 模式运行路线规划。
    
    Args:
        scenario: Scenario 对象
        mode: 模式名称（"efficient", "edl_safe", "edl_robust"）
        use_real_data: 是否使用真实数据（默认 False，使用 demo grid）
    
    Returns:
        SensitivityResult 对象
    """
    result = SensitivityResult(scenario.name, mode)
    
    try:
        # 获取模式配置
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}")
        mode_config = MODES[mode]
        
        # 加载网格和陆地掩码
        if use_real_data:
            try:
                grid = load_real_grid_from_nc()
                if grid is None:
                    raise ValueError("Failed to load real grid")
                land_mask = load_real_landmask_from_nc(grid)
                if land_mask is None:
                    raise ValueError("Failed to load real landmask")
            except Exception as e:
                print(f"[WARN] Failed to load real data: {e}, falling back to demo grid")
                grid, land_mask = make_demo_grid()
        else:
            grid, land_mask = make_demo_grid()
        
        # 构建成本场
        if use_real_data:
            try:
                env = load_real_env_for_grid(grid, ym=scenario.ym)
                cost_field = build_cost_from_real_env(
                    grid=grid,
                    land_mask=land_mask,
                    env=env,
                    ice_penalty=mode_config["ice_penalty"],
                    wave_penalty=0.0,
                    vessel_profile=get_default_profiles().get(scenario.vessel_profile),
                    w_edl=mode_config["w_edl"],
                    use_edl=mode_config["use_edl"],
                    use_edl_uncertainty=mode_config["use_edl_uncertainty"],
                    edl_uncertainty_weight=mode_config["edl_uncertainty_weight"],
                )
            except Exception as e:
                print(f"[WARN] Failed to build cost from real env: {e}, using demo cost")
                cost_field = build_demo_cost(grid, land_mask, ice_penalty=mode_config["ice_penalty"])
        else:
            cost_field = build_demo_cost(grid, land_mask, ice_penalty=mode_config["ice_penalty"])
        
        # 规划路线
        route = plan_route_latlon(
            cost_field,
            scenario.start_lat,
            scenario.start_lon,
            scenario.end_lat,
            scenario.end_lon,
            neighbor8=True,
        )
        
        if not route:
            result.reachable = False
            result.error_message = "Route not reachable"
            return result
        
        result.reachable = True
        
        # 计算路线距离
        from arcticroute.core.analysis import haversine_km
        total_distance = 0.0
        for i in range(len(route) - 1):
            lat1, lon1 = route[i]
            lat2, lon2 = route[i + 1]
            total_distance += haversine_km(lat1, lon1, lat2, lon2)
        result.distance_km = total_distance
        
        # 计算成本分解
        breakdown = compute_route_cost_breakdown(grid, cost_field, route)
        result.total_cost = breakdown.total_cost
        result.components = breakdown.component_totals.copy()
        
        # 提取 EDL 相关成本
        result.edl_risk_cost = breakdown.component_totals.get("edl_risk", 0.0)
        result.edl_uncertainty_cost = breakdown.component_totals.get("edl_uncertainty_penalty", 0.0)
        
        # 计算不确定性统计
        if cost_field.edl_uncertainty is not None:
            # 沿路线采样不确定性
            from arcticroute.core.grid import Grid2D
            lat2d = grid.lat2d
            lon2d = grid.lon2d
            ny, nx = grid.shape()
            
            uncertainties = []
            for lat, lon in route:
                dist = np.sqrt((lat2d - lat) ** 2 + (lon2d - lon) ** 2)
                i, j = np.unravel_index(np.argmin(dist), dist.shape)
                i = np.clip(i, 0, ny - 1)
                j = np.clip(j, 0, nx - 1)
                uncertainties.append(cost_field.edl_uncertainty[i, j])
            
            if uncertainties:
                result.mean_uncertainty = float(np.mean(uncertainties))
                result.max_uncertainty = float(np.max(uncertainties))
        
        return result
        
    except Exception as e:
        result.reachable = False
        result.error_message = str(e)
        print(f"[ERROR] Scenario {scenario.name} mode {mode} failed: {e}")
        return result


def run_all_scenarios(
    scenarios: List[Scenario] | None = None,
    modes: List[str] | None = None,
    use_real_data: bool = False,
    dry_run: bool = False,
) -> List[SensitivityResult]:
    """
    对所有场景和模式运行灵敏度分析。
    
    Args:
        scenarios: 场景列表（默认使用 SCENARIOS）
        modes: 模式列表（默认使用所有模式）
        use_real_data: 是否使用真实数据
        dry_run: 干运行模式（仅返回空结果，不实际计算）
    
    Returns:
        SensitivityResult 列表
    """
    if scenarios is None:
        scenarios = SCENARIOS
    if modes is None:
        modes = list(MODES.keys())
    
    results = []
    
    total = len(scenarios) * len(modes)
    count = 0
    
    for scenario in scenarios:
        for mode in modes:
            count += 1
            print(f"[{count}/{total}] Running {scenario.name} / {mode}...")
            
            if dry_run:
                # 干运行：返回空结果
                result = SensitivityResult(scenario.name, mode)
                result.reachable = False
                result.error_message = "dry_run"
            else:
                result = run_single_scenario_mode(scenario, mode, use_real_data=use_real_data)
            
            results.append(result)
    
    return results


def write_results_to_csv(
    results: List[SensitivityResult],
    output_path: Path | str = "reports/edl_sensitivity_results.csv",
) -> None:
    """
    将结果写入 CSV 文件。
    
    Args:
        results: SensitivityResult 列表
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not results:
        print("[WARN] No results to write")
        return
    
    # 收集所有可能的列名
    fieldnames = set()
    for result in results:
        fieldnames.update(result.to_dict().keys())
    
    # 排序列名，保证一致性
    fieldnames = sorted(list(fieldnames))
    
    # 确保关键列在前面
    key_fields = ["scenario", "mode", "reachable", "distance_km", "total_cost", 
                  "edl_risk_cost", "edl_uncertainty_cost", "mean_uncertainty", "max_uncertainty"]
    for field in key_fields:
        if field in fieldnames:
            fieldnames.remove(field)
    fieldnames = key_fields + fieldnames
    
    # 写入 CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = result.to_dict()
            # 填充缺失的字段
            for field in fieldnames:
                if field not in row:
                    row[field] = ""
            writer.writerow(row)
    
    print(f"[OK] Results written to {output_path}")


def print_summary(results: List[SensitivityResult]) -> None:
    """
    打印结果摘要表。
    
    Args:
        results: SensitivityResult 列表
    """
    print("\n" + "=" * 100)
    print("EDL SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 100)
    
    # 按场景分组
    by_scenario = {}
    for result in results:
        if result.scenario_name not in by_scenario:
            by_scenario[result.scenario_name] = {}
        by_scenario[result.scenario_name][result.mode] = result
    
    # 打印每个场景
    for scenario_name in sorted(by_scenario.keys()):
        print(f"\n[{scenario_name}]")
        print(f"{'Mode':<20} {'Reachable':<12} {'Distance (km)':<15} {'Total Cost':<15} {'EDL Risk':<15} {'EDL Unc':<15}")
        print("-" * 92)
        
        for mode in ["efficient", "edl_safe", "edl_robust"]:
            if mode in by_scenario[scenario_name]:
                result = by_scenario[scenario_name][mode]
                reachable = "Yes" if result.reachable else "No"
                dist = f"{result.distance_km:.2f}" if result.reachable else "N/A"
                cost = f"{result.total_cost:.4f}" if result.reachable else "N/A"
                edl_risk = f"{result.edl_risk_cost:.4f}" if result.reachable else "N/A"
                edl_unc = f"{result.edl_uncertainty_cost:.4f}" if result.reachable else "N/A"
                
                print(f"{mode:<20} {reachable:<12} {dist:<15} {cost:<15} {edl_risk:<15} {edl_unc:<15}")
    
    print("\n" + "=" * 100)


def generate_charts(
    results: List[SensitivityResult],
    output_dir: Path | str = "reports",
) -> None:
    """
    生成简单的柱状图。
    
    Args:
        results: SensitivityResult 列表
        output_dir: 输出目录
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping chart generation")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 按场景分组
    by_scenario = {}
    for result in results:
        if result.scenario_name not in by_scenario:
            by_scenario[result.scenario_name] = {}
        by_scenario[result.scenario_name][result.mode] = result
    
    # 对每个场景生成图表
    for scenario_name, mode_results in by_scenario.items():
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"EDL Sensitivity: {scenario_name}", fontsize=14, fontweight="bold")
            
            modes = ["efficient", "edl_safe", "edl_robust"]
            
            # 准备数据
            total_costs = []
            edl_risk_costs = []
            edl_unc_costs = []
            
            for mode in modes:
                if mode in mode_results:
                    result = mode_results[mode]
                    if result.reachable:
                        total_costs.append(result.total_cost)
                        edl_risk_costs.append(result.edl_risk_cost)
                        edl_unc_costs.append(result.edl_uncertainty_cost)
                    else:
                        total_costs.append(0)
                        edl_risk_costs.append(0)
                        edl_unc_costs.append(0)
                else:
                    total_costs.append(0)
                    edl_risk_costs.append(0)
                    edl_unc_costs.append(0)
            
            # 绘制三个子图
            x = np.arange(len(modes))
            width = 0.6
            
            # 子图 1: 总成本
            axes[0].bar(x, total_costs, width, color="steelblue")
            axes[0].set_ylabel("Total Cost", fontsize=11)
            axes[0].set_title("Total Cost", fontsize=12)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(modes, rotation=15, ha="right")
            axes[0].grid(axis="y", alpha=0.3)
            
            # 子图 2: EDL 风险成本
            axes[1].bar(x, edl_risk_costs, width, color="coral")
            axes[1].set_ylabel("EDL Risk Cost", fontsize=11)
            axes[1].set_title("EDL Risk Cost", fontsize=12)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(modes, rotation=15, ha="right")
            axes[1].grid(axis="y", alpha=0.3)
            
            # 子图 3: EDL 不确定性成本
            axes[2].bar(x, edl_unc_costs, width, color="lightgreen")
            axes[2].set_ylabel("EDL Uncertainty Cost", fontsize=11)
            axes[2].set_title("EDL Uncertainty Cost", fontsize=12)
            axes[2].set_xticks(x)
            axes[2].set_xticklabels(modes, rotation=15, ha="right")
            axes[2].grid(axis="y", alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            output_file = output_dir / f"edl_sensitivity_{scenario_name}.png"
            plt.savefig(output_file, dpi=100, bbox_inches="tight")
            print(f"[OK] Chart saved to {output_file}")
            
            plt.close(fig)
            
        except Exception as e:
            print(f"[WARN] Failed to generate chart for {scenario_name}: {e}")


# ============================================================================
# 主函数
# ============================================================================

def main(
    dry_run: bool = False,
    use_real_data: bool = False,
    output_csv: str = "reports/edl_sensitivity_results.csv",
    output_dir: str = "reports",
) -> None:
    """
    主函数：运行完整的灵敏度分析。
    
    Args:
        dry_run: 干运行模式
        use_real_data: 是否使用真实数据
        output_csv: 输出 CSV 文件路径
        output_dir: 输出目录
    """
    print("[START] EDL Sensitivity Analysis")
    print(f"[CONFIG] dry_run={dry_run}, use_real_data={use_real_data}")
    
    # 运行分析
    results = run_all_scenarios(
        scenarios=SCENARIOS,
        modes=list(MODES.keys()),
        use_real_data=use_real_data,
        dry_run=dry_run,
    )
    
    # 输出结果
    write_results_to_csv(results, output_csv)
    print_summary(results)
    
    # 生成图表
    if not dry_run:
        generate_charts(results, output_dir)
    
    print("[DONE] EDL Sensitivity Analysis Complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="EDL Sensitivity Analysis Script"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (no actual computation)",
    )
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Use real data instead of demo grid",
    )
    parser.add_argument(
        "--output-csv",
        default="reports/edl_sensitivity_results.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Output directory for charts",
    )
    
    args = parser.parse_args()
    
    main(
        dry_run=args.dry_run,
        use_real_data=args.use_real_data,
        output_csv=args.output_csv,
        output_dir=args.output_dir,
    )

