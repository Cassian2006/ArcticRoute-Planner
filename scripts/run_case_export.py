"""
CLI 脚本：运行单个规划案例并导出结果。

用法：
    python -m scripts.run_case_export \\
        --scenario barents_to_chukchi \\
        --mode edl_safe \\
        --use-real-data \\
        --out-csv reports/result.csv \\
        --out-json reports/result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from arcticroute.experiments.runner import run_single_case, ModeName
from arcticroute.config import list_scenarios


def _convert_to_serializable(obj):
    """将 numpy 类型转换为 Python 原生类型，便于 JSON 序列化。"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    else:
        return obj


def print_summary(result) -> None:
    """打印规划结果摘要。"""
    print(f"\n{'='*70}")
    print(f"[SCENARIO] {result.scenario:30s} [MODE] {result.mode}")
    print(f"{'='*70}")
    
    if result.reachable:
        print(f"Reachable: Yes")
        print(f"Distance: {result.distance_km:.1f} km")
        print(f"Total cost: {result.total_cost:.1f}")
        
        # 打印各成本分量
        if result.edl_risk_cost is not None:
            pct = (result.edl_risk_cost / result.total_cost * 100) if result.total_cost > 0 else 0
            print(f"EDL risk:  {result.edl_risk_cost:.1f}   ({pct:.1f}%)")
        
        if result.edl_unc_cost is not None:
            pct = (result.edl_unc_cost / result.total_cost * 100) if result.total_cost > 0 else 0
            print(f"EDL unc:   {result.edl_unc_cost:.1f}   ({pct:.1f}%)")
        
        if result.ice_cost is not None:
            pct = (result.ice_cost / result.total_cost * 100) if result.total_cost > 0 else 0
            print(f"Ice cost:  {result.ice_cost:.1f}   ({pct:.1f}%)")
        
        if result.wave_cost is not None:
            pct = (result.wave_cost / result.total_cost * 100) if result.total_cost > 0 else 0
            print(f"Wave cost: {result.wave_cost:.1f}   ({pct:.1f}%)")
        
        # 元数据
        print(f"\nMetadata:")
        print(f"  Year-Month: {result.meta.get('ym', 'N/A')}")
        print(f"  Use Real Data: {result.meta.get('use_real_data', False)}")
        print(f"  Cost Mode: {result.meta.get('cost_mode', 'N/A')}")
        print(f"  Vessel: {result.meta.get('vessel_profile', 'N/A')}")
        print(f"  EDL Backend: {result.meta.get('edl_backend', 'N/A')}")
        
        if result.meta.get('fallback_reason'):
            print(f"  Fallback Reason: {result.meta['fallback_reason']}")
    else:
        print(f"Reachable: No")
        print(f"Failed to find a route from ({result.meta.get('start_lat', 'N/A')}, "
              f"{result.meta.get('start_lon', 'N/A')}) to "
              f"({result.meta.get('end_lat', 'N/A')}, {result.meta.get('end_lon', 'N/A')})")
    
    print(f"{'='*70}\n")


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(
        description="运行单个规划案例并导出结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python -m scripts.run_case_export \\
    --scenario barents_to_chukchi \\
    --mode edl_safe \\
    --use-real-data \\
    --out-csv reports/result.csv \\
    --out-json reports/result.json
        """
    )
    
    # 获取可用的场景列表
    available_scenarios = list_scenarios()
    
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=available_scenarios,
        help=f"场景名称。可选值：{', '.join(available_scenarios)}"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["efficient", "edl_safe", "edl_robust"],
        help="规划模式"
    )
    
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        default=False,
        help="是否使用真实数据（默认使用 demo 数据）"
    )
    
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="输出 CSV 文件路径（可选）"
    )
    
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="输出 JSON 文件路径（可选）"
    )
    
    args = parser.parse_args()
    
    # ========================================================================
    # 运行规划
    # ========================================================================
    try:
        result = run_single_case(
            scenario=args.scenario,
            mode=args.mode,
            use_real_data=args.use_real_data,
        )
    except Exception as e:
        print(f"Error: Failed to run planning: {e}", file=sys.stderr)
        sys.exit(1)
    
    # ========================================================================
    # 打印摘要
    # ========================================================================
    print_summary(result)
    
    # ========================================================================
    # 导出 CSV
    # ========================================================================
    if args.out_csv:
        try:
            out_csv_path = Path(args.out_csv)
            out_csv_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换为 DataFrame（一行）
            df = pd.DataFrame([result.to_flat_dict()])
            df.to_csv(out_csv_path, index=False)
            print(f"✓ CSV exported to: {out_csv_path}")
        except Exception as e:
            print(f"✗ Failed to export CSV: {e}", file=sys.stderr)
            sys.exit(1)
    
    # ========================================================================
    # 导出 JSON
    # ========================================================================
    if args.out_json:
        try:
            out_json_path = Path(args.out_json)
            out_json_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换为可序列化的字典
            result_dict = result.to_dict()
            result_dict = _convert_to_serializable(result_dict)
            
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
            print(f"✓ JSON exported to: {out_json_path}")
        except Exception as e:
            print(f"✗ Failed to export JSON: {e}", file=sys.stderr)
            sys.exit(1)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()







