#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot scenario results and comparisons from batch runs.

@role: analysis
"""

"""
根据 metrics.csv 绘制成本和风险对比图。
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS = PROJECT_ROOT / "outputs" / "metrics.csv"
OUT_COST = PROJECT_ROOT / "docs" / "metrics_cost.png"
OUT_RISK = PROJECT_ROOT / "docs" / "metrics_risk.png"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False


def plot_metric(df: pd.DataFrame, y_col: str, out_path: Path, ylabel: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    for beta, group in df.groupby("beta"):
        grp = group.sort_values("gamma")
        ax.plot(grp["gamma"], grp[y_col], marker="o", label=f"beta={beta}")
    ax.set_xlabel("gamma")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} 随 gamma 的变化")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="绘制场景指标对比图")
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS), help="指标 CSV 路径")
    parser.add_argument("--out-cost", default=str(OUT_COST), help="成本图路径")
    parser.add_argument("--out-risk", default=str(OUT_RISK), help="风险图路径")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(f"找不到指标文件: {metrics_path}")

    df = pd.read_csv(metrics_path)
    if df.empty:
        raise ValueError("指标数据为空")

    df = df.sort_values(["beta", "gamma"])

    plot_metric(df, "total_cost", Path(args.out_cost), "总成本")
    plot_metric(df, "mean_risk", Path(args.out_risk), "平均风险")

    print(f"[OK] 成本图: {args.out_cost}")
    print(f"[OK] 风险图: {args.out_risk}")


if __name__ == "__main__":
    main()
