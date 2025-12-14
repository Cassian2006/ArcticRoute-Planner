#!/usr/bin/env python3
"""Calibration experiments for accident risk modeling (exploratory).

@role: analysis
"""

"""
ACC-5 | 事故风险 beta_a 扫描与校准辅助。

- 读取基线配置并遍历指定 beta_a 列表
- 调用 api.cli plan 生成路线与报告（可复用既有结果）
- 汇总关键指标到 CSV 并绘制折线图
- 基于结果输出推荐 beta_a 区间的中文摘要
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "acc_calib"
DEFAULT_CSV_PATH = PROJECT_ROOT / "outputs" / "acc_calib.csv"
DEFAULT_PLOT_PATH = PROJECT_ROOT / "docs" / "acc_calib.png"


@dataclass
class CalibrationResult:
    beta_a: float
    total_cost: float
    mean_risk: float
    fuel_proxy: float
    nearest_min: float | None
    nearest_mean: float | None
    nearest_max: float | None


def parse_beta_list(value: str) -> List[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("beta_a 列表不能为空")
    betas: List[float] = []
    for item in items:
        try:
            betas.append(float(item))
        except ValueError as exc:  # pragma: no cover - CLI 输入校验
            raise ValueError(f"无法解析 beta_a 数值: '{item}'") from exc
    return betas


def sanitize_tag(beta: float) -> str:
    if beta == int(beta):
        suffix = f"{int(beta)}"
    else:
        suffix = f"{beta:.3f}".rstrip("0").rstrip(".")
    suffix = suffix.replace("-", "neg").replace(".", "p")
    return f"acc_calib_{suffix}"


def run_plan_once(
    cfg: Path,
    accident_density: Path,
    acc_mode: str,
    beta_a: float,
    output_dir: Path,
    force: bool,
) -> Path:
    tag = sanitize_tag(beta_a)
    report_path = output_dir / f"run_report_{tag}.json"
    if report_path.exists() and not force:
        print(f"[SKIP] {tag} 已存在报告，跳过重新规划（use --force 重新执行）")
        return report_path

    output_dir.mkdir(parents=True, exist_ok=True)
    cli_args = [
        sys.executable,
        "-m",
        "api.cli",
        "plan",
        "--cfg",
        str(cfg),
        "--accident-density",
        str(accident_density),
        "--acc-mode",
        acc_mode,
        "--beta-a",
        f"{beta_a}",
        "--tag",
        tag,
        "--output-dir",
        str(output_dir),
    ]

    print(f"[RUN] 执行 beta_a={beta_a:.3f} 路线规划…")
    completed = subprocess.run(
        cli_args,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise RuntimeError(f"路线规划失败（beta_a={beta_a}）")
    return report_path


def extract_metrics(report_path: Path, beta_a: float) -> CalibrationResult:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    total_cost = float(data.get("total_cost", math.nan))
    mean_risk = float(data.get("mean_risk", math.nan))
    fuel_proxy = float(data.get("fuel_proxy", math.nan))
    nearest = data.get("nearest_accident_km") or {}
    nearest_min = nearest.get("min")
    nearest_mean = nearest.get("mean")
    nearest_max = nearest.get("max")
    return CalibrationResult(
        beta_a=beta_a,
        total_cost=total_cost,
        mean_risk=mean_risk,
        fuel_proxy=fuel_proxy,
        nearest_min=float(nearest_min) if nearest_min is not None else None,
        nearest_mean=float(nearest_mean) if nearest_mean is not None else None,
        nearest_max=float(nearest_max) if nearest_max is not None else None,
    )


def build_dataframe(results: Iterable[CalibrationResult]) -> pd.DataFrame:
    records = []
    for item in results:
        records.append(
            {
                "beta_a": item.beta_a,
                "total_cost": item.total_cost,
                "mean_risk": item.mean_risk,
                "fuel_proxy": item.fuel_proxy,
                "nearest_min": item.nearest_min,
                "nearest_mean": item.nearest_mean,
                "nearest_max": item.nearest_max,
            }
        )
    df = pd.DataFrame(records)
    df.sort_values("beta_a", inplace=True)
    return df


def plot_results(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    ax = axes[0, 0]
    ax.plot(df["beta_a"], df["total_cost"] / 1e6, marker="o", color="#1f77b4")
    ax.set_ylabel("Total Cost (×10⁶)")
    ax.grid(True, linestyle="--", alpha=0.4)

    ax = axes[0, 1]
    ax.plot(df["beta_a"], df["mean_risk"], marker="o", color="#ff7f0e")
    ax.set_ylabel("Mean Risk")
    ax.grid(True, linestyle="--", alpha=0.4)

    ax = axes[1, 0]
    ax.plot(df["beta_a"], df["fuel_proxy"] / 1000, marker="o", color="#2ca02c")
    ax.set_xlabel("beta_a")
    ax.set_ylabel("Fuel Proxy (km-equiv)")
    ax.grid(True, linestyle="--", alpha=0.4)

    ax = axes[1, 1]
    if df["nearest_mean"].notna().any():
        ax.plot(df["beta_a"], df["nearest_min"], marker="o", label="min", color="#9467bd")
        ax.plot(df["beta_a"], df["nearest_mean"], marker="o", label="mean", color="#8c564b")
        ax.plot(df["beta_a"], df["nearest_max"], marker="o", label="max", color="#17becf")
        ax.legend()
    ax.set_xlabel("beta_a")
    ax.set_ylabel("Nearest Accident Distance (km)")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Accident Risk Calibration (beta_a sweep)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] 保存折线图: {output_path}")


def format_summary(df: pd.DataFrame) -> str:
    df = df.dropna(subset=["total_cost", "mean_risk"])
    if df.empty:
        return "未能获取有效的规划结果，请检查输入。"

    baseline = df.loc[df["beta_a"] == 0].head(1)
    baseline_cost = float(baseline["total_cost"].iloc[0]) if not baseline.empty else None
    baseline_beta = 0.0 if not baseline.empty else None

    best_idx = df["total_cost"].idxmin()
    best_row = df.loc[best_idx]
    best_beta = float(best_row["beta_a"])
    best_cost = float(best_row["total_cost"])
    best_risk = float(best_row["mean_risk"])
    best_nearest = float(best_row["nearest_mean"]) if not math.isnan(best_row["nearest_mean"]) else None

    # Identify near-optimal band (within 1% cost increase from min)
    tolerance = 0.01
    mask = df["total_cost"] <= best_cost * (1 + tolerance)
    candidates = df.loc[mask, "beta_a"].tolist()
    if candidates:
        band_min = min(candidates)
        band_max = max(candidates)
        band_text = f"{band_min:.2f}–{band_max:.2f}"
    else:
        band_text = f"{best_beta:.2f}"

    if baseline_cost is not None:
        cost_delta = (best_cost - baseline_cost) / baseline_cost
        trend = "下降" if cost_delta < 0 else "上升"
        cost_change = f"{abs(cost_delta) * 100:.1f}%"
        baseline_text = (
            f"相较 beta_a={baseline_beta:.1f}，总成本{trend}{cost_change}"
            if abs(cost_delta) > 1e-4
            else "与 beta_a=0 成本接近"
        )
    else:
        baseline_text = "基线 beta_a=0 不可用"

    nearest_text = (
        f"经过事故均距离约 {best_nearest:.1f} km"
        if best_nearest is not None
        else "事故距离指标暂无数据"
    )

    return (
        f"推荐优先考虑 beta_a≈{best_beta:.2f}（或 {band_text} 区间），"
        f"{baseline_text}，平均风险 {best_risk:.3f}，{nearest_text}。"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="ACC-5 accident calibration sweep")
    parser.add_argument("--cfg", type=Path, required=True, help="runtime 配置路径")
    parser.add_argument("--accident-density", type=Path, required=True, help="事故密度文件")
    parser.add_argument(
        "--acc-mode",
        choices=["static", "time"],
        default="static",
        help="事故密度模式",
    )
    parser.add_argument(
        "--beta-a-list",
        required=True,
        help="beta_a 值列表，逗号分隔，例如 \"0,0.1,0.3\"",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="规划结果输出目录")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="指标 CSV 输出路径")
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT_PATH, help="指标图输出路径")
    parser.add_argument("--force", action="store_true", help="强制重新执行规划")
    args = parser.parse_args()

    def resolve(path: Path, search_roots: List[Path]) -> Path:
        if path.is_absolute():
            return path
        for root in search_roots:
            candidate = (root / path).resolve()
            if candidate.exists():
                return candidate
        # fallback: first root even if不存在 -> 由后续流程报错
        return (search_roots[0] / path).resolve()

    cfg_path = resolve(args.cfg, [PROJECT_ROOT, REPO_ROOT])
    accident_path = resolve(args.accident_density, [REPO_ROOT, PROJECT_ROOT])
    output_dir = resolve(args.output_dir, [PROJECT_ROOT, REPO_ROOT])
    csv_path = resolve(args.csv, [PROJECT_ROOT, REPO_ROOT])
    plot_path = resolve(args.plot, [PROJECT_ROOT, REPO_ROOT])

    betas = parse_beta_list(args.beta_a_list)
    results: List[CalibrationResult] = []

    for beta_a in betas:
        report_path = run_plan_once(
            cfg=cfg_path,
            accident_density=accident_path,
            acc_mode=args.acc_mode,
            beta_a=beta_a,
            output_dir=output_dir,
            force=args.force,
        )
        result = extract_metrics(report_path, beta_a)
        results.append(result)

    df = build_dataframe(results)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[OK] 保存指标表格: {csv_path}")

    plot_results(df, plot_path)

    summary = format_summary(df)
    print(f"[摘要] {summary}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
