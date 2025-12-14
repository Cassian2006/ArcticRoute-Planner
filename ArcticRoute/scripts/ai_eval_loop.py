#!/usr/bin/env python3
"""Utility script: ai eval loop

@role: analysis
"""

"""Automated loop for evaluating AI advisor recommendations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from matplotlib import pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"
CLI_SCRIPT = PROJECT_ROOT / "api" / "cli.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LLM advisor via automated run loop")
    parser.add_argument("--cfg", default="config/runtime.yaml", help="baseline runtime config")
    parser.add_argument("--use-llm", default="false", help="enable LLM (true/false)")
    parser.add_argument("--tag", default=None, help="optional tag for advice and run output")
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def latest_run_report() -> Optional[Path]:
    reports = list(OUTPUTS_DIR.rglob("run_report_*.json"))
    if not reports:
        return None
    return max(reports, key=lambda p: p.stat().st_mtime)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_cli(command: list[str], *, cwd: Optional[Path] = None) -> subprocess.CompletedProcess[str]:
    print("[AI EVAL] Running:", " ".join(command))
    process = subprocess.run(
        command,
        cwd=str(cwd or REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [os.environ.get("PYTHONPATH"), str(REPO_ROOT)]))},
    )
    if process.returncode != 0:
        print(process.stdout)
        print(process.stderr)
    return process


def advise_parameters(cfg: str, use_llm: bool, tag: Optional[str]) -> Tuple[Dict[str, Any], Path]:
    tag_args = ["--tag", tag] if tag else []
    cmd = [
        sys.executable,
        str(CLI_SCRIPT),
        "ai",
        "advise",
        "--cfg",
        cfg,
        "--use-llm",
        "true" if use_llm else "false",
    ]
    cmd.extend(tag_args)
    result = run_cli(cmd)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("advisor CLI failed")
    advice_path = OUTPUTS_DIR / f"advice_{tag if tag else 'advise'}.json"
    if not advice_path.exists():
        # fallback: find latest advice file
        advice_candidates = sorted(OUTPUTS_DIR.glob("advice_*.json"), key=lambda p: p.stat().st_mtime)
        if not advice_candidates:
            raise FileNotFoundError("advice output not found")
        advice_path = advice_candidates[-1]
    payload = load_json(advice_path)
    return payload["advice"], advice_path


def run_planning(cfg: str, overrides: Dict[str, Any], *, run_tag: Optional[str]) -> Path:
    plan_cmd = [
        sys.executable,
        str(CLI_SCRIPT),
        "plan",
        "--cfg",
        cfg,
    ]
    if run_tag:
        plan_cmd.extend(["--tag", run_tag])

    for key, value in overrides.items():
        cli_key = f"--{key.replace('_', '-')}"
        plan_cmd.extend([cli_key, str(value)])

    result = run_cli(plan_cmd)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("plan CLI failed")

    reports = list(OUTPUTS_DIR.rglob("run_report_*.json"))
    if not reports:
        raise FileNotFoundError("no run_report found after planning")
    latest = max(reports, key=lambda p: p.stat().st_mtime)
    return latest


def metrics_from_report(report: Dict[str, Any]) -> Dict[str, float]:
    corridor_stats = report.get("corridor_stats") or {}
    accident_stats = report.get("accident_stats") or {}
    nearest_accident = report.get("nearest_accident_km") or {}
    return {
        "total_cost": float(report.get("total_cost", 0.0)),
        "mean_risk": float(report.get("mean_risk", report.get("risk_mean", 0.0))),
        "max_risk": float(report.get("max_risk", 0.0)),
        "geodesic_length_km": float(report.get("geodesic_length_m", 0.0)) / 1000.0,
        "corridor_coverage": float(corridor_stats.get("coverage", 0.0)),
        "corridor_mean": float(corridor_stats.get("mean", 0.0)),
        "accident_mean": float(accident_stats.get("mean", 0.0)),
        "nearest_accident_min_km": float(nearest_accident.get("min", 0.0)),
        "nearest_accident_mean_km": float(nearest_accident.get("mean", nearest_accident.get("min", 0.0))),
    }


def build_comparison(baseline: Dict[str, float], advised: Dict[str, float]) -> pd.DataFrame:
    data = []
    for metric in baseline:
        data.append(
            {
                "metric": metric,
                "baseline": baseline[metric],
                "advised": advised[metric],
                "delta": advised[metric] - baseline[metric],
            }
        )
    df = pd.DataFrame(data)
    return df


def save_comparison(df: pd.DataFrame) -> Path:
    csv_path = OUTPUTS_DIR / "ai_eval.csv"
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return csv_path


def plot_comparison(df: pd.DataFrame) -> Path:
    plot_path = DOCS_DIR / "ai_eval_bar.png"
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = ["total_cost", "mean_risk", "geodesic_length_km", "nearest_accident_min_km"]
    subset = df[df["metric"].isin(metrics)]

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    idx = range(len(subset))
    ax.bar([i - width / 2 for i in idx], subset["baseline"], width=width, label="Baseline")
    ax.bar([i + width / 2 for i in idx], subset["advised"], width=width, label="Advised")
    ax.set_xticks(list(idx))
    ax.set_xticklabels(subset["metric"], rotation=15)
    ax.set_ylabel("Value")
    ax.set_title("AI Evaluation Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def summary(df: pd.DataFrame) -> str:
    lines = []
    for _, row in df.iterrows():
        metric = row["metric"]
        delta = row["delta"]
        if abs(delta) < 1e-6:
            continue
        direction = "下降" if delta < 0 else "上升"
        lines.append(f"{metric}: {direction} {abs(delta):.3f}")
    return "; ".join(lines) if lines else "关键指标变化不明显。"


def main() -> int:
    args = parse_args()
    use_llm = parse_bool(args.use_llm)

    baseline_report_path = latest_run_report()
    if not baseline_report_path:
        print("[AI EVAL] 未找到现成基线运行报告，正在执行一次基线规划...")
        baseline_tag = datetime.utcnow().strftime("baseline_%Y%m%d_%H%M%S")
        baseline_report_path = run_planning(args.cfg, {}, run_tag=baseline_tag)
        print("[AI EVAL] 基线规划完成：", baseline_report_path.name)
    baseline_report = load_json(baseline_report_path)
    baseline_metrics = metrics_from_report(baseline_report)

    tag_suffix = args.tag or datetime.utcnow().strftime("ai_eval_%Y%m%d_%H%M%S")
    advice, advice_path = advise_parameters(args.cfg, use_llm, tag_suffix)

    plan_overrides = {key: advice[key] for key in ("beta", "gamma", "p", "beta_a") if key in advice}

    run_tag = f"ai_eval_{tag_suffix}"
    advised_report_path = run_planning(args.cfg, plan_overrides, run_tag=run_tag)

    # Rename advised report with deterministic name
    eval_report_path = OUTPUTS_DIR / f"run_report_ai_eval_{datetime.utcnow():%Y%m%d_%H%M%S}.json"
    eval_report_path.write_text(advised_report_path.read_text(encoding="utf-8"), encoding="utf-8")
    advised_report_path = eval_report_path

    advised_report = load_json(advised_report_path)
    advised_metrics = metrics_from_report(advised_report)

    df = build_comparison(baseline_metrics, advised_metrics)
    csv_path = save_comparison(df)
    plot_path = plot_comparison(df)

    summary_text = summary(df)
    print("[AI EVAL] 基线报告：", baseline_report_path.name)
    print("[AI EVAL] 新报告：", advised_report_path.name)
    print("[AI EVAL] 建议文件：", advice_path.name)
    print("[AI EVAL] 对比表：", csv_path)
    print("[AI EVAL] 对比图：", plot_path)
    print("[AI EVAL] 结论：", summary_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
CLI_SCRIPT = PROJECT_ROOT / "api" / "cli.py"
