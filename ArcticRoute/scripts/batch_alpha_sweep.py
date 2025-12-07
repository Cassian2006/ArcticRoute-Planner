#!/usr/bin/env python3
"""Batch-run Planner with multiple alpha/weights to compare outcomes.

@role: analysis
"""

"""Run batch planner sweeps over multiple alpha_ice values."""
from __future__ import annotations

import argparse
import os
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt

# --- bootstrap imports ---
try:
    from ArcticRoute.scripts._modpath import ensure_path, get_cli_mod
except ModuleNotFoundError:
    HERE = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(HERE, "..", "..")),
        os.path.abspath(os.path.join(HERE, "..")),
        HERE,
    ]
    for candidate in candidates:
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    from scripts._modpath import ensure_path, get_cli_mod  # type: ignore[import]

ensure_path()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = PROJECT_ROOT / "config" / "runtime.yaml"
DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"
DEFAULT_DOCS = PROJECT_ROOT / "docs"
DEFAULT_ALPHAS: Sequence[float] = (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)


def _parse_alphas(value: str) -> List[float]:
    tokens = [item.strip() for item in value.split(",") if item.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("alpha list cannot be empty")
    result: List[float] = []
    for token in tokens:
        try:
            alpha = float(token)
        except ValueError as err:  # pragma: no cover - arg parsing
            raise argparse.ArgumentTypeError(f"invalid alpha value: {token}") from err
        if not (0.0 < alpha <= 1.0):
            raise argparse.ArgumentTypeError(f"alpha must be in (0, 1]: {token}")
        result.append(alpha)
    return result


def _format_tag(prefix: str, alpha: float) -> str:
    sanitized = f"{alpha:.2f}".replace(".", "p")
    return f"{prefix}_{sanitized}"


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(rows: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["alpha_ice", "total_cost", "mean_risk", "max_risk", "distance", "elapsed_s"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_series(alphas: Sequence[float], values: Sequence[float], output: Path, ylabel: str, title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alphas, values, marker="o", color="#1f77b4")
    ax.set_xlabel("alpha_ice")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch-run planner for multiple alpha_ice values.")
    parser.add_argument("--cfg", default=str(DEFAULT_CFG), help="Runtime config file (default: config/runtime.yaml)")
    parser.add_argument("--tidx", type=int, default=0, help="Time index (default: 0)")
    parser.add_argument("--gamma", type=float, default=0.0, help="Gamma override (default: 0)")
    parser.add_argument("--alphas", type=_parse_alphas, help="Comma-separated alpha list")
    parser.add_argument("--tag-prefix", default="sweep_a", help="Tag prefix for planner runs (default: sweep_a)")
    args = parser.parse_args(argv)

    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        parser.error(f"Config file not found: {cfg_path}")

    alphas = list(args.alphas or DEFAULT_ALPHAS)
    results: List[dict] = []
    best_entry: dict | None = None

    cli_module = get_cli_mod()

    for alpha in alphas:
        tag = _format_tag(args.tag_prefix, alpha)
        cmd = [
            sys.executable,
            "-m",
            cli_module,
            "plan",
            "--cfg",
            str(cfg_path),
            "--predictor",
            "cv_sat",
            "--alpha-ice",
            str(alpha),
            "--gamma",
            str(args.gamma),
            "--tidx",
            str(args.tidx),
            "--tag",
            tag,
        ]
        print(f"[SWEEP] Running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if proc.returncode != 0:
            print(f"[SWEEP] Planner failed for alpha_ice={alpha:.2f} (exit code {proc.returncode})")
            return proc.returncode

        report_path = DEFAULT_OUTPUTS / f"run_report_{tag}.json"
        if not report_path.exists():
            print(f"[SWEEP] Missing run report: {report_path}")
            return 1
        report = _load_report(report_path)

        total_cost = float(report.get("total_cost", math.nan))
        mean_risk = float(report.get("mean_risk", math.nan))
        max_risk = float(report.get("max_risk", math.nan))
        distance = float(report.get("geodesic_length_m", math.nan)) / 1000.0
        elapsed = float(report.get("elapsed_s", math.nan))

        print(
            "[SWEEP] alpha_ice={alpha:.2f} -> total_cost={cost:.3f}, mean_risk={mean:.4f}, "
            "max_risk={max_:.4f}, distance_km={distance:.3f}, elapsed_s={elapsed:.2f}".format(
                alpha=alpha,
                cost=total_cost,
                mean=mean_risk,
                max_=max_risk,
                distance=distance,
                elapsed=elapsed,
            )
        )

        entry = {
            "alpha_ice": alpha,
            "total_cost": total_cost,
            "mean_risk": mean_risk,
            "max_risk": max_risk,
            "distance": distance,
            "elapsed_s": elapsed,
            "tag": tag,
        }
        results.append(entry)
        if best_entry is None or total_cost < best_entry["total_cost"]:
            best_entry = entry

    if not results:
        print("[SWEEP] No runs executed.")
        return 0

    results.sort(key=lambda item: item["alpha_ice"])

    csv_rows = [
        {
            "alpha_ice": item["alpha_ice"],
            "total_cost": item["total_cost"],
            "mean_risk": item["mean_risk"],
            "max_risk": item["max_risk"],
            "distance": item["distance"],
            "elapsed_s": item["elapsed_s"],
        }
        for item in results
    ]
    csv_path = DEFAULT_OUTPUTS / "alpha_sweep.csv"
    _write_csv(csv_rows, csv_path)
    print(f"[SWEEP] Aggregated metrics written to {csv_path}")

    alphas_series = [item["alpha_ice"] for item in results]
    total_cost_series = [item["total_cost"] for item in results]
    mean_risk_series = [item["mean_risk"] for item in results]

    cost_plot = DEFAULT_DOCS / "alpha_sweep_cost.png"
    _plot_series(alphas_series, total_cost_series, cost_plot, "Total cost", "Total Cost vs alpha_ice")
    print(f"[SWEEP] Cost trend chart saved to {cost_plot}")

    risk_plot = DEFAULT_DOCS / "alpha_sweep_risk.png"
    _plot_series(alphas_series, mean_risk_series, risk_plot, "Mean risk", "Mean Risk vs alpha_ice")
    print(f"[SWEEP] Risk trend chart saved to {risk_plot}")

    if best_entry is not None:
        print(
            "[SWEEP] Best alpha_ice={alpha:.2f} (tag={tag}) with total_cost={cost:.3f}, mean_risk={mean:.4f}".format(
                alpha=best_entry["alpha_ice"],
                tag=best_entry["tag"],
                cost=best_entry["total_cost"],
                mean=best_entry["mean_risk"],
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
