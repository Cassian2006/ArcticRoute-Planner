#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch-run minimal A* scenarios and collect metrics CSV.

@role: analysis
"""

"""
批量运行 A* 场景，收集指标输出 CSV。
"""

import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROUTE_SCRIPT = PROJECT_ROOT / "scripts" / "route_astar_min.py"
SCENARIO_CONFIG = PROJECT_ROOT / "config" / "scenarios.yaml"
METRICS_CSV = PROJECT_ROOT / "outputs" / "metrics.csv"


def load_scenarios(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"找不到场景配置: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "scenarios" not in data:
        raise ValueError("场景配置格式错误，需包含 scenarios 列表")
    return data


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def run_route(command: List[str]) -> float:
    start = time.perf_counter()
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    duration = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(f"命令失败: {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return duration


def main():
    config = load_scenarios(SCENARIO_CONFIG)
    defaults = config.get("defaults", {})
    scenarios = config["scenarios"]

    env_nc = defaults.get("env_nc")
    var = defaults.get("var", "risk_env")
    p_default = float(defaults.get("p", 1.0))
    tidx_default = int(defaults.get("tidx", 0))
    start_default = defaults.get("start")
    goal_default = defaults.get("goal")
    corridor_default = defaults.get("corridor")
    output_dir = defaults.get("output_dir", "outputs/scenarios")

    if not env_nc or not start_default or not goal_default:
        raise ValueError("defaults 中必须包含 env_nc、start、goal")

    env_path = (PROJECT_ROOT / env_nc) if not Path(env_nc).is_absolute() else Path(env_nc)
    corridor_path = None
    if corridor_default:
        if Path(corridor_default).is_absolute():
            corridor_path = Path(corridor_default)
        else:
            corridor_path = PROJECT_ROOT / corridor_default

    ensure_dir(PROJECT_ROOT / "outputs")
    ensure_dir(PROJECT_ROOT / output_dir)

    metrics_rows: List[Dict] = []

    for scenario in scenarios:
        name = scenario.get("name")
        if not name:
            raise ValueError("每个场景需要 name")
        beta = float(scenario.get("beta", 3.0))
        gamma = float(scenario.get("gamma", 0.0))
        p_value = float(scenario.get("p", p_default))
        tidx = int(scenario.get("tidx", tidx_default))
        start = scenario.get("start", start_default)
        goal = scenario.get("goal", goal_default)
        corridor = scenario.get("corridor", corridor_path)

        geojson_out = PROJECT_ROOT / output_dir / f"{name}.geojson"
        png_out = PROJECT_ROOT / output_dir / f"{name}.png"

        cmd = [
            sys.executable,
            str(ROUTE_SCRIPT),
            "--nc",
            str(env_path),
            "--var",
            var,
            "--tidx",
            str(tidx),
            "--start",
            start,
            "--goal",
            goal,
            "--beta",
            str(beta),
            "--p",
            str(p_value),
            "--gamma",
            str(gamma),
            "--geojson",
            str(geojson_out),
            "--png",
            str(png_out),
        ]
        if corridor and Path(corridor).exists():
            cmd.extend(["--corridor", str(corridor)])

        duration = run_route(cmd)

        data = json.loads(geojson_out.read_text(encoding="utf-8"))
        props = data["features"][0]["properties"]

        metrics_rows.append(
            dict(
                name=name,
                beta=beta,
                gamma=gamma,
                p=p_value,
                tidx=tidx,
                total_cost=props.get("total_cost"),
                geodesic_length_m=float(props.get("path_length_km", 0.0)) * 1000.0,
                node_count=props.get("node_count"),
                mean_risk=props.get("risk_mean"),
                max_risk=props.get("risk_max"),
                corridor_coverage=props.get("corridor_high_ratio"),
                corridor_mean=props.get("corridor_mean"),
                corridor_used=props.get("corridor_used"),
                elapsed_sec=duration,
                geojson=str(geojson_out.relative_to(PROJECT_ROOT)),
                png=str(png_out.relative_to(PROJECT_ROOT)),
            )
        )

    # 写 CSV
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "beta",
        "gamma",
        "p",
        "tidx",
        "total_cost",
        "geodesic_length_m",
        "node_count",
        "mean_risk",
        "max_risk",
        "corridor_mean",
        "corridor_coverage",
        "corridor_used",
        "elapsed_sec",
        "geojson",
        "png",
    ]
    with METRICS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)

    print(f"[OK] 场景运行完成，结果写入 {METRICS_CSV}")


if __name__ == "__main__":
    main()
