from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np

from arcticroute.core.grid import make_demo_grid
from arcticroute.core.env_real import RealEnvLayers
from arcticroute.core.cost import build_cost_from_real_env
from arcticroute.core.astar import plan_route_latlon
from arcticroute.core.analysis import compute_route_cost_breakdown
from arcticroute.core.pareto import (
    ParetoSolution,
    extract_objectives_from_breakdown,
    pareto_front,
    solutions_to_dataframe,
)

# demo start/end consistent with existing demo routing tests (66N,5E)->(78N,150E)
START_LAT, START_LON = 66.0, 5.0
END_LAT, END_LON = 78.0, 150.0


def _make_demo_env(grid, land_mask, seed: int) -> RealEnvLayers:
    rng = np.random.default_rng(seed)
    ny, nx = grid.shape()
    sic = rng.uniform(0.05, 0.85, size=(ny, nx)).astype(float)
    wave = rng.uniform(0.1, 5.0, size=(ny, nx)).astype(float)
    thick = rng.uniform(0.0, 1.2, size=(ny, nx)).astype(float)
    # keep land cells benign; land is blocked by land_mask anyway
    sic[land_mask] = 0.0
    wave[land_mask] = 0.0
    thick[land_mask] = 0.0
    return RealEnvLayers(
        grid=grid,
        sic=sic,
        wave_swh=wave,
        ice_thickness_m=thick,
        land_mask=land_mask,
    )


def _sample_candidates(n: int, seed: int):
    rnd = random.Random(seed)
    cands = []

    # include 3 "named" presets
    presets = [
        ("efficient", dict(ice_penalty=2.0, wave_penalty=1.0, use_edl=False, w_edl=0.0, use_edl_uncertainty=False, edl_uncertainty_weight=0.0)),
        ("edl_safe", dict(ice_penalty=5.0, wave_penalty=2.0, use_edl=True,  w_edl=2.0, use_edl_uncertainty=False, edl_uncertainty_weight=0.0)),
        ("edl_robust", dict(ice_penalty=4.0, wave_penalty=2.0, use_edl=True, w_edl=2.0, use_edl_uncertainty=True, edl_uncertainty_weight=2.0)),
    ]
    cands.extend(presets)

    # random samples
    for i in range(n):
        cands.append(
            (
                f"rand_{i:03d}",
                dict(
                    ice_penalty=rnd.uniform(1.0, 8.0),
                    wave_penalty=rnd.uniform(0.5, 4.0),
                    use_edl=(rnd.random() < 0.7),
                    w_edl=rnd.uniform(0.0, 3.0),
                    use_edl_uncertainty=(rnd.random() < 0.5),
                    edl_uncertainty_weight=rnd.uniform(0.0, 3.0),
                ),
            )
        )
    return cands


def run_pareto_suite(n_random: int = 20, seed: int = 7, output_dir: str = "reports"):
    """
    生成 Pareto 前沿候选解。
    
    Args:
        n_random: 随机候选数量（除 3 个预设外）
        seed: 随机种子
        output_dir: 输出目录
    
    Returns:
        (all_solutions, front_solutions) 元组
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    grid, land_mask = make_demo_grid()
    env = _make_demo_env(grid, land_mask, seed=seed)

    solutions = []
    candidates = _sample_candidates(n_random, seed=seed)

    for key, cfg in candidates:
        cost_field = build_cost_from_real_env(
            grid=grid,
            land_mask=land_mask,
            env=env,
            ice_penalty=float(cfg["ice_penalty"]),
            wave_penalty=float(cfg["wave_penalty"]),
            use_edl=bool(cfg["use_edl"]),
            w_edl=float(cfg["w_edl"]),
            use_edl_uncertainty=bool(cfg["use_edl_uncertainty"]),
            edl_uncertainty_weight=float(cfg["edl_uncertainty_weight"]),
        )

        route = plan_route_latlon(
            cost_field=cost_field,
            start_lat=START_LAT,
            start_lon=START_LON,
            end_lat=END_LAT,
            end_lon=END_LON,
            neighbor8=True,
        )
        if not route:
            continue

        breakdown = compute_route_cost_breakdown(grid, cost_field, route)
        obj = extract_objectives_from_breakdown(breakdown)
        ct = getattr(breakdown, "component_totals", {}) or {}

        solutions.append(
            ParetoSolution(
                key=key,
                objectives=obj,
                route=list(route),
                component_totals=dict(ct),
                meta=dict(cfg),
            )
        )

    fields = ["distance_km", "total_cost", "edl_uncertainty"]
    front = pareto_front(solutions, fields=fields)

    df_all = solutions_to_dataframe(solutions)
    df_front = solutions_to_dataframe(front)

    p_all = outdir / "pareto_solutions.csv"
    p_front = outdir / "pareto_front.csv"
    df_all.to_csv(p_all, index=False)
    df_front.to_csv(p_front, index=False)

    return solutions, front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="number of random candidates (excluding 3 presets)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--pareto-fields", default="distance_km,total_cost,edl_uncertainty", help="comma-separated objective fields (minimize)")
    args = ap.parse_args()

    solutions, front = run_pareto_suite(n_random=args.n, seed=args.seed, output_dir=args.outdir)

    fields = [x.strip() for x in args.pareto_fields.split(",") if x.strip()]
    front = pareto_front(solutions, fields=fields)

    df_all = solutions_to_dataframe(solutions)
    df_front = solutions_to_dataframe(front)

    outdir = Path(args.outdir)
    p_all = outdir / "pareto_solutions.csv"
    p_front = outdir / "pareto_front.csv"

    print(f"[OK] solutions={len(solutions)} -> front={len(front)} fields={fields}")
    print("[OK] wrote:", p_all)
    print("[OK] wrote:", p_front)


if __name__ == "__main__":
    main()
