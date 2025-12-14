# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Evidential + Robust 规划一站式 Demo 脚本

用法（仓库根目录下）：
  - 模块方式（推荐）：
      python -m ArcticRoute.scripts.run_evidential_robust_demo \
        --ym 202412 \
        --start-lat 69.0 --start-lon 33.0 \
        --end-lat 70.5 --end-lon 170.0 \
        --profile balanced \
        --agg cvar --alpha 0.9 --fusion evidential

  - 直接运行脚本（需要 Python 工作目录在仓库根）：
      python ArcticRoute/scripts/run_evidential_robust_demo.py
"""

import argparse
import sys
from typing import Optional
from pathlib import Path

# 统一路径，确保可从任意工作目录运行（先将项目父目录加入 sys.path，再尝试 _modpath）
PROJECT_PARENT = Path(__file__).resolve().parents[1].parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))
try:
    from ArcticRoute.scripts._modpath import ensure_path as _ensure_path
except Exception:
    _ensure_path = None  # type: ignore
if _ensure_path is not None:
    _ensure_path()

from ArcticRoute.core import planner_service as ps
from ArcticRoute.core.planner_service import (
    RobustPlannerConfig,
    run_planning_pipeline_evidential_robust,
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ym", default="202412", help="年月，例如 202412")
    ap.add_argument("--start-lat", type=float, default=69.0)
    ap.add_argument("--start-lon", type=float, default=33.0)
    ap.add_argument("--end-lat", type=float, default=70.5)
    ap.add_argument("--end-lon", type=float, default=140.0)
    ap.add_argument("--profile", default="balanced", help="策略配置名（balanced/safe/efficient 等）")
    ap.add_argument("--agg", default="cvar", choices=["mean", "quantile", "cvar"], help="不确定性聚合模式")
    ap.add_argument("--alpha", type=float, default=0.9, help="quantile/ES 的 alpha")
    ap.add_argument("--fusion", default="evidential", help="融合模式（evidential 等）")
    ap.add_argument("--heuristic", default="euclidean", choices=["manhattan", "euclidean", "octile"]) 
    ap.add_argument("--no-diag", action="store_true", help="禁用斜向移动")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    cfg = RobustPlannerConfig(
        risk_agg_mode=str(args.agg),
        risk_agg_alpha=float(args.alpha),
        fusion_mode=str(args.fusion),
        allow_diagonal=not bool(args.no_diag),
        heuristic=str(args.heuristic),
    )

    print("[DEMO] Robust/Evidential Planner")
    print(f"  ym={args.ym}, profile={args.profile}")
    print(f"  start=({args.start_lat:.3f},{args.start_lon:.3f}), end=({args.end_lat:.3f},{args.end_lon:.3f})")
    print(f"  fusion={cfg.fusion_mode}, agg={cfg.risk_agg_mode}, alpha={cfg.risk_agg_alpha}")

    res = run_planning_pipeline_evidential_robust(
        ym=str(args.ym),
        start_lat=float(args.start_lat),
        start_lon=float(args.start_lon),
        end_lat=float(args.end_lat),
        end_lon=float(args.end_lon),
        profile_name=str(args.profile),
        robust_cfg=cfg,
    )

    env_ctx = res.get("env_ctx")
    route = res.get("route")
    summary = res.get("summary") or {}
    robust_meta = res.get("robust_meta") or {}

    print("\n[RESULT]")
    print(f"  reachable={bool(getattr(route, 'reachable', False))}")
    print(f"  distance_km={summary.get('distance_km', 0.0)}")
    print(f"  total_cost={summary.get('cost_sum', 0.0)}")
    print("  robust_meta:")
    for k in ("fusion_mode", "fusion_mode_effective", "risk_agg_mode", "cost_risk_agg_mode_effective", "risk_agg_alpha", "profile_name", "ym"):
        if k in robust_meta:
            print(f"    - {k}: {robust_meta[k]}")

    if getattr(route, "reachable", False):
        prof = res.get("profile") or {}
        rt = prof.get("risk_total")
        if rt is not None and len(rt) > 0:
            print(f"  profile.risk_total: first={float(rt[0]):.3f}, last={float(rt[-1]):.3f}, n={len(rt)}")
        else:
            print("  profile: not available or empty")
    else:
        print("  route not reachable; no profile.")

    # 额外：打印成本分解
    cb = res.get("cost_breakdown") or {}
    if cb:
        print("\n  cost_breakdown:")
        print(f"    - distance_km: {cb.get('distance_km')}")
        comps = (cb.get("risk_components") or {})
        if comps:
            for name in ["ice", "wave", "accident", "congestion", "prior"]:
                if name in comps:
                    print(f"    - {name}: {comps[name]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

