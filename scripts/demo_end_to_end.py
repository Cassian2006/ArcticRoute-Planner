"""
Phase 13: PolarRoute 端到端 Demo
支持自动选择 planner backend（PolarRoute pipeline/external 或 A* 回退）
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from arcticroute.core.astar import plan_route_latlon
from arcticroute.core.cost import build_demo_cost
from arcticroute.core.grid import make_demo_grid


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def run_planner_selection(
    mode: str,
    pipeline_dir: Optional[str] = None,
    vessel_mesh: Optional[str] = None,
    route_config: Optional[str] = None,
) -> Dict[str, Any]:
    """调用 polarroute_select_and_plan 获取选择结果"""
    cmd = [
        sys.executable,
        "-m",
        "scripts.polarroute_select_and_plan",
        "--mode",
        mode,
        "--out-json",
        "reports/polarroute_selection_temp.json",
    ]
    if pipeline_dir:
        cmd.extend(["--pipeline-dir", pipeline_dir])
    if vessel_mesh:
        cmd.extend(["--vessel-mesh", vessel_mesh])
    if route_config:
        cmd.extend(["--route-config", route_config])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"[WARN] planner selection failed: {result.stderr}")
        # 返回默认 astar
        return {
            "planner_used": "astar",
            "planner_mode": "astar",
            "fallback_reason": "selection_script_failed",
        }

    try:
        return json.loads(result.stdout)
    except Exception as e:
        print(f"[WARN] failed to parse selection result: {e}")
        return {
            "planner_used": "astar",
            "planner_mode": "astar",
            "fallback_reason": "selection_parse_failed",
        }


def plan_with_astar(start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> list:
    """使用 A* 规划路径"""
    grid, land_mask = make_demo_grid()
    cf = build_demo_cost(grid, land_mask)
    path = plan_route_latlon(cf, start_lat, start_lon, end_lat, end_lon)
    return path if path else []


def plan_with_polarroute_pipeline(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    pipeline_dir: str,
) -> list:
    """使用 PolarRoute pipeline 规划路径（占位符）"""
    print(f"[INFO] PolarRoute pipeline mode not fully implemented, using pipeline_dir={pipeline_dir}")
    # 实际应调用 PolarRoute pipeline
    # 这里回退到 A*
    return plan_with_astar(start_lat, start_lon, end_lat, end_lon)


def plan_with_polarroute_external(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    vessel_mesh: str,
    route_config: str,
) -> list:
    """使用 PolarRoute external 规划路径（占位符）"""
    print(f"[INFO] PolarRoute external mode not fully implemented")
    print(f"      vessel_mesh={vessel_mesh}, route_config={route_config}")
    # 实际应调用 PolarRoute CLI
    # 这里回退到 A*
    return plan_with_astar(start_lat, start_lon, end_lat, end_lon)


def export_geojson(path: list, outfile: Path) -> None:
    """导出路径为 GeoJSON"""
    features = []
    if path:
        coords = [[lon, lat] for lat, lon in path]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {"name": "route"},
            }
        )
    geojson = {"type": "FeatureCollection", "features": features}
    outfile.write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")


def export_cost_breakdown(meta: Dict[str, Any], path: list, outfile: Path) -> None:
    """导出成本分解（包含 planner meta）"""
    breakdown = {
        "meta": meta,
        "route_length": len(path),
        "timestamp": _now_iso(),
    }
    outfile.write_text(json.dumps(breakdown, ensure_ascii=False, indent=2), encoding="utf-8")


def export_diagnostics(outfile: Path) -> None:
    """导出诊断信息（占位符）"""
    outfile.write_text("timestamp,metric,value\n", encoding="utf-8")
    outfile.write_text(f"{_now_iso()},demo_run,1\n", encoding="utf-8")


def export_summary(meta: Dict[str, Any], path: list, outfile: Path) -> None:
    """导出摘要"""
    lines = [
        "=" * 60,
        "Phase 13: PolarRoute End-to-End Demo Summary",
        "=" * 60,
        f"Timestamp: {_now_iso()}",
        "",
        "Planner Selection:",
        f"  Requested Mode: {meta.get('requested_mode', 'N/A')}",
        f"  Planner Used: {meta.get('planner_used', 'N/A')}",
        f"  Planner Mode: {meta.get('planner_mode', 'N/A')}",
        f"  Fallback Reason: {meta.get('fallback_reason', 'N/A')}",
        f"  Pipeline Dir: {meta.get('pipeline_dir', 'N/A')}",
        f"  External Vessel Mesh: {meta.get('external_vessel_mesh', 'N/A')}",
        f"  External Route Config: {meta.get('external_route_config', 'N/A')}",
        "",
        "Route Result:",
        f"  Route Length (points): {len(path)}",
        f"  Route Found: {'Yes' if path else 'No'}",
        "",
        "=" * 60,
    ]
    outfile.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 13: PolarRoute End-to-End Demo")
    ap.add_argument("--outdir", default="reports/demo_run_phase13", help="输出目录")
    ap.add_argument(
        "--planner-mode",
        default="auto",
        choices=["auto", "astar", "polarroute_pipeline", "polarroute_external"],
        help="Planner 模式",
    )
    ap.add_argument("--pipeline-dir", default=None, help="PolarRoute pipeline 目录")
    ap.add_argument("--vessel-mesh", default=None, help="PolarRoute vessel mesh 文件")
    ap.add_argument("--route-config", default=None, help="PolarRoute route config 文件")
    ap.add_argument("--start-lat", type=float, default=66.0)
    ap.add_argument("--start-lon", type=float, default=5.0)
    ap.add_argument("--end-lat", type=float, default=78.0)
    ap.add_argument("--end-lon", type=float, default=150.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Running Phase 13 demo with planner_mode={args.planner_mode}")

    # 1. 选择 planner
    meta = run_planner_selection(
        args.planner_mode, args.pipeline_dir, args.vessel_mesh, args.route_config
    )
    print(f"[INFO] Selected planner: {meta.get('planner_used')} (mode={meta.get('planner_mode')})")
    if meta.get("fallback_reason"):
        print(f"[WARN] Fallback reason: {meta.get('fallback_reason')}")

    # 2. 规划路径
    planner_used = meta.get("planner_used", "astar")
    planner_mode = meta.get("planner_mode", "astar")

    if planner_used == "polarroute" and planner_mode == "pipeline":
        path = plan_with_polarroute_pipeline(
            args.start_lat, args.start_lon, args.end_lat, args.end_lon, meta.get("pipeline_dir")
        )
    elif planner_used == "polarroute" and planner_mode == "external":
        path = plan_with_polarroute_external(
            args.start_lat,
            args.start_lon,
            args.end_lat,
            args.end_lon,
            meta.get("external_vessel_mesh"),
            meta.get("external_route_config"),
        )
    else:
        path = plan_with_astar(args.start_lat, args.start_lon, args.end_lat, args.end_lon)

    print(f"[INFO] Route planned: {len(path)} points")

    # 3. 导出结果
    export_geojson(path, outdir / "route.geojson")
    export_cost_breakdown(meta, path, outdir / "cost_breakdown.json")
    export_diagnostics(outdir / "polaris_diagnostics.csv")
    export_summary(meta, path, outdir / "summary.txt")

    print(f"[INFO] Results written to {outdir}")
    print("[INFO] Phase 13 demo completed successfully")


if __name__ == "__main__":
    main()
