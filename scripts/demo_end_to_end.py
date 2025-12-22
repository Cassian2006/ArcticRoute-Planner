"""
端到端演示脚本。

生成完整的规划输出，包括：
- route.geojson: 路线 GeoJSON
- cost_breakdown.json: 成本分解
- polaris_diagnostics.csv: POLARIS 诊断（如果启用）
- summary.txt: 摘要信息
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

from arcticroute.core.astar import plan_route_latlon
from arcticroute.core.cost import build_demo_cost
from arcticroute.core.grid import make_demo_grid
from arcticroute.core.analysis import compute_route_cost_breakdown


def main():
    parser = argparse.ArgumentParser(description="端到端演示脚本")
    parser.add_argument("--outdir", type=str, default="reports/demo_run", help="输出目录")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[DEMO] 输出目录: {outdir}")

    # 1. 创建网格和成本场
    grid, land_mask = make_demo_grid()
    cost_field = build_demo_cost(grid, land_mask)

    # 2. 规划路线
    start_lat, start_lon = 66.0, 5.0
    end_lat, end_lon = 78.0, 150.0

    print(f"[DEMO] 规划路线: ({start_lat}, {start_lon}) -> ({end_lat}, {end_lon})")
    
    path = plan_route_latlon(cost_field, start_lat, start_lon, end_lat, end_lon)

    if not path:
        print("[DEMO] 路线不可达")
        return

    print(f"[DEMO] 路线长度: {len(path)} 个点")

    # 3. 计算成本分解
    breakdown = compute_route_cost_breakdown(grid, cost_field, path)

    # 4. 生成 route.geojson
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon, lat] for lat, lon in path]
                },
                "properties": {
                    "name": "demo_route",
                    "total_cost": float(breakdown.total_cost),
                    "distance_km": float(breakdown.s_km[-1]) if breakdown.s_km else 0.0,
                }
            }
        ]
    }

    route_file = outdir / "route.geojson"
    with open(route_file, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)
    print(f"[DEMO] 已生成: {route_file}")

    # 5. 生成 cost_breakdown.json
    breakdown_data = {
        "total_cost": float(breakdown.total_cost),
        "component_totals": {k: float(v) for k, v in breakdown.component_totals.items()},
        "component_fractions": {k: float(v) for k, v in breakdown.component_fractions.items()},
        "meta": {
            "planner_used": "astar",
            "planner_mode": "demo",
            "timestamp": datetime.utcnow().isoformat(),
        }
    }

    breakdown_file = outdir / "cost_breakdown.json"
    with open(breakdown_file, "w", encoding="utf-8") as f:
        json.dump(breakdown_data, f, indent=2, ensure_ascii=False)
    print(f"[DEMO] 已生成: {breakdown_file}")

    # 6. 生成 polaris_diagnostics.csv（占位）
    polaris_file = outdir / "polaris_diagnostics.csv"
    with open(polaris_file, "w", encoding="utf-8", newline="") as f:
        f.write("segment_id,rule_triggered,severity,message\n")
        f.write("0,none,info,POLARIS rules not enabled in demo mode\n")
    print(f"[DEMO] 已生成: {polaris_file}")

    # 7. 生成 summary.txt
    summary_file = outdir / "summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=== ArcticRoute Demo End-to-End Summary ===\n\n")
        f.write(f"Timestamp: {datetime.utcnow().isoformat()}\n")
        f.write(f"Planner Used: astar\n")
        f.write(f"Planner Mode: demo\n\n")
        f.write(f"Route Points: {len(path)}\n")
        f.write(f"Total Cost: {breakdown.total_cost:.2f}\n")
        f.write(f"Distance: {breakdown.s_km[-1]:.2f} km\n\n" if breakdown.s_km else "Distance: N/A\n\n")
        f.write("Cost Components:\n")
        for comp, value in breakdown.component_totals.items():
            fraction = breakdown.component_fractions.get(comp, 0.0)
            f.write(f"  - {comp}: {value:.2f} ({fraction*100:.1f}%)\n")
        f.write("\nOutput Files:\n")
        f.write(f"  - route.geojson\n")
        f.write(f"  - cost_breakdown.json\n")
        f.write(f"  - polaris_diagnostics.csv\n")
        f.write(f"  - summary.txt\n")
    print(f"[DEMO] 已生成: {summary_file}")

    print(f"\n[DEMO] 完成！所有文件已生成到: {outdir}")


if __name__ == "__main__":
    main()
