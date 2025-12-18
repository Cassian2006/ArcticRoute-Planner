"""
端到端演示脚本。

执行完整的路由规划流程，产出：
- route.geojson: 路由几何
- cost_breakdown.json: 成本分解
- polaris_diagnostics.csv: Polaris 诊断数据
- summary.txt: 总结报告

内置 nextsim 可用优先、不可用回退逻辑。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arcticroute.core.astar import plan_route_latlon
from arcticroute.core.cost import build_demo_cost
from arcticroute.core.grid import make_demo_grid


def load_cmems_strategy() -> dict | None:
    """加载 CMEMS 策略（如果存在）。"""
    strategy_path = Path("reports/cmems_strategy.json")
    if strategy_path.exists():
        return json.loads(strategy_path.read_text(encoding="utf-8"))
    return None


def generate_route_geojson(path: list[tuple[float, float]], output_path: Path) -> None:
    """生成路由 GeoJSON。"""
    geojson = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[lon, lat] for lat, lon in path],
        },
        "properties": {
            "route_type": "demo",
            "num_points": len(path),
        },
    }
    output_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")


def generate_cost_breakdown(path: list[tuple[float, float]], output_path: Path) -> None:
    """生成成本分解 JSON。"""
    breakdown = {
        "total_distance_km": len(path) * 10.0,  # 模拟
        "total_time_hours": len(path) * 0.5,
        "fuel_cost_usd": len(path) * 100.0,
        "ice_penalty": 0.15,
        "components": {
            "base_distance": len(path) * 8.0,
            "ice_resistance": len(path) * 2.0,
        },
    }
    output_path.write_text(json.dumps(breakdown, indent=2), encoding="utf-8")


def generate_polaris_diagnostics(output_path: Path) -> None:
    """生成 Polaris 诊断 CSV。"""
    csv_content = "step,lat,lon,ice_conc,speed_kts,fuel_rate\n"
    csv_content += "0,66.0,5.0,0.1,12.5,2.3\n"
    csv_content += "1,68.0,10.0,0.3,10.2,2.8\n"
    csv_content += "2,70.0,20.0,0.5,8.1,3.5\n"
    output_path.write_text(csv_content, encoding="utf-8")


def generate_summary(
    path: list[tuple[float, float]],
    strategy: dict | None,
    output_path: Path,
) -> None:
    """生成总结报告。"""
    lines = [
        "=" * 60,
        "Demo End-to-End Route Planning Summary",
        "=" * 60,
        "",
        f"Route points: {len(path)}",
        f"Start: {path[0] if path else 'N/A'}",
        f"End: {path[-1] if path else 'N/A'}",
        "",
        "CMEMS Strategy:",
    ]
    
    if strategy:
        lines.append(f"  Selected: {strategy.get('selected', 'N/A')}")
        lines.append(f"  nextsim available: {strategy.get('nextsim_available', False)}")
        lines.append(f"  L4 available: {strategy.get('l4_available', False)}")
        if strategy.get("fallback_reason"):
            lines.append(f"  Fallback reason: {strategy['fallback_reason']}")
    else:
        lines.append("  No strategy loaded (using demo data)")
    
    lines.extend([
        "",
        "Status: [OK] SUCCESS",
        "=" * 60,
    ])
    
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """主函数。"""
    parser = argparse.ArgumentParser(description="Demo end-to-end route planning")
    parser.add_argument("--outdir", type=str, default="reports/demo_run", help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Demo End-to-End Route Planning")
    print("=" * 60)
    
    # 加载 CMEMS 策略
    print("\n[1/6] Loading CMEMS strategy...")
    strategy = load_cmems_strategy()
    if strategy:
        print(f"  Selected: {strategy.get('selected', 'N/A')}")
    else:
        print("  No strategy found, using demo data")
    
    # 构建网格和成本函数
    print("\n[2/6] Building grid and cost function...")
    grid, land_mask = make_demo_grid()
    cf = build_demo_cost(grid, land_mask)
    print("  [OK] Grid ready")
    
    # 规划路由
    print("\n[3/6] Planning route...")
    start_lat, start_lon = 66.0, 5.0
    end_lat, end_lon = 78.0, 150.0
    
    path = plan_route_latlon(cf, start_lat, start_lon, end_lat, end_lon)
    
    if not path:
        print("  [X] Route not reachable")
        return
    
    print(f"  [OK] Route found ({len(path)} points)")
    
    # 生成产物
    print("\n[4/6] Generating route.geojson...")
    generate_route_geojson(path, output_dir / "route.geojson")
    print("  [OK] Done")
    
    print("\n[5/6] Generating cost_breakdown.json...")
    generate_cost_breakdown(path, output_dir / "cost_breakdown.json")
    print("  [OK] Done")
    
    print("\n[6/6] Generating polaris_diagnostics.csv...")
    generate_polaris_diagnostics(output_dir / "polaris_diagnostics.csv")
    print("  [OK] Done")
    
    # 生成总结
    print("\nGenerating summary.txt...")
    generate_summary(path, strategy, output_dir / "summary.txt")
    print("  [OK] Done")
    
    print("\n" + "=" * 60)
    print("[OK] Demo complete")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

