#!/usr/bin/env python3
"""
简化的 PolarRoute 演示脚本

不依赖 ArcticRoute 的复杂模块，直接演示 vessel_mesh.json 的生成和使用。
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def create_demo_route():
    """创建演示路由"""
    # 从 Murmansk 到 Sabetta 的演示路由
    waypoints = [
        (68.95, 33.08),   # Murmansk
        (69.50, 40.00),   # Checkpoint 1
        (70.50, 50.00),   # Checkpoint 2
        (71.27, 72.00),   # Sabetta
    ]
    return waypoints


def load_mesh(mesh_path: Path) -> dict:
    """加载 mesh 文件"""
    with open(mesh_path) as f:
        return json.load(f)


def add_vehicle_to_mesh(mesh: dict, vessel_id: str, vessel_type: str) -> None:
    """向 mesh 添加船舶"""
    vehicle = {
        "id": vessel_id,
        "type": vessel_type,
        "ice_class": "PC7",
        "max_ice_thickness_m": 1.2,
        "design_speed_kn": 14.0,
        "max_draft_m": 10.0,
        "beam_m": 32.0,
        "length_m": 190.0,
    }
    mesh["vehicles"].append(vehicle)
    logger.info(f"Added vehicle: {vessel_id}")


def add_route_to_mesh(mesh: dict, route_id: str, vessel_id: str, waypoints: list) -> None:
    """向 mesh 添加路由"""
    route = {
        "id": route_id,
        "vessel_id": vessel_id,
        "waypoints": [
            {"id": f"wp_{i:03d}", "latitude": lat, "longitude": lon}
            for i, (lat, lon) in enumerate(waypoints)
        ],
        "distance_nm": 500.0,  # 简化估计
        "status": "planned",
    }
    mesh["routes"].append(route)
    logger.info(f"Added route: {route_id} with {len(waypoints)} waypoints")


def save_mesh(mesh: dict, output_path: Path) -> None:
    """保存 mesh 文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(mesh, f, indent=2)
    logger.info(f"Mesh saved to: {output_path}")


def export_geojson(mesh: dict, output_path: Path) -> None:
    """导出为 GeoJSON"""
    features = []

    # 添加路由
    for route in mesh.get("routes", []):
        waypoints = route.get("waypoints", [])
        if waypoints:
            coordinates = [[wp["longitude"], wp["latitude"]] for wp in waypoints]
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "type": "route",
                        "id": route["id"],
                        "vessel_id": route.get("vessel_id"),
                        "distance_nm": route.get("distance_nm", 0),
                        "status": route.get("status"),
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coordinates,
                    },
                }
            )

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)
    logger.info(f"GeoJSON exported to: {output_path}")


def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 60)
    logger.info("PolarRoute Integration Demo (Simplified)")
    logger.info("=" * 60)

    # 路径
    mesh_path = Path("data_sample/polarroute/vessel_mesh_empty.json")
    output_mesh_path = Path("data_sample/polarroute/vessel_mesh_demo.json")
    geojson_path = Path("data_sample/polarroute/routes_demo.geojson")

    # 1. 加载 mesh
    logger.info("\n[Step 1] Loading empty mesh...")
    mesh = load_mesh(mesh_path)
    logger.info(f"  - Grid type: {mesh['grid']['type']}")
    logger.info(f"  - Grid resolution: {mesh['grid']['resolution_degrees']}°")

    # 2. 添加船舶
    logger.info("\n[Step 2] Adding vehicle...")
    add_vehicle_to_mesh(mesh, "vessel_001", "handysize")

    # 3. 创建演示路由
    logger.info("\n[Step 3] Creating demo route...")
    waypoints = create_demo_route()
    logger.info(f"  - Route: Murmansk → Sabetta")
    logger.info(f"  - Waypoints: {len(waypoints)}")

    # 4. 添加路由到 mesh
    logger.info("\n[Step 4] Adding route to mesh...")
    add_route_to_mesh(mesh, "route_001", "vessel_001", waypoints)

    # 5. 保存 mesh
    logger.info("\n[Step 5] Saving mesh...")
    save_mesh(mesh, output_mesh_path)

    # 6. 导出 GeoJSON
    logger.info("\n[Step 6] Exporting GeoJSON...")
    export_geojson(mesh, geojson_path)

    # 7. 打印摘要
    logger.info("\n" + "=" * 60)
    logger.info("Demo Summary")
    logger.info("=" * 60)
    logger.info(f"✓ Mesh file: {output_mesh_path}")
    logger.info(f"✓ GeoJSON file: {geojson_path}")
    logger.info(f"✓ Vehicles: {len(mesh['vehicles'])}")
    logger.info(f"✓ Routes: {len(mesh['routes'])}")
    logger.info("\nNext steps:")
    logger.info("1. Use the mesh file with PolarRoute:")
    logger.info(f"   optimise_routes config.json {output_mesh_path} waypoints.json")
    logger.info("2. View the GeoJSON in QGIS or Leaflet")
    logger.info("3. Integrate with real environmental data")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

