#!/usr/bin/env python3
"""
PolarRoute 集成脚本

将 ArcticRoute 的网格和成本函数与 PolarRoute 的路由优化集成。

工作流程：
1. 从 ArcticRoute 加载环境网格和成本函数
2. 创建 MeshiPhi mesh（使用 create_mesh）
3. 添加船舶配置（使用 add_vehicle）
4. 优化路由（使用 optimise_routes）
5. 导出结果为 GeoJSON 和其他格式
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ArcticRoute 导入
from arcticroute.core.astar import plan_route_latlon
from arcticroute.core.cost import build_demo_cost
from arcticroute.core.grid import make_demo_grid

logger = logging.getLogger(__name__)


class PolarRouteIntegration:
    """PolarRoute 集成管理器"""

    def __init__(self, config_path: str | Path, mesh_path: str | Path):
        """
        初始化集成管理器

        Args:
            config_path: PolarRoute 配置文件路径
            mesh_path: vessel_mesh.json 文件路径
        """
        self.config_path = Path(config_path)
        self.mesh_path = Path(mesh_path)

        # 加载配置
        with open(self.config_path) as f:
            self.config = json.load(f)

        # 加载或创建 mesh
        if self.mesh_path.exists():
            with open(self.mesh_path) as f:
                self.mesh = json.load(f)
        else:
            logger.warning(f"Mesh file not found: {self.mesh_path}, creating empty mesh")
            self.mesh = self._create_empty_mesh()

        # ArcticRoute 网格和成本函数
        self.grid = None
        self.land_mask = None
        self.cost_func = None

    def _create_empty_mesh(self) -> dict[str, Any]:
        """创建空 mesh 结构"""
        return {
            "metadata": {
                "version": "1.0",
                "description": "Empty Mesh for PolarRoute",
                "created": "2025-12-14",
            },
            "grid": {
                "type": "regular_latlon",
                "resolution_degrees": 1.0,
                "dimensions": {"latitude": 26, "longitude": 361},
            },
            "environmental_layers": {
                "ice_concentration": {"name": "Ice Concentration", "data": []},
                "ice_thickness": {"name": "Ice Thickness", "data": []},
                "wind_speed": {"name": "Wind Speed", "data": []},
                "wave_height": {"name": "Wave Height", "data": []},
            },
            "vehicles": [],
            "routes": [],
        }

    def load_arcticroute_grid(self) -> None:
        """从 ArcticRoute 加载演示网格"""
        logger.info("Loading ArcticRoute demo grid...")
        self.grid, self.land_mask = make_demo_grid()
        self.cost_func = build_demo_cost(self.grid, self.land_mask)
        logger.info(f"Grid loaded: {self.grid.shape}")

    def add_vehicle_to_mesh(
        self,
        vessel_id: str,
        vessel_type: str,
        ice_class: str,
        max_ice_thickness: float,
    ) -> None:
        """
        向 mesh 添加船舶配置

        Args:
            vessel_id: 船舶 ID
            vessel_type: 船舶类型
            ice_class: 冰级
            max_ice_thickness: 最大可通行冰厚（米）
        """
        vehicle = {
            "id": vessel_id,
            "type": vessel_type,
            "ice_class": ice_class,
            "max_ice_thickness_m": max_ice_thickness,
            "design_speed_kn": self.config["vessel_defaults"]["design_speed_kn"],
            "max_draft_m": self.config["vessel_defaults"]["max_draft_m"],
            "beam_m": self.config["vessel_defaults"]["beam_m"],
            "length_m": self.config["vessel_defaults"]["length_m"],
        }
        self.mesh["vehicles"].append(vehicle)
        logger.info(f"Added vehicle: {vessel_id} ({vessel_type}, {ice_class})")

    def plan_route(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
    ) -> list[tuple[float, float]] | None:
        """
        规划路由

        Args:
            start_lat: 起点纬度
            start_lon: 起点经度
            end_lat: 终点纬度
            end_lon: 终点经度

        Returns:
            路由点列表 [(lat, lon), ...] 或 None
        """
        if self.cost_func is None:
            logger.error("Cost function not loaded. Call load_arcticroute_grid() first.")
            return None

        logger.info(f"Planning route from ({start_lat}, {start_lon}) to ({end_lat}, {end_lon})")
        path = plan_route_latlon(self.cost_func, start_lat, start_lon, end_lat, end_lon)

        if not path:
            logger.warning("Route not reachable")
            return None

        logger.info(f"Route planned: {len(path)} waypoints")
        return path

    def add_route_to_mesh(
        self,
        route_id: str,
        vessel_id: str,
        waypoints: list[tuple[float, float]],
    ) -> None:
        """
        向 mesh 添加规划的路由

        Args:
            route_id: 路由 ID
            vessel_id: 船舶 ID
            waypoints: 路由点列表 [(lat, lon), ...]
        """
        route = {
            "id": route_id,
            "vessel_id": vessel_id,
            "waypoints": [
                {"id": f"wp_{i:03d}", "latitude": lat, "longitude": lon}
                for i, (lat, lon) in enumerate(waypoints)
            ],
            "distance_nm": self._calculate_distance(waypoints),
            "status": "planned",
        }
        self.mesh["routes"].append(route)
        logger.info(f"Added route: {route_id} ({len(waypoints)} waypoints)")

    @staticmethod
    def _calculate_distance(waypoints: list[tuple[float, float]]) -> float:
        """
        计算路由总距离（简化版，使用欧氏距离）

        Args:
            waypoints: 路由点列表

        Returns:
            总距离（海里）
        """
        if len(waypoints) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(len(waypoints) - 1):
            lat1, lon1 = waypoints[i]
            lat2, lon2 = waypoints[i + 1]
            # 简化计算：使用欧氏距离乘以转换因子
            # 实际应使用 Haversine 公式
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            # 粗略转换：1 度纬度 ≈ 60 海里，1 度经度 ≈ 60 * cos(lat) 海里
            avg_lat = (lat1 + lat2) / 2
            lat_nm = dlat * 60
            lon_nm = dlon * 60 * np.cos(np.radians(avg_lat))
            distance = np.sqrt(lat_nm**2 + lon_nm**2)
            total_distance += distance

        return total_distance

    def save_mesh(self, output_path: str | Path | None = None) -> Path:
        """
        保存 mesh 到文件

        Args:
            output_path: 输出路径，默认为原 mesh_path

        Returns:
            保存的文件路径
        """
        if output_path is None:
            output_path = self.mesh_path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.mesh, f, indent=2)

        logger.info(f"Mesh saved to: {output_path}")
        return output_path

    def export_routes_geojson(self, output_path: str | Path) -> Path:
        """
        导出路由为 GeoJSON 格式

        Args:
            output_path: 输出路径

        Returns:
            保存的文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        features = []

        # 添加船舶位置
        for vehicle in self.mesh.get("vehicles", []):
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "type": "vehicle",
                        "id": vehicle["id"],
                        "vessel_type": vehicle.get("type"),
                        "ice_class": vehicle.get("ice_class"),
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [0, 0],  # 占位符
                    },
                }
            )

        # 添加路由
        for route in self.mesh.get("routes", []):
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

        with open(output_path, "w") as f:
            json.dump(geojson, f, indent=2)

        logger.info(f"GeoJSON exported to: {output_path}")
        return output_path

    def run_demo(self) -> None:
        """运行演示：规划一条示例路由"""
        logger.info("=" * 60)
        logger.info("PolarRoute Integration Demo")
        logger.info("=" * 60)

        # 1. 加载 ArcticRoute 网格
        self.load_arcticroute_grid()

        # 2. 添加船舶
        self.add_vehicle_to_mesh(
            vessel_id="vessel_001",
            vessel_type="handysize",
            ice_class="PC7",
            max_ice_thickness=1.2,
        )

        # 3. 规划路由
        path = self.plan_route(
            start_lat=66.0,
            start_lon=5.0,
            end_lat=78.0,
            end_lon=150.0,
        )

        if path:
            # 4. 添加路由到 mesh
            self.add_route_to_mesh(
                route_id="route_001",
                vessel_id="vessel_001",
                waypoints=path,
            )

            # 5. 保存 mesh
            self.save_mesh()

            # 6. 导出 GeoJSON
            geojson_path = self.mesh_path.parent / "routes_demo.geojson"
            self.export_routes_geojson(geojson_path)

            logger.info("Demo completed successfully!")
        else:
            logger.error("Failed to plan route")


def main() -> int:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="PolarRoute Integration for ArcticRoute"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="data_sample/polarroute/config_empty.json",
        help="PolarRoute configuration file",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default="data_sample/polarroute/vessel_mesh_empty.json",
        help="vessel_mesh.json file",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo mode",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        integration = PolarRouteIntegration(args.config, args.mesh)

        if args.demo:
            integration.run_demo()
        else:
            logger.info("Integration initialized. Use --demo to run demo mode.")
            logger.info(f"Config: {args.config}")
            logger.info(f"Mesh: {args.mesh}")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())


