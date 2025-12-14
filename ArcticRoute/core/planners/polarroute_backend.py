#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PolarRoute 后端实现 (Phase 5A + 5B)

通过 CLI 调用 PolarRoute 的 optimise_routes 命令，解析输出 route.json。

支持两种模式：
1. Phase 5A：外部 vessel_mesh.json + route_config.json
2. Phase 5B：从 pipeline 目录自动查找最新的 vessel_mesh.json

使用方式：
    # Phase 5A：外部文件模式
    backend = PolarRouteBackend(
        vessel_mesh_path="/path/to/vessel_mesh.json",
        route_config_path="/path/to/route_config.json"
    )
    
    # Phase 5B：Pipeline 模式
    backend = PolarRouteBackend(
        pipeline_dir="/path/to/pipeline"
    )
    
    path = backend.plan((start_lat, start_lon), (end_lat, end_lon))
"""

import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, List, Optional, Any
import logging

from .base import RoutePlannerBackend, PlannerBackendError

logger = logging.getLogger(__name__)


class PolarRouteBackend:
    """
    PolarRoute 路由规划后端。
    
    通过 CLI 调用 PolarRoute 的 optimise_routes 命令。
    支持两种初始化模式：
    - 外部文件模式（Phase 5A）：vessel_mesh_path + route_config_path
    - Pipeline 模式（Phase 5B）：pipeline_dir（自动查找最新的 vessel_mesh）
    """
    
    def __init__(
        self,
        vessel_mesh_path: Optional[str] = None,
        route_config_path: Optional[str] = None,
        pipeline_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        """
        初始化 PolarRoute 后端。
        
        Args:
            vessel_mesh_path: vessel_mesh.json 路径（Phase 5A 模式）
            route_config_path: route_config.json 路径（Phase 5A 模式）
            pipeline_dir: Pipeline 目录路径（Phase 5B 模式）
            output_dir: 输出目录（默认使用临时目录）
        
        Raises:
            PlannerBackendError: 如果参数不合法或文件不存在
        """
        # 检查 optimise_routes 命令
        if not shutil.which("optimise_routes"):
            raise PlannerBackendError(
                "optimise_routes 命令不可用。请安装 PolarRoute: pip install polar-route"
            )
        
        # 模式判断
        if pipeline_dir:
            # Phase 5B：Pipeline 模式
            self._init_pipeline_mode(pipeline_dir)
        elif vessel_mesh_path and route_config_path:
            # Phase 5A：外部文件模式
            self._init_external_mode(vessel_mesh_path, route_config_path)
        else:
            raise PlannerBackendError(
                "必须提供以下之一：\n"
                "1. vessel_mesh_path + route_config_path（Phase 5A 外部文件模式）\n"
                "2. pipeline_dir（Phase 5B Pipeline 模式）"
            )
        
        self.output_dir = Path(output_dir) if output_dir else None
    
    def _init_external_mode(self, vessel_mesh_path: str, route_config_path: str) -> None:
        """初始化外部文件模式（Phase 5A）"""
        self.vessel_mesh_path = Path(vessel_mesh_path)
        self.route_config_path = Path(route_config_path)
        self.pipeline_dir = None
        
        # 检查输入文件
        if not self.vessel_mesh_path.exists():
            raise PlannerBackendError(
                f"vessel_mesh 文件不存在: {self.vessel_mesh_path}"
            )
        if not self.route_config_path.exists():
            raise PlannerBackendError(
                f"route_config 文件不存在: {self.route_config_path}"
            )
        
        logger.info(f"PolarRoute 后端初始化（Phase 5A 外部文件模式）")
        logger.debug(f"  vessel_mesh: {self.vessel_mesh_path}")
        logger.debug(f"  route_config: {self.route_config_path}")
    
    def _init_pipeline_mode(self, pipeline_dir: str) -> None:
        """初始化 Pipeline 模式（Phase 5B）"""
        from arcticroute.integrations.polarroute_artifacts import find_latest_vessel_mesh
        
        self.pipeline_dir = Path(pipeline_dir)
        
        if not self.pipeline_dir.exists():
            raise PlannerBackendError(
                f"Pipeline 目录不存在: {self.pipeline_dir}"
            )
        
        # 查找最新的 vessel_mesh.json
        mesh_path = find_latest_vessel_mesh(str(self.pipeline_dir))
        
        if not mesh_path:
            raise PlannerBackendError(
                f"未找到 vessel_mesh.json 在 Pipeline 目录中: {self.pipeline_dir}\n"
                "请先执行 pipeline execute，或检查目录结构。"
            )
        
        self.vessel_mesh_path = Path(mesh_path)
        
        # 对于 route_config，暂时假设在 pipeline 根目录或 config 子目录
        # 如果找不到，使用 None（可能需要用户手动指定）
        potential_config_paths = [
            self.pipeline_dir / "route_config.json",
            self.pipeline_dir / "config" / "route_config.json",
            self.pipeline_dir / "configs" / "route_config.json",
        ]
        
        self.route_config_path = None
        for config_path in potential_config_paths:
            if config_path.exists():
                self.route_config_path = config_path
                break
        
        if not self.route_config_path:
            raise PlannerBackendError(
                f"未找到 route_config.json 在 Pipeline 目录中: {self.pipeline_dir}\n"
                "请确保 route_config.json 在以下位置之一：\n"
                f"  - {self.pipeline_dir / 'route_config.json'}\n"
                f"  - {self.pipeline_dir / 'config' / 'route_config.json'}\n"
                f"  - {self.pipeline_dir / 'configs' / 'route_config.json'}"
            )
        
        logger.info(f"PolarRoute 后端初始化（Phase 5B Pipeline 模式）")
        logger.debug(f"  pipeline_dir: {self.pipeline_dir}")
        logger.debug(f"  vessel_mesh: {self.vessel_mesh_path}")
        logger.debug(f"  route_config: {self.route_config_path}")
    
    def plan(
        self,
        start_latlon: Tuple[float, float],
        end_latlon: Tuple[float, float],
        **kwargs: Any
    ) -> List[Tuple[float, float]]:
        """
        规划一条路线。
        
        Args:
            start_latlon: (latitude, longitude) 起点
            end_latlon: (latitude, longitude) 终点
            **kwargs: 其他参数（目前未使用）
        
        Returns:
            [(lat, lon), ...] 路径点列表（纬度在前，经度在后）
        
        Raises:
            PlannerBackendError: 规划失败时抛出
        """
        start_lat, start_lon = start_latlon
        end_lat, end_lon = end_latlon
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # 创建 waypoints.csv
            waypoints_csv = tmpdir_path / "waypoints.csv"
            self._write_waypoints_csv(
                waypoints_csv,
                start_lat, start_lon,
                end_lat, end_lon
            )
            
            # 确定输出目录
            output_dir = self.output_dir or tmpdir_path
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 构建 CLI 命令
            cmd = [
                "optimise_routes",
                str(self.route_config_path),
                str(self.vessel_mesh_path),
                str(waypoints_csv),
                "-p",  # --path_geojson
                "-o", str(output_dir),
            ]
            
            logger.debug(f"执行 PolarRoute 命令: {' '.join(cmd)}")
            
            # 执行命令
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 分钟超时
                )
            except subprocess.TimeoutExpired:
                raise PlannerBackendError(
                    "PolarRoute optimise_routes 命令执行超时（5分钟）"
                )
            except Exception as e:
                raise PlannerBackendError(
                    f"PolarRoute optimise_routes 命令执行失败: {e}"
                )
            
            # 检查返回码
            if result.returncode != 0:
                error_log = output_dir / "polarroute_last_error.log"
                error_log.write_text(
                    f"命令: {' '.join(cmd)}\n"
                    f"返回码: {result.returncode}\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}\n"
                )
                raise PlannerBackendError(
                    f"PolarRoute optimise_routes 失败（返回码 {result.returncode}）。"
                    f"详见 {error_log}"
                )
            
            # 解析输出 route.json
            route_json_path = output_dir / "route.json"
            if not route_json_path.exists():
                error_log = output_dir / "polarroute_last_error.log"
                error_log.write_text(
                    f"命令: {' '.join(cmd)}\n"
                    f"返回码: {result.returncode}\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}\n"
                    f"错误: route.json 未生成\n"
                )
                raise PlannerBackendError(
                    f"PolarRoute 未生成 route.json。详见 {error_log}"
                )
            
            # 解析 route.json
            try:
                with open(route_json_path) as f:
                    route_data = json.load(f)
            except Exception as e:
                raise PlannerBackendError(
                    f"无法解析 route.json: {e}"
                )
            
            # 提取坐标
            path = self._extract_path_from_route_json(route_data)
            
            if not path:
                raise PlannerBackendError(
                    "route.json 中未找到有效的路径坐标"
                )
            
            logger.debug(f"PolarRoute 规划成功，路径包含 {len(path)} 个点")
            return path
    
    @staticmethod
    def _write_waypoints_csv(
        csv_path: Path,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
    ) -> None:
        """
        写入 waypoints.csv 文件。
        
        格式：
            id,latitude,longitude
            0,start_lat,start_lon
            1,end_lat,end_lon
        """
        csv_content = (
            "id,latitude,longitude\n"
            f"0,{start_lat},{start_lon}\n"
            f"1,{end_lat},{end_lon}\n"
        )
        csv_path.write_text(csv_content)
    
    @staticmethod
    def _extract_path_from_route_json(route_data: dict) -> List[Tuple[float, float]]:
        """
        从 route.json 中提取路径坐标。
        
        route.json 结构：
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[lon, lat], [lon, lat], ...]
                    },
                    ...
                }
            ]
        }
        
        返回 [(lat, lon), ...] 列表。
        """
        try:
            features = route_data.get("features", [])
            if not features:
                return []
            
            # 取第一个 feature
            feature = features[0]
            geometry = feature.get("geometry", {})
            
            if geometry.get("type") != "LineString":
                return []
            
            coordinates = geometry.get("coordinates", [])
            
            # 转换 [lon, lat] 为 (lat, lon)
            path = [(lat, lon) for lon, lat in coordinates]
            
            return path
        except Exception as e:
            logger.error(f"解析 route.json 失败: {e}")
            return []


class AStarBackend:
    """
    A* 规划器后端（包装现有的 plan_route_latlon）。
    """
    
    def __init__(self, cost_field: Any):
        """
        初始化 A* 后端。
        
        Args:
            cost_field: CostField 对象
        """
        self.cost_field = cost_field
    
    def plan(
        self,
        start_latlon: Tuple[float, float],
        end_latlon: Tuple[float, float],
        **kwargs: Any
    ) -> List[Tuple[float, float]]:
        """
        使用 A* 规划一条路线。
        
        Args:
            start_latlon: (latitude, longitude) 起点
            end_latlon: (latitude, longitude) 终点
            **kwargs: 其他参数（neighbor8, max_expansions 等）
        
        Returns:
            [(lat, lon), ...] 路径点列表
        
        Raises:
            PlannerBackendError: 规划失败时抛出
        """
        from ..astar import plan_route_latlon
        
        start_lat, start_lon = start_latlon
        end_lat, end_lon = end_latlon
        
        try:
            path = plan_route_latlon(
                self.cost_field,
                start_lat,
                start_lon,
                end_lat,
                end_lon,
                neighbor8=kwargs.get("neighbor8", True),
                max_expansions=kwargs.get("max_expansions", None),
            )
            
            if not path:
                raise PlannerBackendError("A* 规划失败：无法找到路径")
            
            return path
        except Exception as e:
            raise PlannerBackendError(f"A* 规划失败: {e}")

