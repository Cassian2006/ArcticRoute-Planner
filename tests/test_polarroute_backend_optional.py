#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PolarRoute 后端可选 Smoke Test (Phase 5A)

此测试在以下情况下会被跳过：
1. polar_route 包未安装
2. optimise_routes CLI 不可用
3. 环境变量 AR_POLAR_VESSEL_MESH 或 AR_POLAR_ROUTE_CONFIG 未设置

使用：
    # 设置环境变量
    export AR_POLAR_VESSEL_MESH=/path/to/vessel_mesh.json
    export AR_POLAR_ROUTE_CONFIG=/path/to/route_config.json
    
    # 运行测试
    pytest tests/test_polarroute_backend_optional.py -v
"""

import os
import shutil
import pytest
from pathlib import Path

# 尝试导入 PolarRoute
try:
    import polar_route
    POLAR_ROUTE_AVAILABLE = True
except ImportError:
    POLAR_ROUTE_AVAILABLE = False

# 检查 optimise_routes CLI
OPTIMISE_ROUTES_AVAILABLE = shutil.which("optimise_routes") is not None


def skip_if_no_polarroute():
    """如果 PolarRoute 不可用则跳过测试。"""
    if not POLAR_ROUTE_AVAILABLE:
        pytest.skip("polar_route 包未安装")
    if not OPTIMISE_ROUTES_AVAILABLE:
        pytest.skip("optimise_routes CLI 不可用")


class TestPolarRouteBackendOptional:
    """PolarRoute 后端可选测试。"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """检查前置条件。"""
        skip_if_no_polarroute()
        
        # 检查环境变量
        self.vessel_mesh_path = os.getenv("AR_POLAR_VESSEL_MESH")
        self.route_config_path = os.getenv("AR_POLAR_ROUTE_CONFIG")
        
        if not self.vessel_mesh_path:
            pytest.skip("环境变量 AR_POLAR_VESSEL_MESH 未设置")
        if not self.route_config_path:
            pytest.skip("环境变量 AR_POLAR_ROUTE_CONFIG 未设置")
        
        # 检查文件存在
        if not Path(self.vessel_mesh_path).exists():
            pytest.skip(f"vessel_mesh 文件不存在: {self.vessel_mesh_path}")
        if not Path(self.route_config_path).exists():
            pytest.skip(f"route_config 文件不存在: {self.route_config_path}")
    
    def test_polarroute_backend_import(self):
        """测试 PolarRoute 后端导入。"""
        from arcticroute.core.planners.polarroute_backend import PolarRouteBackend
        assert PolarRouteBackend is not None
    
    def test_polarroute_backend_initialization(self):
        """测试 PolarRoute 后端初始化。"""
        from arcticroute.core.planners.polarroute_backend import PolarRouteBackend
        
        backend = PolarRouteBackend(
            vessel_mesh_path=self.vessel_mesh_path,
            route_config_path=self.route_config_path,
        )
        assert backend is not None
        assert backend.vessel_mesh_path == Path(self.vessel_mesh_path)
        assert backend.route_config_path == Path(self.route_config_path)
    
    def test_polarroute_backend_plan(self):
        """测试 PolarRoute 后端规划。"""
        from arcticroute.core.planners.polarroute_backend import PolarRouteBackend
        
        backend = PolarRouteBackend(
            vessel_mesh_path=self.vessel_mesh_path,
            route_config_path=self.route_config_path,
        )
        
        # 规划一条路线（北极示例坐标）
        start_latlon = (66.0, 5.0)
        end_latlon = (78.0, 150.0)
        
        path = backend.plan(start_latlon, end_latlon)
        
        # 断言：路径应该包含至少 2 个点
        assert isinstance(path, list), "路径应该是列表"
        assert len(path) >= 2, f"路径应该包含至少 2 个点，实际 {len(path)} 个"
        
        # 断言：每个点应该是 (lat, lon) 元组
        for point in path:
            assert isinstance(point, tuple), "每个点应该是元组"
            assert len(point) == 2, "每个点应该有 2 个坐标"
            lat, lon = point
            assert isinstance(lat, (int, float)), "纬度应该是数字"
            assert isinstance(lon, (int, float)), "经度应该是数字"
        
        # 断言：起点和终点应该接近输入坐标
        first_point = path[0]
        last_point = path[-1]
        
        # 允许一定的误差（例如 1 度）
        assert abs(first_point[0] - start_latlon[0]) < 2.0, "起点纬度偏差过大"
        assert abs(first_point[1] - start_latlon[1]) < 2.0, "起点经度偏差过大"
        assert abs(last_point[0] - end_latlon[0]) < 2.0, "终点纬度偏差过大"
        assert abs(last_point[1] - end_latlon[1]) < 2.0, "终点经度偏差过大"


class TestAStarBackend:
    """A* 后端测试。"""
    
    def test_astar_backend_import(self):
        """测试 A* 后端导入。"""
        from arcticroute.core.planners.polarroute_backend import AStarBackend
        assert AStarBackend is not None
    
    def test_astar_backend_initialization(self):
        """测试 A* 后端初始化。"""
        from arcticroute.core.planners.polarroute_backend import AStarBackend
        from arcticroute.core.cost import build_demo_cost
        from arcticroute.core.grid import make_demo_grid
        
        grid, land_mask = make_demo_grid()
        cost_field = build_demo_cost(grid, land_mask)
        backend = AStarBackend(cost_field)
        assert backend is not None
        assert backend.cost_field is cost_field
    
    def test_astar_backend_plan(self):
        """测试 A* 后端规划。"""
        from arcticroute.core.planners.polarroute_backend import AStarBackend
        from arcticroute.core.cost import build_demo_cost
        from arcticroute.core.grid import make_demo_grid
        
        grid, land_mask = make_demo_grid()
        cost_field = build_demo_cost(grid, land_mask)
        backend = AStarBackend(cost_field)
        
        # 规划一条路线
        start_latlon = (66.0, 5.0)
        end_latlon = (78.0, 150.0)
        
        path = backend.plan(start_latlon, end_latlon)
        
        # 断言：路径应该包含至少 2 个点
        assert isinstance(path, list), "路径应该是列表"
        assert len(path) >= 2, f"路径应该包含至少 2 个点，实际 {len(path)} 个"
        
        # 断言：每个点应该是 (lat, lon) 元组
        for point in path:
            assert isinstance(point, tuple), "每个点应该是元组"
            assert len(point) == 2, "每个点应该有 2 个坐标"


class TestPlannerBackendProtocol:
    """规划器后端协议测试。"""
    
    def test_backend_protocol_import(self):
        """测试规划器后端协议导入。"""
        from arcticroute.core.planners.base import RoutePlannerBackend, PlannerBackendError
        assert RoutePlannerBackend is not None
        assert PlannerBackendError is not None
    
    def test_backend_error_exception(self):
        """测试规划器后端错误异常。"""
        from arcticroute.core.planners.base import PlannerBackendError
        
        with pytest.raises(PlannerBackendError):
            raise PlannerBackendError("测试错误")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

