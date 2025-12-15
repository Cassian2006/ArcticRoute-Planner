#!/usr/bin/env python3
"""
PolarRoute 集成测试脚本

验证：
1. vessel_mesh.json 结构完整性
2. config.json 有效性
3. 集成脚本功能
4. 输出格式正确性
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_mesh_structure(mesh: dict) -> tuple[bool, list[str]]:
    """
    验证 vessel_mesh.json 结构

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    # 检查必需的顶级键
    required_keys = ["metadata", "grid", "environmental_layers"]
    for key in required_keys:
        if key not in mesh:
            errors.append(f"Missing required key: {key}")

    # 验证 metadata
    if "metadata" in mesh:
        metadata = mesh["metadata"]
        required_meta = ["version", "description", "created"]
        for key in required_meta:
            if key not in metadata:
                errors.append(f"Missing metadata key: {key}")

    # 验证 grid
    if "grid" in mesh:
        grid = mesh["grid"]
        required_grid = ["type", "resolution_degrees", "dimensions"]
        for key in required_grid:
            if key not in grid:
                errors.append(f"Missing grid key: {key}")

        if "dimensions" in grid:
            dims = grid["dimensions"]
            if "latitude" not in dims or "longitude" not in dims:
                errors.append("Grid dimensions must have 'latitude' and 'longitude'")

    # 验证 environmental_layers
    if "environmental_layers" in mesh:
        layers = mesh["environmental_layers"]
        required_layers = ["ice_concentration", "ice_thickness"]
        for layer in required_layers:
            if layer not in layers:
                errors.append(f"Missing environmental layer: {layer}")

    # 验证 vehicles（可选但如果存在应有效）
    if "vehicles" in mesh:
        if not isinstance(mesh["vehicles"], list):
            errors.append("'vehicles' must be a list")
        for i, vehicle in enumerate(mesh["vehicles"]):
            if "id" not in vehicle:
                errors.append(f"Vehicle {i} missing 'id'")
            if "type" not in vehicle:
                errors.append(f"Vehicle {i} missing 'type'")

    # 验证 routes（可选但如果存在应有效）
    if "routes" in mesh:
        if not isinstance(mesh["routes"], list):
            errors.append("'routes' must be a list")
        for i, route in enumerate(mesh["routes"]):
            if "id" not in route:
                errors.append(f"Route {i} missing 'id'")
            if "waypoints" not in route:
                errors.append(f"Route {i} missing 'waypoints'")

    return len(errors) == 0, errors


def validate_config_structure(config: dict) -> tuple[bool, list[str]]:
    """
    验证 config.json 结构

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    # 检查必需的顶级键
    required_keys = ["metadata", "routing", "vessel_defaults"]
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required key: {key}")

    # 验证 routing
    if "routing" in config:
        routing = config["routing"]
        required_routing = ["algorithm", "optimization_method"]
        for key in required_routing:
            if key not in routing:
                errors.append(f"Missing routing key: {key}")

    # 验证 vessel_defaults
    if "vessel_defaults" in config:
        defaults = config["vessel_defaults"]
        required_defaults = ["design_speed_kn", "ice_class"]
        for key in required_defaults:
            if key not in defaults:
                errors.append(f"Missing vessel_defaults key: {key}")

    # 验证 environmental_weights
    if "environmental_weights" in config:
        weights = config["environmental_weights"]
        if not isinstance(weights, dict):
            errors.append("'environmental_weights' must be a dictionary")
        # 检查权重和是否接近 1.0
        total_weight = sum(weights.values())
        if not (0.9 <= total_weight <= 1.1):
            errors.append(f"Environmental weights sum to {total_weight}, expected ~1.0")

    return len(errors) == 0, errors


def test_mesh_file(mesh_path: Path) -> bool:
    """测试 mesh 文件"""
    logger.info(f"Testing mesh file: {mesh_path}")

    if not mesh_path.exists():
        logger.error(f"Mesh file not found: {mesh_path}")
        return False

    try:
        with open(mesh_path) as f:
            mesh = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in mesh file: {e}")
        return False

    is_valid, errors = validate_mesh_structure(mesh)

    if is_valid:
        logger.info("✓ Mesh structure is valid")
        logger.info(f"  - Version: {mesh.get('metadata', {}).get('version')}")
        logger.info(f"  - Grid type: {mesh.get('grid', {}).get('type')}")
        logger.info(f"  - Vehicles: {len(mesh.get('vehicles', []))}")
        logger.info(f"  - Routes: {len(mesh.get('routes', []))}")
        return True
    else:
        logger.error("✗ Mesh structure validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False


def test_config_file(config_path: Path) -> bool:
    """测试 config 文件"""
    logger.info(f"Testing config file: {config_path}")

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return False

    try:
        with open(config_path) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        return False

    is_valid, errors = validate_config_structure(config)

    if is_valid:
        logger.info("✓ Config structure is valid")
        logger.info(f"  - Algorithm: {config.get('routing', {}).get('algorithm')}")
        logger.info(f"  - Optimization: {config.get('routing', {}).get('optimization_method')}")
        logger.info(f"  - Vessel type: {config.get('vessel_defaults', {}).get('ice_class')}")
        return True
    else:
        logger.error("✗ Config structure validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False


def test_waypoints_file(waypoints_path: Path) -> bool:
    """测试 waypoints 文件"""
    logger.info(f"Testing waypoints file: {waypoints_path}")

    if not waypoints_path.exists():
        logger.warning(f"Waypoints file not found: {waypoints_path}")
        return True  # 可选文件

    try:
        with open(waypoints_path) as f:
            waypoints = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in waypoints file: {e}")
        return False

    if "routes" not in waypoints:
        logger.error("Missing 'routes' key in waypoints file")
        return False

    routes = waypoints["routes"]
    if not isinstance(routes, list):
        logger.error("'routes' must be a list")
        return False

    logger.info("✓ Waypoints structure is valid")
    logger.info(f"  - Routes: {len(routes)}")
    for route in routes:
        waypoints_count = len(route.get("waypoints", []))
        logger.info(f"    - {route.get('id')}: {waypoints_count} waypoints")

    return True


def test_integration_import() -> bool:
    """测试集成脚本导入"""
    logger.info("Testing integration script import")

    try:
        import sys
        from pathlib import Path
        
        # 添加项目根目录到 sys.path
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from integrate_polarroute import PolarRouteIntegration

        logger.info("✓ Successfully imported PolarRouteIntegration")
        return True
    except ImportError as e:
        logger.error(f"✗ Failed to import PolarRouteIntegration: {e}")
        return False


def test_integration_initialization(config_path: Path, mesh_path: Path) -> bool:
    """测试集成脚本初始化"""
    logger.info("Testing integration initialization")

    try:
        import sys
        from pathlib import Path as PathlibPath
        
        # 添加项目根目录到 sys.path
        project_root = PathlibPath(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from integrate_polarroute import PolarRouteIntegration

        integration = PolarRouteIntegration(config_path, mesh_path)
        logger.info("✓ Successfully initialized PolarRouteIntegration")
        logger.info(f"  - Config loaded: {integration.config is not None}")
        logger.info(f"  - Mesh loaded: {integration.mesh is not None}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to initialize PolarRouteIntegration: {e}")
        return False


def run_all_tests() -> int:
    """运行所有测试"""
    logger.info("=" * 60)
    logger.info("PolarRoute Integration Tests")
    logger.info("=" * 60)

    base_path = Path("data_sample/polarroute")
    mesh_path = base_path / "vessel_mesh_empty.json"
    config_path = base_path / "config_empty.json"
    waypoints_path = base_path / "waypoints_example.json"

    results = {
        "Mesh file validation": test_mesh_file(mesh_path),
        "Config file validation": test_config_file(config_path),
        "Waypoints file validation": test_waypoints_file(waypoints_path),
        "Integration import": test_integration_import(),
        "Integration initialization": test_integration_initialization(config_path, mesh_path),
    }

    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info("=" * 60)
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    sys.exit(run_all_tests())

