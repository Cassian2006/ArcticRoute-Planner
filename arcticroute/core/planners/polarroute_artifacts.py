"""
PolarRoute Pipeline Artifacts 发现模块（Windows 兼容）

负责在 pipeline 目录中查找 vessel_mesh.json、route_config.json 等 artifacts。
所有路径处理都使用 Path 对象，避免 Windows 路径问题。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple


def get_pipeline_recent_dir(pipeline_dir: str) -> Path:
    """
    获取 pipeline 的真实 recent 目录。
    
    Windows 安全：不使用 drive 参与目录名。
    真实路径: <pipeline_dir>/workflow-manager/recent.operational-polarroute
    
    Args:
        pipeline_dir: Pipeline 根目录路径
    
    Returns:
        recent 目录的 Path 对象
    """
    pipeline_path = Path(pipeline_dir)
    return pipeline_path / "workflow-manager" / "recent.operational-polarroute"


def get_pipeline_artifact_roots(pipeline_dir: str) -> List[Path]:
    """
    获取 pipeline artifacts 搜索根目录列表。
    
    Windows 安全：所有路径都使用 Path 对象，不拼接字符串。
    
    Args:
        pipeline_dir: Pipeline 根目录路径
    
    Returns:
        存在的 artifact 根目录列表（按优先级排序）
    """
    pipeline_path = Path(pipeline_dir)
    roots = [
        get_pipeline_recent_dir(pipeline_dir),
        pipeline_path / "outputs",
        pipeline_path / "workflow-manager",
        pipeline_path / "push" / "upload",
        pipeline_path,  # 最后搜索整个 pipeline 目录
    ]
    # 只返回存在的目录
    return [r for r in roots if r.exists() and r.is_dir()]


def find_latest_vessel_mesh(pipeline_dir: str) -> Optional[str]:
    """
    在 pipeline 目录中查找最新的 vessel_mesh*.json 文件。
    
    Windows 安全：使用 Path 对象，不拼接字符串。
    
    Args:
        pipeline_dir: Pipeline 根目录路径
    
    Returns:
        最新的 vessel_mesh*.json 文件路径（字符串），如果未找到则返回 None
    """
    artifact_roots = get_pipeline_artifact_roots(pipeline_dir)
    
    candidates = []
    for root in artifact_roots:
        if not root.exists() or not root.is_dir():
            continue
        
        # 递归搜索 vessel_mesh*.json
        for pattern in ["vessel_mesh*.json", "**/vessel_mesh*.json"]:
            try:
                for file_path in root.glob(pattern):
                    if file_path.is_file() and file_path.stat().st_size > 0:
                        candidates.append(file_path)
            except Exception:
                # 忽略权限错误等
                pass
    
    if not candidates:
        return None
    
    # 按修改时间排序，返回最新的
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def find_latest_route_config(pipeline_dir: str) -> Optional[str]:
    """
    在 pipeline 目录中查找最新的 route_config*.json 文件。
    
    Windows 安全：使用 Path 对象，不拼接字符串。
    
    Args:
        pipeline_dir: Pipeline 根目录路径
    
    Returns:
        最新的 route_config*.json 文件路径（字符串），如果未找到则返回 None
    """
    artifact_roots = get_pipeline_artifact_roots(pipeline_dir)
    
    candidates = []
    for root in artifact_roots:
        if not root.exists() or not root.is_dir():
            continue
        
        # 递归搜索 route_config*.json
        for pattern in ["route_config*.json", "**/route_config*.json"]:
            try:
                for file_path in root.glob(pattern):
                    if file_path.is_file() and file_path.stat().st_size > 0:
                        candidates.append(file_path)
            except Exception:
                # 忽略权限错误等
                pass
    
    if not candidates:
        return None
    
    # 按修改时间排序，返回最新的
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def find_all_artifacts(pipeline_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """
    查找所有 artifacts。
    
    Args:
        pipeline_dir: Pipeline 根目录路径
    
    Returns:
        (vessel_mesh_path, route_config_path) 元组
    """
    vessel_mesh = find_latest_vessel_mesh(pipeline_dir)
    route_config = find_latest_route_config(pipeline_dir)
    return vessel_mesh, route_config

