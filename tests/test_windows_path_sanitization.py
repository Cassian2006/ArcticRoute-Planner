"""
Windows 路径清理测试：确保所有生成的路径都不包含冒号（除了 drive 本身）。

这个测试是离线的，不需要 PolarRoute 安装。
"""

import pytest
from pathlib import Path
from arcticroute.core.planners.polarroute_artifacts import (
    get_pipeline_recent_dir,
    get_pipeline_artifact_roots,
    find_latest_vessel_mesh,
)


def test_pipeline_recent_dir_no_colon_in_parts():
    """测试 recent 目录路径的每个部分都不包含冒号。"""
    pipeline_dir = r"D:\polarroute-pipeline\PolarRoute-pipeline"
    
    recent_dir = get_pipeline_recent_dir(pipeline_dir)
    
    # 检查路径的每个部分（排除 drive）
    parts = recent_dir.parts
    # Windows Path.parts 的第一个元素是 drive（如 "D:\\"），需要排除
    # drive 格式是 "D:\\"，包含冒号，这是正常的
    non_drive_parts = [p for p in parts if not (len(p) >= 2 and p[1] == ":" and p.endswith("\\"))]
    for part in non_drive_parts:
        # 路径部分不应该包含冒号（除了 drive 本身）
        assert ":" not in part, f"路径部分包含冒号: {part} in {parts}"
    
    # 完整路径应该是有效的
    assert str(recent_dir) == r"D:\polarroute-pipeline\PolarRoute-pipeline\workflow-manager\recent.operational-polarroute"


def test_pipeline_artifact_roots_no_colon_in_parts():
    """测试 artifact 根目录路径的每个部分都不包含冒号。"""
    pipeline_dir = r"D:\polarroute-pipeline\PolarRoute-pipeline"
    
    artifact_roots = get_pipeline_artifact_roots(pipeline_dir)
    
    for root in artifact_roots:
        parts = root.parts
        # Windows Path.parts 的第一个元素是 drive（如 "D:\\"），需要排除
        # drive 格式是 "D:\\"，包含冒号，这是正常的
        non_drive_parts = [p for p in parts if not (len(p) >= 2 and p[1] == ":" and p.endswith("\\"))]
        for part in non_drive_parts:
            # 路径部分不应该包含冒号（除了 drive 本身）
            assert ":" not in part, f"路径部分包含冒号: {part} in {parts}"


def test_pipeline_recent_dir_uses_correct_structure():
    """测试 recent 目录使用正确的结构（不使用 drive 参与目录名）。"""
    pipeline_dir = r"D:\polarroute-pipeline\PolarRoute-pipeline"
    
    recent_dir = get_pipeline_recent_dir(pipeline_dir)
    
    # 应该使用 workflow-manager/recent.operational-polarroute
    assert recent_dir.name == "recent.operational-polarroute"
    assert recent_dir.parent.name == "workflow-manager"
    
    # 不应该包含 drive 字母
    assert not recent_dir.name.startswith("recent.D")
    assert not recent_dir.name.startswith("recent.d")


def test_pipeline_artifact_roots_includes_expected_dirs():
    """测试 artifact 根目录包含预期的目录。"""
    pipeline_dir = r"D:\polarroute-pipeline\PolarRoute-pipeline"
    
    artifact_roots = get_pipeline_artifact_roots(pipeline_dir)
    root_strs = [str(r) for r in artifact_roots]
    
    # 应该包含 recent 目录（如果存在）
    recent_dir = get_pipeline_recent_dir(pipeline_dir)
    if recent_dir.exists():
        assert str(recent_dir) in root_strs


def test_find_latest_vessel_mesh_returns_valid_path():
    """测试 find_latest_vessel_mesh 返回的路径（如果存在）是有效的。"""
    pipeline_dir = r"D:\polarroute-pipeline\PolarRoute-pipeline"
    
    vessel_mesh = find_latest_vessel_mesh(pipeline_dir)
    
    if vessel_mesh:
        # 如果找到了，路径应该是有效的
        mesh_path = Path(vessel_mesh)
        assert mesh_path.exists()
        assert mesh_path.is_file()
        assert mesh_path.stat().st_size > 0
        
        # 路径的每个部分都不应该包含冒号（除了 drive）
        for part in mesh_path.parts:
            if ":" in part and part != mesh_path.drive:
                pytest.fail(f"路径部分包含冒号: {part}")


def test_path_construction_uses_pathlib():
    """测试所有路径构造都使用 Path 对象，不拼接字符串。"""
    pipeline_dir = r"D:\polarroute-pipeline\PolarRoute-pipeline"
    
    recent_dir = get_pipeline_recent_dir(pipeline_dir)
    
    # 应该是一个 Path 对象
    assert isinstance(recent_dir, Path)
    
    # 不应该包含字符串拼接的痕迹（如 f"recent.{drive}"）
    recent_str = str(recent_dir)
    assert "recent.D:" not in recent_str
    assert "recent.d:" not in recent_str

