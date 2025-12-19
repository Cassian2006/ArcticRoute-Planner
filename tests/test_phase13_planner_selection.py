"""
Phase 13: 测试 planner 选择逻辑（离线，不需要真实 PolarRoute）
"""
from __future__ import annotations

import sys
from pathlib import Path

# 确保可以导入 scripts 模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.polarroute_select_and_plan import choose_backend


def test_mode_astar_forced():
    """测试强制 astar 模式"""
    meta = choose_backend("astar", None, None, None)
    assert meta["planner_used"] == "astar"
    assert meta["planner_mode"] == "astar"
    assert meta["fallback_reason"] is None


def test_auto_no_pipeline_dir():
    """测试 auto 模式，pipeline_dir 不存在 → 回退 astar"""
    meta = choose_backend("auto", None, None, None)
    assert meta["planner_used"] == "astar"
    assert meta["planner_mode"] == "astar"
    assert "pipeline_unavailable" in str(meta.get("fallback_reason", ""))


def test_auto_pipeline_dir_missing():
    """测试 auto 模式，pipeline_dir 指向不存在的目录 → 回退 astar"""
    meta = choose_backend("auto", "/nonexistent/pipeline", None, None)
    assert meta["planner_used"] == "astar"
    assert meta["planner_mode"] == "astar"
    assert "pipeline_unavailable" in str(meta.get("fallback_reason", ""))


def test_polarroute_pipeline_unavailable():
    """测试 polarroute_pipeline 模式，不可用 → 回退 astar"""
    meta = choose_backend("polarroute_pipeline", None, None, None)
    assert meta["planner_used"] == "astar"
    assert meta["planner_mode"] == "astar"
    assert "pipeline_unavailable" in str(meta.get("fallback_reason", ""))


def test_polarroute_external_files_missing():
    """测试 polarroute_external 模式，文件缺失 → 回退 astar"""
    meta = choose_backend("polarroute_external", None, None, None)
    assert meta["planner_used"] == "astar"
    assert meta["planner_mode"] == "astar"
    assert meta["fallback_reason"] == "external_files_missing"


def test_polarroute_external_files_not_found():
    """测试 polarroute_external 模式，文件不存在 → 回退 astar"""
    meta = choose_backend(
        "polarroute_external", None, "/nonexistent/vessel.json", "/nonexistent/route.json"
    )
    assert meta["planner_used"] == "astar"
    assert meta["planner_mode"] == "astar"
    assert meta["fallback_reason"] == "external_files_not_found"


def test_meta_structure():
    """测试返回的 meta 结构完整性"""
    meta = choose_backend("auto", None, None, None)
    required_keys = [
        "timestamp",
        "requested_mode",
        "planner_used",
        "planner_mode",
        "fallback_reason",
        "pipeline_dir",
        "external_vessel_mesh",
        "external_route_config",
    ]
    for key in required_keys:
        assert key in meta, f"Missing key: {key}"

