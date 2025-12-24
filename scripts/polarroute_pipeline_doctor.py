"""
PolarRoute Pipeline 诊断脚本（Windows 兼容）
检查 pipeline 目录、路径问题、artifacts 发现等
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional


def get_pipeline_recent_dir(pipeline_dir: str) -> Path:
    """
    获取 pipeline 的真实 recent 目录。
    
    Windows 安全：不使用 drive 参与目录名。
    """
    pipeline_path = Path(pipeline_dir)
    return pipeline_path / "workflow-manager" / "recent.operational-polarroute"


def get_pipeline_artifact_roots(pipeline_dir: str) -> List[Path]:
    """
    获取 pipeline artifacts 搜索根目录列表。
    
    Windows 安全：所有路径都使用 Path 对象，不拼接字符串。
    """
    pipeline_path = Path(pipeline_dir)
    roots = [
        pipeline_path / "outputs",
        get_pipeline_recent_dir(pipeline_dir),
        pipeline_path / "push" / "upload",
    ]
    # 只返回存在的目录
    return [r for r in roots if r.exists() and r.is_dir()]


def search_vessel_mesh_files(pipeline_dir: str) -> tuple[List[str], Optional[str]]:
    """
    搜索 vessel_mesh*.json 文件。
    
    Windows 安全：使用 Path 对象，不拼接字符串。
    """
    pipeline_path = Path(pipeline_dir)
    if not pipeline_path.exists():
        return [], None
    
    # 搜索根目录
    search_roots = get_pipeline_artifact_roots(pipeline_dir)
    
    # 也搜索整个 pipeline 目录（递归）
    search_roots.append(pipeline_path)
    
    candidates = []
    for search_root in search_roots:
        if not search_root.exists() or not search_root.is_dir():
            continue
        
        # 递归搜索 vessel_mesh*.json
        for pattern in ["vessel_mesh*.json", "**/vessel_mesh*.json"]:
            try:
                for file_path in search_root.glob(pattern):
                    if file_path.is_file() and file_path.stat().st_size > 0:
                        candidates.append(str(file_path))
            except Exception as e:
                # 忽略权限错误等
                pass
    
    # 去重并排序（按修改时间倒序）
    unique_candidates = list(set(candidates))
    unique_candidates.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    
    # 选择最新的1个
    selected = unique_candidates[0] if unique_candidates else None
    
    return unique_candidates, selected


def check_pipeline_dir(pipeline_dir: str) -> Dict[str, Any]:
    """检查 pipeline 目录状态。"""
    if not pipeline_dir:
        return {
            "pipeline_dir": None,
            "pipeline_ok": False,
            "missing": ["pipeline_dir 未提供"],
            "reason": "pipeline_dir 未提供",
        }
    
    pipeline_path = Path(pipeline_dir)
    missing = []
    
    # 检查目录存在
    if not pipeline_path.exists():
        return {
            "pipeline_dir": str(pipeline_path),
            "pipeline_ok": False,
            "missing": [f"pipeline_dir 不存在: {pipeline_path}"],
            "reason": f"pipeline_dir 不存在: {pipeline_path}",
        }
    
    if not pipeline_path.is_dir():
        return {
            "pipeline_dir": str(pipeline_path),
            "pipeline_ok": False,
            "missing": [f"pipeline_dir 不是目录: {pipeline_path}"],
            "reason": f"pipeline_dir 不是目录: {pipeline_path}",
        }
    
    # 检查必需文件
    operational_py = pipeline_path / "workflow-manager" / "operational-polarroute.py"
    if not operational_py.exists():
        missing.append(f"missing operational-polarroute.py: {operational_py}")
    
    # 检查 venv python（Windows）
    venv_python = pipeline_path / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        missing.append(f"missing pipeline venv python: {venv_python}")
    
    # 检查 recent 目录
    recent_dir = get_pipeline_recent_dir(pipeline_dir)
    recent_exists = recent_dir.exists() and recent_dir.is_dir()
    
    # 搜索 vessel_mesh 文件
    vessel_mesh_candidates, vessel_mesh_selected = search_vessel_mesh_files(pipeline_dir)
    
    # 检查 artifact 根目录
    artifact_roots = get_pipeline_artifact_roots(pipeline_dir)
    
    # 判断是否可用
    pipeline_ok = len(missing) == 0
    
    # 构建 reason
    if not pipeline_ok:
        reason = "pipeline_unavailable: " + "; ".join(missing)
    elif vessel_mesh_selected is None:
        reason = "pipeline_unavailable: vessel_mesh.json not found (run pipeline_refresh_once)"
    else:
        reason = None
    
    return {
        "pipeline_dir": str(pipeline_path),
        "pipeline_ok": pipeline_ok,
        "missing": missing,
        "recent_dir": str(recent_dir),
        "recent_exists": recent_exists,
        "artifact_roots": [str(r) for r in artifact_roots],
        "vessel_mesh_candidates": vessel_mesh_candidates[:10],
        "vessel_mesh_selected": vessel_mesh_selected,
        "reason": reason,
    }


def main():
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="PolarRoute Pipeline 诊断脚本")
    parser.add_argument("--pipeline-dir", type=str, help="Pipeline 目录路径")
    parser.add_argument("pipeline_dir_pos", nargs="?", type=str, help="Pipeline 目录路径（位置参数）")
    
    args = parser.parse_args()
    
    pipeline_dir = args.pipeline_dir or args.pipeline_dir_pos or os.getenv("POLARROUTE_PIPELINE_DIR")
    
    if not pipeline_dir:
        print("错误: 请提供 --pipeline-dir 参数或设置 POLARROUTE_PIPELINE_DIR 环境变量")
        return 1
    
    result = check_pipeline_dir(pipeline_dir)
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    return 0 if result["pipeline_ok"] and result.get("vessel_mesh_selected") else 1


if __name__ == "__main__":
    exit(main())

