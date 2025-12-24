"""
一键刷新 PolarRoute Pipeline 并生成 vessel_mesh.json

用法:
    python -m scripts.pipeline_refresh_once --pipeline-dir <path> --mode status
    python -m scripts.pipeline_refresh_once --pipeline-dir <path> --mode execute-and-status --timeout 7200
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional


def get_pipeline_recent_dir(pipeline_dir: str) -> Path:
    """
    获取 pipeline 的真实 recent 目录。
    
    Windows 安全：不使用 drive 参与目录名。
    """
    pipeline_path = Path(pipeline_dir)
    return pipeline_path / "workflow-manager" / "recent.operational-polarroute"


def search_vessel_mesh_files(pipeline_dir: str) -> tuple[list[str], Optional[str]]:
    """
    搜索 vessel_mesh*.json 文件。
    
    Windows 安全：使用 Path 对象，不拼接字符串。
    """
    pipeline_path = Path(pipeline_dir)
    if not pipeline_path.exists():
        return [], None
    
    # 搜索根目录（按优先级）
    search_dirs = [
        get_pipeline_recent_dir(pipeline_dir),
        pipeline_path / "workflow-manager",
        pipeline_path / "outputs",
        pipeline_path,
    ]
    
    candidates = []
    for search_dir in search_dirs:
        if not search_dir.exists() or not search_dir.is_dir():
            continue
        
        # 递归搜索 vessel_mesh*.json
        for pattern in ["vessel_mesh*.json", "**/vessel_mesh*.json"]:
            try:
                for file_path in search_dir.glob(pattern):
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


def check_status(pipeline_dir: str) -> Dict[str, Any]:
    """检查当前状态，不执行 pipeline。"""
    vessel_mesh_candidates, vessel_mesh_selected = search_vessel_mesh_files(pipeline_dir)
    
    result = {
        "pipeline_dir": pipeline_dir,
        "vessel_mesh_found": vessel_mesh_selected is not None,
        "vessel_mesh_path": vessel_mesh_selected,
        "vessel_mesh_candidates": vessel_mesh_candidates[:10],  # 最多10个
        "timestamp": time.time(),
    }
    
    if vessel_mesh_selected:
        mesh_path = Path(vessel_mesh_selected)
        result["vessel_mesh_size"] = mesh_path.stat().st_size
        result["vessel_mesh_mtime"] = mesh_path.stat().st_mtime
    
    return result


def execute_pipeline(pipeline_dir: str, timeout: int = 7200) -> Dict[str, Any]:
    """
    执行 pipeline 生成 vessel_mesh.json。
    
    Args:
        pipeline_dir: Pipeline 目录路径
        timeout: 超时时间（秒），默认 7200 秒（2小时）
    
    Returns:
        执行结果字典
    """
    pipeline_path = Path(pipeline_dir)
    if not pipeline_path.exists():
        return {
            "success": False,
            "error": f"Pipeline 目录不存在: {pipeline_dir}",
        }
    
    # 检查必需文件
    venv_python = pipeline_path / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        return {
            "success": False,
            "error": f"Pipeline venv Python 未找到: {venv_python}",
        }
    
    # 检查 jug 可执行文件
    jug_exe = pipeline_path / ".venv" / "Scripts" / "jug.exe"
    if not jug_exe.exists():
        # 如果没有 jug.exe，尝试使用 python -m jug
        jug_exe = None
    
    operational_py = pipeline_path / "workflow-manager" / "operational-polarroute.py"
    if not operational_py.exists():
        return {
            "success": False,
            "error": f"operational-polarroute.py 未找到: {operational_py}",
        }
    
    # 设置环境变量（Windows 安全：使用绝对路径字符串，不拼接）
    env = os.environ.copy()
    env["PIPELINE_DIRECTORY"] = str(pipeline_path.resolve())
    env["SCRIPTS_DIRECTORY"] = str((pipeline_path / "scripts").resolve())
    
    # 执行 pipeline
    print(f"[执行] 开始运行 pipeline: {pipeline_dir}")
    print(f"[执行] 超时设置: {timeout} 秒")
    
    start_time = time.time()
    
    try:
        # 使用 jug execute 运行 pipeline
        if jug_exe and jug_exe.exists():
            cmd = [
                str(jug_exe),
                "execute",
                str(operational_py),
            ]
        else:
            # 回退到 python -m jug
            cmd = [
                str(venv_python),
                "-m", "jug",
                "execute",
                str(operational_py),
            ]
        
        result = subprocess.run(
            cmd,
            cwd=str(pipeline_path.resolve()),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"[执行] Pipeline 执行成功（耗时: {elapsed_time:.1f} 秒）")
            return {
                "success": True,
                "elapsed_time": elapsed_time,
                "returncode": result.returncode,
            }
        else:
            print(f"[执行] Pipeline 执行失败（返回码: {result.returncode}）")
            return {
                "success": False,
                "error": f"Pipeline 返回非零退出码: {result.returncode}",
                "elapsed_time": elapsed_time,
                "returncode": result.returncode,
                "stderr": result.stderr[:1000] if result.stderr else None,
            }
    
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"[执行] Pipeline 执行超时（{timeout} 秒）")
        return {
            "success": False,
            "error": f"Pipeline 执行超时（{timeout} 秒）",
            "elapsed_time": elapsed_time,
        }
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"[执行] Pipeline 执行出错: {e}")
        return {
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed_time,
        }


def execute_and_status(pipeline_dir: str, timeout: int = 7200) -> Dict[str, Any]:
    """
    执行 pipeline 并检查状态。
    
    Args:
        pipeline_dir: Pipeline 目录路径
        timeout: 超时时间（秒），默认 7200 秒（2小时）
    
    Returns:
        完整结果字典
    """
    # 先检查当前状态
    status_before = check_status(pipeline_dir)
    
    # 如果已经有 vessel_mesh.json，询问是否继续
    if status_before["vessel_mesh_found"]:
        print(f"[状态] 已找到 vessel_mesh.json: {status_before['vessel_mesh_path']}")
        print("[执行] 继续执行 pipeline 以生成新版本...")
    
    # 执行 pipeline
    execute_result = execute_pipeline(pipeline_dir, timeout=timeout)
    
    # 再次检查状态
    status_after = check_status(pipeline_dir)
    
    # 合并结果
    result = {
        "pipeline_dir": pipeline_dir,
        "status_before": status_before,
        "execute_result": execute_result,
        "status_after": status_after,
        "timestamp": time.time(),
    }
    
    return result


def save_result(result: Dict[str, Any], output_file: Optional[Path] = None) -> Path:
    """保存结果到文件。"""
    if output_file is None:
        reports_dir = Path(__file__).resolve().parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        output_file = reports_dir / "pipeline_refresh_last.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"[保存] 结果已保存到: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="一键刷新 PolarRoute Pipeline 并生成 vessel_mesh.json"
    )
    parser.add_argument(
        "--pipeline-dir",
        type=str,
        required=True,
        help="Pipeline 目录路径",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["status", "execute-and-status"],
        default="status",
        help="运行模式: status (仅检查) 或 execute-and-status (执行并检查)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="超时时间（秒），默认 7200 秒（2小时）",
    )
    
    args = parser.parse_args()
    
    pipeline_dir = os.path.abspath(args.pipeline_dir)
    
    if args.mode == "status":
        print(f"[状态] 检查 pipeline 状态: {pipeline_dir}")
        result = check_status(pipeline_dir)
        
        if result["vessel_mesh_found"]:
            print(f"[OK] 找到 vessel_mesh.json: {result['vessel_mesh_path']}")
            print(f"   大小: {result['vessel_mesh_size']} 字节")
        else:
            print("[WARN] 未找到 vessel_mesh.json")
            if result["vessel_mesh_candidates"]:
                print(f"   找到 {len(result['vessel_mesh_candidates'])} 个候选文件（但大小可能为0）")
        
        save_result(result)
    
    elif args.mode == "execute-and-status":
        print(f"[执行] 执行 pipeline 并检查状态: {pipeline_dir}")
        result = execute_and_status(pipeline_dir, timeout=args.timeout)
        
        save_result(result)
        
        # 显示最终状态
        status_after = result["status_after"]
        if status_after["vessel_mesh_found"]:
            print(f"\n[OK] 成功！vessel_mesh.json 路径: {status_after['vessel_mesh_path']}")
        else:
            print("\n[ERROR] 未找到 vessel_mesh.json，可能需要检查 pipeline 执行日志")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
