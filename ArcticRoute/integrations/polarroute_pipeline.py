#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PolarRoute Pipeline 集成 (Phase 5B)

最小但可靠的 subprocess 封装层，支持 pipeline 官方命令：
- pipeline build <path-to-pipeline-directory>
- pipeline status <path-to-pipeline-directory> [--short]
- pipeline execute <path-to-pipeline-directory>
- pipeline reset <path-to-pipeline-directory>
- pipeline halt <path-to-pipeline-directory>

所有命令的 stdout/stderr 都会写到 reports/polarroute_pipeline_last_{out,err}.log

使用方式：
    from arcticroute.integrations.polarroute_pipeline import (
        pipeline_build, pipeline_status, pipeline_execute, 
        pipeline_reset, pipeline_halt
    )
    
    # 获取 pipeline 状态
    success, output = pipeline_status("/path/to/pipeline", short=True)
    
    # 执行 pipeline
    success, output = pipeline_execute("/path/to/pipeline")
"""

import subprocess
import logging
import shutil
import os
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


def _ensure_reports_dir() -> Path:
    """确保 reports 目录存在"""
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def _resolve_pipeline_executable(pipeline_dir: Path) -> str | None:
    """寻找可用的 pipeline 可执行文件。
    优先顺序：
    1) PATH 中的 pipeline
    2) <pipeline_dir>/.venv/Scripts/pipeline.exe (Windows)
    3) <pipeline_dir>/.venv/bin/pipeline (POSIX)
    """
    # 1) PATH
    exe = shutil.which("pipeline")
    if exe:
        return exe

    # 2) 本地 venv 典型位置
    candidates = [
        pipeline_dir / ".venv" / "Scripts" / "pipeline.exe",
        pipeline_dir / ".venv" / "Scripts" / "pipeline.EXE",
        pipeline_dir / ".venv" / "bin" / "pipeline",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    return None


def _run_pipeline_command(
    command: str,
    pipeline_dir: str,
    extra_args: list[str] | None = None,
    timeout: int = 300,
) -> Tuple[bool, str]:
    """
    运行 pipeline 命令的通用函数。
    
    Args:
        command: 命令名称 (build/status/execute/reset/halt)
        pipeline_dir: Pipeline 目录路径
        extra_args: 额外参数列表（如 ["--short"]）
        timeout: 超时时间（秒）
    
    Returns:
        (success, output) - 成功标志和输出文本
    """
    reports_dir = _ensure_reports_dir()

    pipedir = Path(pipeline_dir)
    if not pipedir.exists():
        return False, f"pipeline_dir 不存在: {pipedir}"

    # 寻找 pipeline 可执行文件
    pipeline_exe = _resolve_pipeline_executable(pipedir)
    if not pipeline_exe:
        logger.error("未找到 pipeline 可执行文件 (不在 PATH，且 .venv 下未发现)")
        return False, "pipeline command not found (try activating pipeline venv or install to PATH)"

    # 在 Windows 下，部分 CLI 会把传入路径与 CWD 直接拼接，
    # 为避免路径畸形，强制切换到 pipeline_dir 并以 "." 作为目标参数
    cmd = [pipeline_exe, command, "."]
    if extra_args:
        cmd.extend(extra_args)

    logger.debug(f"执行 pipeline 命令: {' '.join(cmd)} (cwd={pipedir})")

    try:
        # 构造环境变量，确保 pipeline venv 的 Scripts 在 PATH 最前
        env = {**os.environ}
        scripts_dir_win = pipedir / ".venv" / "Scripts"
        scripts_dir_posix = pipedir / ".venv" / "bin"
        path_parts = []
        if scripts_dir_win.exists():
            path_parts.append(str(scripts_dir_win))
        if scripts_dir_posix.exists():
            path_parts.append(str(scripts_dir_posix))
        path_parts.append(env.get("PATH", ""))
        env["PATH"] = os.pathsep.join(path_parts)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(pipedir),
            env=env,
        )

        output = result.stdout or ""
        error = result.stderr or ""

        # 写入日志文件
        out_log = reports_dir / "polarroute_pipeline_last_out.log"
        err_log = reports_dir / "polarroute_pipeline_last_err.log"

        out_log.write_text(output)
        err_log.write_text(error)

        success = result.returncode == 0

        if success:
            logger.info(f"✓ pipeline {command} 成功")
        else:
            logger.error(
                f"✗ pipeline {command} 失败 (返回码 {result.returncode})\n"
                f"详见 {out_log} 和 {err_log}"
            )

        # 返回 stdout + stderr 合并
        combined_output = output
        if error:
            combined_output += f"\n[stderr]\n{error}"

        return success, combined_output

    except subprocess.TimeoutExpired:
        logger.error(f"pipeline {command} 超时（{timeout}秒）")
        return False, f"Command timeout after {timeout} seconds"

    except Exception as e:
        logger.error(f"pipeline {command} 执行异常: {e}")
        return False, str(e)


def pipeline_build(pipeline_dir: str, timeout: int = 600) -> Tuple[bool, str]:
    """
    执行 pipeline build 命令。
    
    Args:
        pipeline_dir: Pipeline 目录路径
        timeout: 超时时间（秒，默认 10 分钟）
    
    Returns:
        (success, output)
    """
    return _run_pipeline_command("build", pipeline_dir, timeout=timeout)


def pipeline_status(
    pipeline_dir: str,
    short: bool = True,
    timeout: int = 30,
) -> Tuple[bool, str]:
    """
    执行 pipeline status 命令。
    
    Args:
        pipeline_dir: Pipeline 目录路径
        short: 是否使用 --short 标志（简短输出）
        timeout: 超时时间（秒）
    
    Returns:
        (success, output)
    """
    extra_args = ["--short"] if short else []
    return _run_pipeline_command("status", pipeline_dir, extra_args, timeout=timeout)


def pipeline_execute(pipeline_dir: str, timeout: int = 600) -> Tuple[bool, str]:
    """
    执行 pipeline execute 命令。
    
    Args:
        pipeline_dir: Pipeline 目录路径
        timeout: 超时时间（秒，默认 10 分钟）
    
    Returns:
        (success, output)
    """
    return _run_pipeline_command("execute", pipeline_dir, timeout=timeout)


def pipeline_reset(pipeline_dir: str, timeout: int = 60) -> Tuple[bool, str]:
    """
    执行 pipeline reset 命令。
    
    Args:
        pipeline_dir: Pipeline 目录路径
        timeout: 超时时间（秒）
    
    Returns:
        (success, output)
    """
    return _run_pipeline_command("reset", pipeline_dir, timeout=timeout)


def pipeline_halt(pipeline_dir: str, timeout: int = 60) -> Tuple[bool, str]:
    """
    执行 pipeline halt 命令。
    
    Args:
        pipeline_dir: Pipeline 目录路径
        timeout: 超时时间（秒）
    
    Returns:
        (success, output)
    """
    return _run_pipeline_command("halt", pipeline_dir, timeout=timeout)

