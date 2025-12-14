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
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


def _ensure_reports_dir() -> Path:
    """确保 reports 目录存在"""
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


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
    
    # 构建命令
    cmd = ["pipeline", command, pipeline_dir]
    if extra_args:
        cmd.extend(extra_args)
    
    logger.debug(f"执行 pipeline 命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        output = result.stdout
        error = result.stderr
        
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
    
    except FileNotFoundError:
        logger.error("pipeline 命令未找到。请确保已安装 PolarRoute-pipeline")
        return False, "pipeline command not found"
    
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

