#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PolarRoute Pipeline 医生脚本 (Phase 5B)

检测 pipeline CLI 是否可用，并可选地运行 pipeline status 诊断。

使用方式：
    python -m scripts.polarroute_pipeline_doctor
    python -m scripts.polarroute_pipeline_doctor --pipeline-dir "D:\\polarroute-pipeline"
"""

import argparse
import subprocess
import shutil
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def check_pipeline_cli_available() -> tuple[bool, str | None]:
    """
    检测 pipeline CLI 是否可用。
    
    Returns:
        (is_available, cli_path)
    """
    # 尝试找到 pipeline 命令
    pipeline_path = shutil.which("pipeline")
    
    if pipeline_path:
        return True, pipeline_path
    
    return False, None


def check_pipeline_help() -> tuple[bool, str]:
    """
    运行 pipeline --help 检查命令可用性。
    
    Returns:
        (success, output)
    """
    try:
        result = subprocess.run(
            ["pipeline", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


def check_pipeline_status_help() -> tuple[bool, str]:
    """
    运行 pipeline status --help 检查命令可用性。
    
    Returns:
        (success, output)
    """
    try:
        result = subprocess.run(
            ["pipeline", "status", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


def run_pipeline_status_short(pipeline_dir: str) -> tuple[bool, str]:
    """
    运行 pipeline status <dir> --short 命令。
    
    Args:
        pipeline_dir: Pipeline 目录路径
    
    Returns:
        (success, output)
    """
    try:
        result = subprocess.run(
            ["pipeline", "status", pipeline_dir, "--short"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        return result.returncode == 0, output
    except Exception as e:
        return False, str(e)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="PolarRoute Pipeline 医生脚本 - 检测 pipeline CLI 可用性"
    )
    parser.add_argument(
        "--pipeline-dir",
        type=str,
        default=None,
        help="Pipeline 目录路径（可选）。若给定，将运行 pipeline status <dir> --short",
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("PolarRoute Pipeline 医生脚本")
    logger.info("=" * 70)
    
    # 1. 检查 pipeline CLI 可用性
    logger.info("\n[1] 检查 pipeline CLI 可用性...")
    is_available, cli_path = check_pipeline_cli_available()
    
    if is_available:
        logger.info(f"✓ pipeline CLI 已找到: {cli_path}")
    else:
        logger.warning("✗ pipeline CLI 未找到。请确保已安装 PolarRoute-pipeline")
    
    # 2. 运行 pipeline --help
    logger.info("\n[2] 运行 pipeline --help...")
    success, output = check_pipeline_help()
    
    if success:
        logger.info("✓ pipeline --help 成功")
        logger.debug(f"输出:\n{output[:200]}...")
    else:
        logger.warning(f"✗ pipeline --help 失败: {output[:200]}")
    
    # 3. 运行 pipeline status --help
    logger.info("\n[3] 运行 pipeline status --help...")
    success, output = check_pipeline_status_help()
    
    if success:
        logger.info("✓ pipeline status --help 成功")
        logger.debug(f"输出:\n{output[:200]}...")
    else:
        logger.warning(f"✗ pipeline status --help 失败: {output[:200]}")
    
    # 4. 若给定 pipeline_dir，运行 pipeline status <dir> --short
    if args.pipeline_dir:
        logger.info(f"\n[4] 运行 pipeline status --short (dir={args.pipeline_dir})...")
        
        pipeline_dir_path = Path(args.pipeline_dir)
        if not pipeline_dir_path.exists():
            logger.error(f"✗ Pipeline 目录不存在: {args.pipeline_dir}")
            sys.exit(1)
        
        success, output = run_pipeline_status_short(args.pipeline_dir)
        
        if success:
            logger.info("✓ pipeline status --short 成功")
            logger.info(f"输出:\n{output}")
        else:
            logger.error(f"✗ pipeline status --short 失败: {output}")
            sys.exit(1)
    else:
        logger.info("\n[4] 跳过 pipeline status --short（未指定 --pipeline-dir）")
    
    # 总结
    logger.info("\n" + "=" * 70)
    logger.info("诊断完成")
    logger.info("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


