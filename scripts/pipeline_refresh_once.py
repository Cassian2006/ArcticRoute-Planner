#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键刷新脚本：PolarRoute Pipeline 状态检查与执行

用途：
- 定期运行以检查 pipeline 状态
- 自动执行 pipeline 以获取最新的 vessel_mesh.json
- 记录最新的 mesh 路径到 reports/pipeline_refresh_last.json

使用示例：
    python -m scripts.pipeline_refresh_once --pipeline-dir "D:\\polarroute-pipeline" --mode status
    python -m scripts.pipeline_refresh_once --pipeline-dir "D:\\polarroute-pipeline" --mode execute --timeout 7200
    python -m scripts.pipeline_refresh_once --pipeline-dir "D:\\polarroute-pipeline" --mode execute-and-status --timeout 7200

Windows 任务计划程序示例：
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "cd D:\\AR_final; .venv\\Scripts\\Activate.ps1; python -m scripts.pipeline_refresh_once --pipeline-dir 'D:\\polarroute-pipeline' --mode execute --timeout 7200"
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def pipeline_status(pipeline_dir: str, short: bool = True) -> Tuple[bool, str]:
    """
    运行 pipeline status 命令。
    
    Args:
        pipeline_dir: Pipeline 目录路径
        short: 是否使用 --short 标志
    
    Returns:
        (success, output)
    """
    import subprocess
    
    try:
        cmd = ["pipeline", "status"]
        if short:
            cmd.append("--short")
        
        result = subprocess.run(
            cmd,
            cwd=pipeline_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        if result.returncode == 0:
            logger.info("✓ Pipeline status 成功")
            return True, result.stdout
        else:
            logger.error(f"✗ Pipeline status 失败: {result.stderr}")
            return False, result.stderr
    
    except Exception as e:
        logger.error(f"✗ Pipeline status 异常: {e}")
        return False, str(e)


def pipeline_execute(pipeline_dir: str, timeout: int = 3600) -> Tuple[bool, str]:
    """
    运行 pipeline execute 命令。
    
    Args:
        pipeline_dir: Pipeline 目录路径
        timeout: 超时时间（秒）
    
    Returns:
        (success, output)
    """
    import subprocess
    
    try:
        cmd = ["pipeline", "execute"]
        
        logger.info(f"开始执行 pipeline execute（超时: {timeout}s）...")
        result = subprocess.run(
            cmd,
            cwd=pipeline_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        if result.returncode == 0:
            logger.info("✓ Pipeline execute 成功")
            return True, result.stdout
        else:
            logger.error(f"✗ Pipeline execute 失败: {result.stderr}")
            return False, result.stderr
    
    except subprocess.TimeoutExpired:
        logger.error(f"✗ Pipeline execute 超时（{timeout}s）")
        return False, f"Pipeline execute 超时（{timeout}s）"
    
    except Exception as e:
        logger.error(f"✗ Pipeline execute 异常: {e}")
        return False, str(e)


def find_latest_vessel_mesh(pipeline_dir: str) -> Path | None:
    """
    在 pipeline 目录中查找最新的 vessel_mesh.json。
    
    搜索路径：
    - {pipeline_dir}/outputs/push/upload/vessel_mesh.json
    - {pipeline_dir}/outputs/vessel_mesh.json
    - {pipeline_dir}/vessel_mesh.json
    
    Args:
        pipeline_dir: Pipeline 目录路径
    
    Returns:
        最新的 vessel_mesh.json 路径，或 None
    """
    pipeline_path = Path(pipeline_dir)
    
    # 搜索候选路径
    candidates = [
        pipeline_path / "outputs" / "push" / "upload" / "vessel_mesh.json",
        pipeline_path / "outputs" / "vessel_mesh.json",
        pipeline_path / "vessel_mesh.json",
    ]
    
    # 查找存在的文件
    existing = [p for p in candidates if p.exists()]
    
    if not existing:
        logger.warning(f"未找到 vessel_mesh.json 在 {pipeline_dir}")
        return None
    
    # 返回最新修改的文件
    latest = max(existing, key=lambda p: p.stat().st_mtime)
    logger.info(f"✓ 找到最新 vessel_mesh.json: {latest}")
    return latest


def save_refresh_report(mesh_path: Path | None, mode: str, status: bool, output: str) -> None:
    """
    保存刷新报告到 reports/pipeline_refresh_last.json。
    
    Args:
        mesh_path: 最新的 vessel_mesh.json 路径
        mode: 运行模式（status / execute / execute-and-status）
        status: 是否成功
        output: 命令输出
    """
    report_dir = Path(__file__).parent.parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = report_dir / "pipeline_refresh_last.json"
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "success": status,
        "mesh_path": str(mesh_path) if mesh_path else None,
        "output_preview": output[:500] if output else None,
    }
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ 报告已保存: {report_file}")
    except Exception as e:
        logger.error(f"✗ 保存报告失败: {e}")


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(
        description="PolarRoute Pipeline 一键刷新脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 检查状态
  python -m scripts.pipeline_refresh_once --pipeline-dir "D:\\polarroute-pipeline" --mode status
  
  # 执行 pipeline
  python -m scripts.pipeline_refresh_once --pipeline-dir "D:\\polarroute-pipeline" --mode execute --timeout 7200
  
  # 执行并检查状态
  python -m scripts.pipeline_refresh_once --pipeline-dir "D:\\polarroute-pipeline" --mode execute-and-status --timeout 7200
        """
    )
    
    parser.add_argument(
        "--pipeline-dir",
        required=True,
        help="PolarRoute Pipeline 目录路径（必需）",
    )
    
    parser.add_argument(
        "--mode",
        choices=["status", "execute", "execute-and-status"],
        default="status",
        help="运行模式（默认: status）",
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="pipeline execute 超时时间（秒，默认: 3600）",
    )
    
    args = parser.parse_args()
    
    pipeline_dir = args.pipeline_dir
    mode = args.mode
    timeout = args.timeout
    
    # 验证 pipeline 目录存在
    if not Path(pipeline_dir).exists():
        logger.error(f"✗ Pipeline 目录不存在: {pipeline_dir}")
        sys.exit(1)
    
    logger.info(f"开始运行 pipeline_refresh_once")
    logger.info(f"  Pipeline 目录: {pipeline_dir}")
    logger.info(f"  运行模式: {mode}")
    if mode in ["execute", "execute-and-status"]:
        logger.info(f"  超时时间: {timeout}s")
    
    overall_success = True
    latest_mesh = None
    
    # 根据模式执行
    if mode == "status":
        success, output = pipeline_status(pipeline_dir, short=True)
        overall_success = success
        if success:
            print(output)
        latest_mesh = find_latest_vessel_mesh(pipeline_dir)
    
    elif mode == "execute":
        success, output = pipeline_execute(pipeline_dir, timeout=timeout)
        overall_success = success
        if success:
            print(output)
        latest_mesh = find_latest_vessel_mesh(pipeline_dir)
    
    elif mode == "execute-and-status":
        # 先执行 status
        logger.info("第 1 步：检查 pipeline 状态...")
        success1, output1 = pipeline_status(pipeline_dir, short=True)
        print(output1)
        
        # 再执行 execute
        logger.info("第 2 步：执行 pipeline execute...")
        success2, output2 = pipeline_execute(pipeline_dir, timeout=timeout)
        print(output2)
        
        # 最后再检查 status
        logger.info("第 3 步：再次检查 pipeline 状态...")
        success3, output3 = pipeline_status(pipeline_dir, short=True)
        print(output3)
        
        overall_success = success2  # 以 execute 的结果为准
        latest_mesh = find_latest_vessel_mesh(pipeline_dir)
    
    # 保存报告
    save_refresh_report(latest_mesh, mode, overall_success, "")
    
    # 打印最终结果
    if latest_mesh:
        logger.info(f"✓ 最新 vessel_mesh.json: {latest_mesh}")
        print(f"\n最新 mesh 路径: {latest_mesh}")
    
    if overall_success:
        logger.info("✓ Pipeline 刷新成功")
        sys.exit(0)
    else:
        logger.error("✗ Pipeline 刷新失败")
        sys.exit(1)


if __name__ == "__main__":
    main()

