#!/usr/bin/env python
"""
CMEMS 数据同步到 newenv 目录

将最新的 SIC 和 SWH nc 文件复制到标准位置，供规划器使用
"""
import shutil
import logging
from pathlib import Path
from typing import Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def find_latest_nc(cache_dir: str = "data/cmems_cache", pattern: str = "sic") -> Optional[Path]:
    """
    从缓存目录中查找最新的 nc 文件
    
    Args:
        cache_dir: 缓存目录路径
        pattern: 文件名模式（sic 或 swh）
    
    Returns:
        最新的 nc 文件路径，或 None 如果找不到
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        logger.warning(f"缓存目录不存在: {cache_path}")
        return None
    
    # 查找匹配的 nc 文件
    matching_files = list(cache_path.glob(f"{pattern}*.nc"))
    if not matching_files:
        logger.warning(f"未找到匹配 '{pattern}' 的 nc 文件")
        return None
    
    # 按修改时间排序，返回最新的
    latest = max(matching_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"找到最新 {pattern} 文件: {latest.name}")
    return latest


def sync_to_newenv(
    cache_dir: str = "data/cmems_cache",
    newenv_dir: str = "ArcticRoute/data_processed/newenv",
) -> bool:
    """
    将最新的 CMEMS 数据同步到 newenv 目录
    
    Args:
        cache_dir: 缓存目录
        newenv_dir: 目标 newenv 目录
    
    Returns:
        True 如果至少一个文件成功同步
    """
    newenv_path = Path(newenv_dir)
    newenv_path.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    
    # 同步 SIC
    logger.info("=" * 60)
    logger.info("同步 SIC 数据")
    logger.info("=" * 60)
    
    sic_src = find_latest_nc(cache_dir, "sic")
    if sic_src:
        sic_dst = newenv_path / "ice_copernicus_sic.nc"
        try:
            shutil.copy2(sic_src, sic_dst)
            logger.info(f"✅ SIC 已复制: {sic_dst}")
            success_count += 1
        except Exception as e:
            logger.error(f"❌ SIC 复制失败: {e}")
    else:
        logger.warning("未找到 SIC 源文件")
    
    # 同步 SWH
    logger.info("=" * 60)
    logger.info("同步 SWH 数据")
    logger.info("=" * 60)
    
    swh_src = find_latest_nc(cache_dir, "swh")
    if swh_src:
        swh_dst = newenv_path / "wave_swh.nc"
        try:
            shutil.copy2(swh_src, swh_dst)
            logger.info(f"✅ SWH 已复制: {swh_dst}")
            success_count += 1
        except Exception as e:
            logger.error(f"❌ SWH 复制失败: {e}")
    else:
        logger.warning("未找到 SWH 源文件")
    
    logger.info("=" * 60)
    logger.info(f"同步完成: {success_count} 个文件成功")
    logger.info("=" * 60)
    
    return success_count > 0


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="CMEMS 数据同步到 newenv")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cmems_cache",
        help="缓存目录（默认 data/cmems_cache）",
    )
    parser.add_argument(
        "--newenv-dir",
        type=str,
        default="ArcticRoute/data_processed/newenv",
        help="目标 newenv 目录（默认 ArcticRoute/data_processed/newenv）",
    )
    
    args = parser.parse_args()
    
    success = sync_to_newenv(args.cache_dir, args.newenv_dir)
    sys.exit(0 if success else 1)

