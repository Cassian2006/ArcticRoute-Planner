#!/usr/bin/env python
"""
CMEMS 工具函数库
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Optional


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
        return None
    
    # 查找匹配的 nc 文件
    matching_files = list(cache_path.glob(f"{pattern}*.nc"))
    if not matching_files:
        return None
    
    # 按修改时间排序，返回最新的
    latest = max(matching_files, key=lambda p: p.stat().st_mtime)
    return latest


def load_resolved_config() -> dict:
    """
    加载已解析的 CMEMS 配置
    
    Returns:
        配置字典，包含 sic 和 wav 的 dataset_id 和 variables
    """
    config_path = Path("reports/cmems_resolved.json")
    if not config_path.exists():
        raise FileNotFoundError(
            f"配置文件不存在: {config_path}\n"
            "请先运行: python scripts/cmems_resolve.py"
        )
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_refresh_record() -> Optional[dict]:
    """
    加载最后一次刷新的元数据记录
    
    Returns:
        刷新记录字典，或 None 如果文件不存在
    """
    record_path = Path("reports/cmems_refresh_last.json")
    if not record_path.exists():
        return None
    
    with open(record_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_sic_variable(config: dict) -> Optional[str]:
    """
    从配置中获取 SIC 变量名
    
    Args:
        config: 从 load_resolved_config() 返回的配置
    
    Returns:
        变量名，或 None
    """
    sic_config = config.get("sic", {})
    variables = sic_config.get("variables", [])
    if variables:
        # 优先返回 "sic"，否则返回第一个
        for var in variables:
            if "sic" in var.lower() and "uncertainty" not in var.lower():
                return var
        return variables[0]
    return None


def get_swh_variable(config: dict) -> Optional[str]:
    """
    从配置中获取 SWH（有效波高）变量名
    
    Args:
        config: 从 load_resolved_config() 返回的配置
    
    Returns:
        变量名，或 None
    """
    wav_config = config.get("wav", {})
    variables = wav_config.get("variables", [])
    if variables:
        # 优先查找包含 "significant_height" 的变量
        for var in variables:
            if "significant_height" in var.lower():
                return var
        # 其次查找 "swh" 或 "hs"
        for var in variables:
            if "swh" in var.lower() or "hs" in var.lower():
                return var
        return variables[0]
    return None


if __name__ == "__main__":
    # 测试
    try:
        config = load_resolved_config()
        print(f"[OK] 配置已加载")
        print(f"SIC 变量: {get_sic_variable(config)}")
        print(f"SWH 变量: {get_swh_variable(config)}")
        
        latest_sic = find_latest_nc(pattern="sic")
        print(f"最新 SIC 文件: {latest_sic}")
        
        latest_swh = find_latest_nc(pattern="swh")
        print(f"最新 SWH 文件: {latest_swh}")
        
        record = load_refresh_record()
        if record:
            print(f"最后刷新时间: {record.get('timestamp')}")
    except Exception as e:
        print(f"[ERROR] {e}")

