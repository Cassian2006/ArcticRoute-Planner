"""
数据定位器 - 统一的数据源搜索和发现机制

提供 CMEMS 环境数据和 AIS 密度数据的自动发现功能
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DataLayerInfo:
    """数据层信息"""
    found: bool
    path: Optional[str]
    source: str  # "newenv" | "cache" | "manual" | "missing"
    reason: str
    size_mb: Optional[float] = None
    mtime: Optional[datetime] = None


@dataclass
class AISCandidateInfo:
    """AIS 候选文件信息"""
    path: str
    size_mb: float
    mtime: datetime
    shape: Optional[Tuple[int, int]] = None


# 默认搜索目录
DEFAULT_NEWENV_DIRS = [
    "data_processed/newenv",
    "data/newenv",
    "ArcticRoute/data_processed/newenv",
]

DEFAULT_CACHE_DIRS = [
    "data/cmems_cache",
    "data_processed/cmems_cache",
    "cache/cmems",
]

DEFAULT_AIS_DIRS = [
    "data/ais_density",
    "data_real/ais",
    "data_real/ais/density",
    "data_real/ais/derived",
    "ArcticRoute/data_processed/ais",
    "data_processed/ais",
]


def _find_file_in_dirs(
    dirs: List[str],
    patterns: List[str],
    recursive: bool = False
) -> Optional[Path]:
    """
    在指定目录中搜索匹配模式的文件
    
    Args:
        dirs: 目录列表
        patterns: 文件名模式列表（支持通配符）
        recursive: 是否递归搜索
    
    Returns:
        找到的第一个文件路径，如果未找到则返回 None
    """
    for dir_str in dirs:
        dir_path = Path(dir_str)
        if not dir_path.exists():
            continue
        
        for pattern in patterns:
            if recursive:
                matches = list(dir_path.rglob(pattern))
            else:
                matches = list(dir_path.glob(pattern))
            
            if matches:
                # 返回最新的文件
                return max(matches, key=lambda p: p.stat().st_mtime)
    
    return None


def _get_file_info(path: Path) -> Tuple[float, datetime]:
    """获取文件大小和修改时间"""
    stat = path.stat()
    size_mb = stat.st_size / (1024 * 1024)
    mtime = datetime.fromtimestamp(stat.st_mtime)
    return size_mb, mtime


@lru_cache(maxsize=1)
def _load_cmems_index() -> Optional[Dict]:
    """?? CMEMS newenv ????"""
    index_candidates = [
        Path("data_processed/newenv/cmems_newenv_index.json"),
        Path("reports/cmems_newenv_index.json"),
    ]
    for index_path in index_candidates:
        if not index_path.exists():
            continue
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load CMEMS index {index_path}: {e}")
    return None


def _get_index_layer_path(index: Dict, key: str) -> Optional[str]:
    """?????????????????"""
    if not isinstance(index, dict):
        return None
    if "layers" in index and isinstance(index["layers"], dict):
        layer_entry = index["layers"].get(key)
        if isinstance(layer_entry, dict):
            return layer_entry.get("target_path") or layer_entry.get("path")
        if isinstance(layer_entry, str):
            return layer_entry
    if key in index and isinstance(index[key], str):
        return index[key]
    return None


def discover_cmems_layers(
    manual_paths: Optional[Dict[str, str]] = None,
    newenv_dirs: Optional[List[str]] = None,
    cache_dirs: Optional[List[str]] = None,
) -> Dict[str, DataLayerInfo]:
    """
    发现 CMEMS 环境数据层
    
    Args:
        manual_paths: 手动指定的路径字典 {"sic": "/path/to/sic.nc", ...}
        newenv_dirs: newenv 目录列表（覆盖默认值）
        cache_dirs: cache 目录列表（覆盖默认值）
    
    Returns:
        数据层信息字典 {"sic": DataLayerInfo, "swh": DataLayerInfo, ...}
    """
    if manual_paths is None:
        manual_paths = {}
    
    if newenv_dirs is None:
        newenv_dirs = DEFAULT_NEWENV_DIRS
    
    if cache_dirs is None:
        cache_dirs = DEFAULT_CACHE_DIRS
    
    # 加载索引文件
    index = _load_cmems_index()
    
    layers = {}
    
    # 定义各层的搜索配置
    layer_configs = {
        "sic": {
            "newenv_patterns": ["ice_copernicus_sic.nc", "sic_latest.nc", "sic.nc"],
            "cache_patterns": ["*sic*.nc", "*siconc*.nc"],
            "index_key": "sic",
        },
        "swh": {
            "newenv_patterns": ["wave_swh.nc", "swh_latest.nc", "swh.nc"],
            "cache_patterns": ["*swh*.nc", "*wave*.nc", "*wav*.nc"],
            "index_key": "swh",
        },
        "sit": {
            "newenv_patterns": ["ice_thickness.nc", "sit_latest.nc", "sit.nc", "ice_copernicus_sit.nc"],
            "cache_patterns": ["*thickness*.nc", "*sit*.nc"],
            "index_key": "sit",
        },
        "drift": {
            "newenv_patterns": ["ice_drift.nc", "drift_latest.nc", "drift.nc"],
            "cache_patterns": ["*drift*.nc", "*uice*.nc", "*vice*.nc", "*ice_velocity*.nc"],
            "index_key": "drift",
        },
    }
    
    for layer_name, config in layer_configs.items():
        # 1. 检查手动路径
        if layer_name in manual_paths:
            manual_path = Path(manual_paths[layer_name])
            if manual_path.exists():
                size_mb, mtime = _get_file_info(manual_path)
                layers[layer_name] = DataLayerInfo(
                    found=True,
                    path=str(manual_path),
                    source="manual",
                    reason="User-specified path",
                    size_mb=size_mb,
                    mtime=mtime,
                )
                continue
            else:
                layers[layer_name] = DataLayerInfo(
                    found=False,
                    path=None,
                    source="missing",
                    reason=f"Manual path does not exist: {manual_path}",
                )
                continue
        
        # 2. 搜索 newenv 目录
        newenv_file = _find_file_in_dirs(newenv_dirs, config["newenv_patterns"])
        if newenv_file:
            size_mb, mtime = _get_file_info(newenv_file)
            layers[layer_name] = DataLayerInfo(
                found=True,
                path=str(newenv_file),
                source="newenv",
                reason=f"Found in newenv: {newenv_file.name}",
                size_mb=size_mb,
                mtime=mtime,
            )
            continue
        
        # 3. 搜索 cache 目录
        cache_file = _find_file_in_dirs(cache_dirs, config["cache_patterns"], recursive=True)
        if cache_file:
            size_mb, mtime = _get_file_info(cache_file)
            layers[layer_name] = DataLayerInfo(
                found=True,
                path=str(cache_file),
                source="cache",
                reason=f"Found in cache: {cache_file.name}",
                size_mb=size_mb,
                mtime=mtime,
            )
            continue
        
        # 4. 检查索引文件
        index_path_str = _get_index_layer_path(index, config["index_key"]) if index else None
        if index_path_str:
            index_path = Path(index_path_str)
            if index_path.exists():
                size_mb, mtime = _get_file_info(index_path)
                layers[layer_name] = DataLayerInfo(
                    found=True,
                    path=str(index_path),
                    source="newenv",
                    reason=f"Found via index: {index_path.name}",
                    size_mb=size_mb,
                    mtime=mtime,
                )
                continue
        
        # 5. 未找到
        searched_dirs = newenv_dirs + cache_dirs
        layers[layer_name] = DataLayerInfo(
            found=False,
            path=None,
            source="missing",
            reason=f"Not found in newenv or cache. Searched: {', '.join(searched_dirs[:3])}...",
        )
    
    return layers


def discover_ais_density_nc(
    search_dirs: Optional[List[str]] = None,
    additional_dirs: Optional[List[str]] = None,
) -> Tuple[List[AISCandidateInfo], Optional[AISCandidateInfo]]:
    """
    发现 AIS 密度 NetCDF 文件
    
    Args:
        search_dirs: 搜索目录列表（覆盖默认值）
        additional_dirs: 额外的搜索目录（追加到默认值）
    
    Returns:
        (候选文件列表, 最佳文件)
        候选文件按 mtime 从新到旧排序
    """
    if search_dirs is None:
        search_dirs = DEFAULT_AIS_DIRS.copy()
    
    if additional_dirs:
        search_dirs.extend(additional_dirs)
    
    candidates = []
    
    # 关键词匹配（大小写不敏感）
    keywords = ["density", "ais", "traffic", "corridor"]
    
    for dir_str in search_dirs:
        dir_path = Path(dir_str)
        if not dir_path.exists():
            continue
        
        # 递归扫描所有 .nc 文件
        for nc_file in dir_path.rglob("*.nc"):
            if not nc_file.is_file():
                continue
            
            # 检查文件名是否包含关键词
            filename_lower = nc_file.name.lower()
            has_keyword = any(kw in filename_lower for kw in keywords)
            
            try:
                size_mb, mtime = _get_file_info(nc_file)
                
                # 尝试获取 shape（可选）
                shape = None
                try:
                    import xarray as xr
                    with xr.open_dataset(nc_file) as ds:
                        # 尝试获取第一个数据变量的 shape
                        if ds.data_vars:
                            first_var = list(ds.data_vars.values())[0]
                            shape = first_var.shape
                except Exception:
                    pass
                
                candidates.append(AISCandidateInfo(
                    path=str(nc_file),
                    size_mb=size_mb,
                    mtime=mtime,
                    shape=shape,
                ))
            except Exception as e:
                logger.warning(f"Failed to process {nc_file}: {e}")
                continue
    
    # 按 mtime 排序（新到旧）
    candidates.sort(key=lambda c: c.mtime, reverse=True)
    
    # 选择最佳文件（最新的）
    best = candidates[0] if candidates else None
    
    return candidates, best


def clear_discovery_caches():
    """清理数据发现缓存"""
    # 清理 lru_cache
    _load_cmems_index.cache_clear()
    
    logger.info("Data discovery caches cleared")


# 便捷函数：获取 CMEMS 层的简单状态
def get_cmems_status_summary(layers: Dict[str, DataLayerInfo]) -> Dict[str, any]:
    """
    获取 CMEMS 层状态摘要
    
    Returns:
        {"found_count": int, "missing_count": int, "layers": {...}}
    """
    found_count = sum(1 for layer in layers.values() if layer.found)
    missing_count = len(layers) - found_count
    
    return {
        "found_count": found_count,
        "missing_count": missing_count,
        "total_count": len(layers),
        "layers": layers,
    }


# 便捷函数：获取 AIS 搜索摘要
def get_ais_search_summary(
    candidates: List[AISCandidateInfo],
    search_dirs: List[str]
) -> Dict[str, any]:
    """
    获取 AIS 搜索摘要
    
    Returns:
        {"found_count": int, "searched_dirs": [...], "candidates": [...]}
    """
    return {
        "found_count": len(candidates),
        "searched_dirs": search_dirs,
        "candidates": candidates,
    }

