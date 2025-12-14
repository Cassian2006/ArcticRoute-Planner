#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PolarRoute 工件解析器 (Phase 5B)

从 pipeline 输出目录中查找最新的工件（vessel_mesh.json 等）。

使用方式：
    from arcticroute.integrations.polarroute_artifacts import find_latest_vessel_mesh
    
    mesh_path = find_latest_vessel_mesh("/path/to/pipeline")
    if mesh_path:
        print(f"找到最新 vessel_mesh: {mesh_path}")
    else:
        print("未找到 vessel_mesh.json")
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def find_latest_vessel_mesh(pipeline_dir: str) -> Optional[str]:
    """
    从 pipeline 目录中查找最新的 vessel_mesh.json。
    
    扫描以下目录（存在就扫）：
    - <pipeline>/outputs
    - <pipeline>/push
    - <pipeline>/upload
    
    匹配文件名：
    - vessel_mesh.json（精确匹配）
    - *vessel*mesh*.json（兜底匹配）
    
    以 mtime（修改时间）最新为准。
    
    Args:
        pipeline_dir: Pipeline 目录路径
    
    Returns:
        最新 vessel_mesh.json 的路径（字符串），或 None（未找到）
    """
    pipeline_path = Path(pipeline_dir)
    
    if not pipeline_path.exists():
        logger.warning(f"Pipeline 目录不存在: {pipeline_dir}")
        return None
    
    # 要扫描的子目录
    scan_dirs = [
        pipeline_path / "outputs",
        pipeline_path / "push",
        pipeline_path / "upload",
    ]
    
    candidates = []
    
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            logger.debug(f"目录不存在（跳过）: {scan_dir}")
            continue
        
        logger.debug(f"扫描目录: {scan_dir}")
        
        # 精确匹配 vessel_mesh.json
        exact_matches = list(scan_dir.rglob("vessel_mesh.json"))
        candidates.extend(exact_matches)
        
        # 兜底匹配 *vessel*mesh*.json
        pattern_matches = list(scan_dir.rglob("*vessel*mesh*.json"))
        candidates.extend(pattern_matches)
    
    if not candidates:
        logger.warning(f"未找到 vessel_mesh.json 在 {pipeline_dir}")
        return None
    
    # 去重
    candidates = list(set(candidates))
    
    # 按 mtime 排序，最新的在前
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    latest = candidates[0]
    logger.info(f"找到最新 vessel_mesh: {latest} (mtime: {latest.stat().st_mtime})")
    
    return str(latest)


def find_latest_route_json(pipeline_dir: str) -> Optional[str]:
    """
    从 pipeline 目录中查找最新的 route.json。
    
    扫描以下目录（存在就扫）：
    - <pipeline>/outputs
    - <pipeline>/push
    - <pipeline>/upload
    
    匹配文件名：
    - route.json（精确匹配）
    - *route*.json（兜底匹配）
    
    以 mtime（修改时间）最新为准。
    
    Args:
        pipeline_dir: Pipeline 目录路径
    
    Returns:
        最新 route.json 的路径（字符串），或 None（未找到）
    """
    pipeline_path = Path(pipeline_dir)
    
    if not pipeline_path.exists():
        logger.warning(f"Pipeline 目录不存在: {pipeline_dir}")
        return None
    
    # 要扫描的子目录
    scan_dirs = [
        pipeline_path / "outputs",
        pipeline_path / "push",
        pipeline_path / "upload",
    ]
    
    candidates = []
    
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            logger.debug(f"目录不存在（跳过）: {scan_dir}")
            continue
        
        logger.debug(f"扫描目录: {scan_dir}")
        
        # 精确匹配 route.json
        exact_matches = list(scan_dir.rglob("route.json"))
        candidates.extend(exact_matches)
        
        # 兜底匹配 *route*.json
        pattern_matches = list(scan_dir.rglob("*route*.json"))
        candidates.extend(pattern_matches)
    
    if not candidates:
        logger.warning(f"未找到 route.json 在 {pipeline_dir}")
        return None
    
    # 去重
    candidates = list(set(candidates))
    
    # 按 mtime 排序，最新的在前
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    latest = candidates[0]
    logger.info(f"找到最新 route.json: {latest} (mtime: {latest.stat().st_mtime})")
    
    return str(latest)


def find_latest_route_config(pipeline_dir: str) -> Optional[str]:
    """
    从 pipeline 目录中查找最新的 route_config.json。
    
    扫描以下目录（存在就扫）：
    - <pipeline>/outputs
    - <pipeline>/push
    - <pipeline>/upload
    
    匹配文件名：
    - route_config.json（精确匹配）
    - *route*config*.json（兜底匹配）
    
    以 mtime（修改时间）最新为准。
    
    Args:
        pipeline_dir: Pipeline 目录路径
    
    Returns:
        最新 route_config.json 的路径（字符串），或 None（未找到）
    """
    pipeline_path = Path(pipeline_dir)
    
    if not pipeline_path.exists():
        logger.warning(f"Pipeline 目录不存在: {pipeline_dir}")
        return None
    
    # 要扫描的子目录
    scan_dirs = [
        pipeline_path / "outputs",
        pipeline_path / "push",
        pipeline_path / "upload",
    ]
    
    candidates = []
    
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            logger.debug(f"目录不存在（跳过）: {scan_dir}")
            continue
        
        logger.debug(f"扫描目录: {scan_dir}")
        
        # 精确匹配 route_config.json
        exact_matches = list(scan_dir.rglob("route_config.json"))
        candidates.extend(exact_matches)
        
        # 兜底匹配 *route*config*.json
        pattern_matches = list(scan_dir.rglob("*route*config*.json"))
        candidates.extend(pattern_matches)
    
    if not candidates:
        logger.warning(f"未找到 route_config.json 在 {pipeline_dir}")
        return None
    
    # 去重
    candidates = list(set(candidates))
    
    # 按 mtime 排序，最新的在前
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    latest = candidates[0]
    logger.info(f"找到最新 route_config.json: {latest} (mtime: {latest.stat().st_mtime})")
    
    return str(latest)

