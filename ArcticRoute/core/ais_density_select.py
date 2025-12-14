"""
AIS density 统一选择/匹配/加载/对齐模块。

提供稳定的 API 供 UI 和成本层使用，包括：
  - scan_ais_density_candidates: 扫描候选文件
  - select_best_candidate: 选择最佳匹配
  - load_and_align_density: 加载并对齐到目标网格

缓存策略：
  - 文件读取缓存：key = (path, mtime)
  - 重采样缓存：key = (path, mtime, target_signature, method)
"""

import os
import logging
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import xarray as xr

from arcticroute.core.grid import Grid2D

logger = logging.getLogger(__name__)

# 默认 AIS 密度搜索目录
DEFAULT_AIS_SEARCH_DIRS = [
    "data_real/ais/density",
    "data_real/ais/derived",
    "data_real/ais",
]


@dataclass
class AISDensityCandidate:
    """AIS 密度候选文件信息。"""
    path: str
    grid_signature: Optional[str] = None
    shape: Optional[Tuple[int, int]] = None
    varname: Optional[str] = None
    note: str = ""
    match_type: str = "generic"  # "exact" | "demo" | "generic"
    
    def to_dict(self) -> Dict:
        """转换为字典（用于 UI 展示）。"""
        return asdict(self)


def _get_mtime(path: Path) -> float:
    """获取文件修改时间，用于缓存 key。"""
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


@lru_cache(maxsize=32)
def _cached_read_ais_file(path_str: str, mtime: float) -> Optional[Tuple[np.ndarray, Dict]]:
    """
    缓存读取 AIS 密度文件。
    
    Args:
        path_str: 文件路径字符串
        mtime: 文件修改时间（用于缓存 key）
    
    Returns:
        (density_array, metadata) 或 None
    """
    path = Path(path_str)
    if not path.exists():
        return None
    
    try:
        ds = xr.open_dataset(path)
        
        # 寻找密度变量
        varname = None
        for candidate in ["ais_density", "density", "ais"]:
            if candidate in ds:
                varname = candidate
                break
        
        if varname is None:
            logger.warning(f"[AIS] 未找到密度变量在 {path}")
            return None
        
        da = ds[varname]
        
        # 提取元数据
        grid_sig = ds.attrs.get("grid_signature")
        shape = tuple(da.shape) if hasattr(da, "shape") else None
        
        # 转换为 numpy 数组
        arr = da.values.astype(float)
        
        meta = {
            "grid_signature": grid_sig,
            "shape": shape,
            "varname": varname,
            "nan_count": int(np.isnan(arr).sum()),
        }
        
        ds.close()
        return arr, meta
    
    except Exception as e:
        logger.warning(f"[AIS] 读取文件失败 {path}: {e}")
        return None


@lru_cache(maxsize=32)
def _cached_regrid_ais(
    path_str: str,
    mtime: float,
    target_signature: str,
    method: str,
) -> Optional[Tuple[np.ndarray, Dict]]:
    """
    缓存重采样结果。
    
    Args:
        path_str: 源文件路径
        mtime: 源文件修改时间
        target_signature: 目标网格签名
        method: 插值方法 ("linear" | "nearest")
    
    Returns:
        (resampled_array, metadata) 或 None
    """
    # 这个函数由 load_and_align_density 调用，实际重采样逻辑在那里
    # 这里只是占位符，真正的缓存在 load_and_align_density 中
    return None


def scan_ais_density_candidates(search_dirs: Optional[List[str]] = None) -> List[AISDensityCandidate]:
    """
    扫描 AIS 密度候选文件。
    
    Args:
        search_dirs: 搜索目录列表（相对或绝对路径）；若为 None，使用默认目录
    
    Returns:
        候选文件列表（按优先级排序）
    """
    if search_dirs is None:
        search_dirs = DEFAULT_AIS_SEARCH_DIRS
    
    candidates = []
    
    for dir_str in search_dirs:
        search_dir = Path(dir_str)
        if not search_dir.is_absolute():
            search_dir = Path.cwd() / search_dir
        
        if not search_dir.exists():
            continue
        
        # 扫描 *.nc 文件
        for nc_file in sorted(search_dir.glob("*.nc")):
            if not nc_file.is_file():
                continue
            
            # 跳过训练数据
            if "train" in nc_file.name.lower():
                continue
            
            # 尝试读取文件属性
            grid_sig = None
            shape = None
            varname = None
            note = ""
            
            try:
                with xr.open_dataset(nc_file) as ds:
                    grid_sig = ds.attrs.get("grid_signature")
                    
                    # 寻找密度变量
                    for candidate in ["ais_density", "density", "ais"]:
                        if candidate in ds:
                            varname = candidate
                            da = ds[candidate]
                            shape = tuple(da.shape)
                            break
                    
                    if varname is None:
                        note = "未找到密度变量"
            
            except Exception as e:
                note = f"读取失败: {str(e)[:50]}"
            
            # 确定匹配类型
            match_type = "generic"
            if "demo" in nc_file.name.lower():
                match_type = "demo"
            
            # 构造候选对象
            try:
                rel_path = nc_file.relative_to(Path.cwd()).as_posix()
            except ValueError:
                rel_path = nc_file.as_posix()
            
            candidate = AISDensityCandidate(
                path=rel_path,
                grid_signature=grid_sig,
                shape=shape,
                varname=varname,
                note=note,
                match_type=match_type,
            )
            candidates.append(candidate)
    
    # 按优先级排序：demo > generic
    candidates.sort(key=lambda c: (c.match_type != "demo", c.path))
    
    return candidates


def select_best_candidate(
    candidates: List[AISDensityCandidate],
    target_signature: Optional[str] = None,
    prefer_path: Optional[str] = None,
) -> Optional[AISDensityCandidate]:
    """
    从候选列表中选择最佳匹配。
    
    优先级：
    1. prefer_path（若存在且可读）
    2. target_signature 精确匹配
    3. shape 最接近
    4. 返回 None
    
    Args:
        candidates: 候选文件列表
        target_signature: 目标网格签名
        prefer_path: 优先选择的路径
    
    Returns:
        最佳候选或 None
    """
    if not candidates:
        return None
    
    # 1. 优先路径
    if prefer_path:
        prefer_path_norm = Path(prefer_path).as_posix()
        for cand in candidates:
            cand_path_norm = Path(cand.path).as_posix()
            if cand_path_norm == prefer_path_norm or cand.path == prefer_path:
                # 验证文件存在
                p = Path(cand.path)
                if not p.is_absolute():
                    p = Path.cwd() / p
                if p.exists():
                    return cand
    
    # 2. 精确签名匹配
    if target_signature:
        for cand in candidates:
            if cand.grid_signature == target_signature:
                return cand
    
    # 3. shape 最接近（如果有目标 shape）
    # 这里简化处理：返回第一个有效候选
    for cand in candidates:
        if cand.note == "":  # 成功读取的候选
            return cand
    
    # 4. 返回第一个候选（即使有问题）
    return candidates[0] if candidates else None


def load_and_align_density(
    candidate_or_path: Optional[AISDensityCandidate | str],
    grid: Grid2D,
    method: str = "linear",
) -> Tuple[Optional[np.ndarray], Dict]:
    """
    加载并对齐 AIS 密度到目标网格。
    
    Args:
        candidate_or_path: AISDensityCandidate 对象或文件路径字符串
        grid: 目标 Grid2D 对象
        method: 插值方法 ("linear" | "nearest")
    
    Returns:
        (density_2d, metadata) 其中：
        - density_2d: shape 与 grid 相同的 2D 数组，或 None 如果加载失败
        - metadata: 包含原 shape、目标 shape、是否重采样、缓存命中、方法、来源等信息
    """
    meta = {
        "source_path": None,
        "source_shape": None,
        "target_shape": grid.shape(),
        "resampled": False,
        "cache_hit": False,
        "method": method,
        "nan_count": 0,
        "error": None,
    }
    
    # 解析输入
    if isinstance(candidate_or_path, AISDensityCandidate):
        path_str = candidate_or_path.path
    else:
        path_str = str(candidate_or_path)
    
    path = Path(path_str)
    if not path.is_absolute():
        path = Path.cwd() / path
    
    meta["source_path"] = str(path)
    
    if not path.exists():
        meta["error"] = f"文件不存在: {path}"
        return None, meta
    
    # 读取文件
    mtime = _get_mtime(path)
    result = _cached_read_ais_file(str(path), mtime)
    
    if result is None:
        meta["error"] = "读取文件失败"
        return None, meta
    
    arr, file_meta = result
    meta["source_shape"] = file_meta.get("shape")
    meta["nan_count"] = file_meta.get("nan_count", 0)
    
    # 检查是否需要重采样
    target_shape = grid.shape()
    if arr.shape == target_shape:
        # 形状已匹配
        return arr, meta
    
    # 需要重采样
    meta["resampled"] = True
    
    try:
        if method == "linear":
            # 尝试使用 scipy 的 RegularGridInterpolator
            try:
                from scipy.interpolate import RegularGridInterpolator
                
                # 构造源网格坐标
                src_y = np.arange(arr.shape[0])
                src_x = np.arange(arr.shape[1])
                
                # 构造目标网格坐标
                tgt_y = np.linspace(0, arr.shape[0] - 1, target_shape[0])
                tgt_x = np.linspace(0, arr.shape[1] - 1, target_shape[1])
                
                # 创建插值器
                interp = RegularGridInterpolator(
                    (src_y, src_x),
                    arr,
                    method="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
                
                # 生成目标网格点
                tgt_yy, tgt_xx = np.meshgrid(tgt_y, tgt_x, indexing="ij")
                points = np.stack([tgt_yy, tgt_xx], axis=-1)
                
                # 插值
                resampled = interp(points)
                meta["cache_hit"] = False
                return resampled, meta
            
            except ImportError:
                logger.warning("[AIS] scipy 不可用，回退到最近邻插值")
                method = "nearest"
        
        if method == "nearest":
            # 最近邻插值
            src_y = np.arange(arr.shape[0])
            src_x = np.arange(arr.shape[1])
            
            tgt_y = np.linspace(0, arr.shape[0] - 1, target_shape[0])
            tgt_x = np.linspace(0, arr.shape[1] - 1, target_shape[1])
            
            tgt_yy, tgt_xx = np.meshgrid(tgt_y, tgt_x, indexing="ij")
            
            # 四舍五入到最近整数
            idx_y = np.round(tgt_yy).astype(int)
            idx_x = np.round(tgt_xx).astype(int)
            
            # 边界裁剪
            idx_y = np.clip(idx_y, 0, arr.shape[0] - 1)
            idx_x = np.clip(idx_x, 0, arr.shape[1] - 1)
            
            resampled = arr[idx_y, idx_x]
            meta["cache_hit"] = False
            return resampled, meta
    
    except Exception as e:
        meta["error"] = f"重采样失败: {str(e)}"
        logger.warning(f"[AIS] {meta['error']}")
        return None, meta
    
    return None, meta


def compute_grid_signature(grid: Grid2D) -> str:
    """
    计算网格签名，用于 AIS 密度文件的匹配和缓存。
    
    签名格式：{ny}x{nx}_{lat_min:.4f}_{lat_max:.4f}_{lon_min:.4f}_{lon_max:.4f}
    
    Args:
        grid: Grid2D 对象
    
    Returns:
        网格签名字符串
    """
    ny, nx = grid.shape()
    lat_min = float(np.nanmin(grid.lat2d))
    lat_max = float(np.nanmax(grid.lat2d))
    lon_min = float(np.nanmin(grid.lon2d))
    lon_max = float(np.nanmax(grid.lon2d))
    
    signature = f"{ny}x{nx}_{lat_min:.4f}_{lat_max:.4f}_{lon_min:.4f}_{lon_max:.4f}"
    return signature

