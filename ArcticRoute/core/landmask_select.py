"""
Landmask 统一选择/匹配/加载/对齐模块。

提供稳定的 API 供 UI 和网格加载使用，包括：
  - scan_landmask_candidates: 扫描候选文件
  - select_best_candidate: 选择最佳匹配
  - load_and_align_landmask: 加载并对齐到目标网格

缓存策略：
  - 文件读取缓存：key = (path, mtime)
  - 重采样缓存：key = (path, mtime, target_signature, method)
"""

import logging
import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from arcticroute.core.grid import Grid2D

logger = logging.getLogger(__name__)

# 默认 landmask 搜索目录
DEFAULT_LANDMASK_SEARCH_DIRS = [
    "data_real/landmask",
    "data_real/env",
    "data_real",
]


@dataclass
class LandmaskCandidate:
    """Landmask 候选文件信息。"""

    path: str
    grid_signature: Optional[str] = None
    shape: Optional[Tuple[int, int]] = None
    varname: Optional[str] = None
    note: str = ""

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
def _cached_read_landmask_file(
    path_str: str, mtime: float
) -> Optional[Tuple[np.ndarray, Dict]]:
    """
    缓存读取 landmask 文件。

    Args:
        path_str: 文件路径字符串
        mtime: 文件修改时间（用于缓存 key）

    Returns:
        (landmask_array, metadata) 或 None
    """
    path = Path(path_str)
    if not path.exists():
        return None

    try:
        ds = xr.open_dataset(path)

        # 寻找 landmask 变量
        varname = None
        for candidate in [
            "land_mask",
            "landmask",
            "mask",
            "lsm",
            "land",
            "is_land",
        ]:
            if candidate in ds:
                varname = candidate
                break

        if varname is None:
            # 尝试找第一个 2D 布尔或整数变量
            for var_name in ds.data_vars:
                da = ds[var_name]
                if da.ndim == 2 and da.dtype in [bool, np.int32, np.int64, np.float32, np.float64]:
                    varname = var_name
                    break

        if varname is None:
            logger.warning(f"[LANDMASK] 未找到 landmask 变量在 {path}")
            return None

        da = ds[varname]

        # 提取元数据
        grid_sig = ds.attrs.get("grid_signature")
        shape = tuple(da.shape) if hasattr(da, "shape") else None

        # 转换为 numpy 数组
        arr = da.values

        meta = {
            "grid_signature": grid_sig,
            "shape": shape,
            "varname": varname,
            "nan_count": int(np.isnan(arr).sum()) if arr.dtype == float else 0,
        }

        ds.close()
        return arr, meta

    except Exception as e:
        logger.warning(f"[LANDMASK] 读取文件失败 {path}: {e}")
        return None


def scan_landmask_candidates(
    search_dirs: Optional[List[str]] = None,
) -> List[LandmaskCandidate]:
    """
    扫描 landmask 候选文件。

    Args:
        search_dirs: 搜索目录列表（相对或绝对路径）；若为 None，使用默认目录

    Returns:
        候选文件列表（按优先级排序）
    """
    if search_dirs is None:
        search_dirs = DEFAULT_LANDMASK_SEARCH_DIRS

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

            # 尝试读取文件属性
            grid_sig = None
            shape = None
            varname = None
            note = ""

            try:
                with xr.open_dataset(nc_file) as ds:
                    grid_sig = ds.attrs.get("grid_signature")

                    # 寻找 landmask 变量
                    for candidate in [
                        "land_mask",
                        "landmask",
                        "mask",
                        "lsm",
                        "land",
                        "is_land",
                    ]:
                        if candidate in ds:
                            varname = candidate
                            da = ds[candidate]
                            shape = tuple(da.shape)
                            break

                    if varname is None:
                        note = "未找到 landmask 变量"

            except Exception as e:
                note = f"读取失败: {str(e)[:50]}"

            # 构造候选对象
            try:
                rel_path = nc_file.relative_to(Path.cwd()).as_posix()
            except ValueError:
                rel_path = nc_file.as_posix()

            candidate = LandmaskCandidate(
                path=rel_path,
                grid_signature=grid_sig,
                shape=shape,
                varname=varname,
                note=note,
            )
            candidates.append(candidate)

    return candidates


def select_best_candidate(
    candidates: List[LandmaskCandidate],
    target_signature: Optional[str] = None,
    prefer_path: Optional[str] = None,
) -> Optional[LandmaskCandidate]:
    """
    从候选列表中选择最佳匹配。

    优先级：
    1. prefer_path（若存在且可读）
    2. target_signature 精确匹配
    3. 文件名含 landmask/land_mask
    4. shape 最接近
    5. 返回第一个有效候选

    Args:
        candidates: 候选文件列表
        target_signature: 目标网格签名
        prefer_path: 优先选择的路径

    Returns:
        最佳候选或 None
    """
    if not candidates:
        return None

    # 1. 优先路径（需要文件存在且成功读取）
    if prefer_path:
        prefer_path_norm = Path(prefer_path).as_posix()
        for cand in candidates:
            cand_path_norm = Path(cand.path).as_posix()
            # 比较路径（支持相对和绝对路径）
            if cand_path_norm == prefer_path_norm or cand.path == prefer_path:
                # 验证文件存在且成功读取
                p = Path(cand.path)
                if not p.is_absolute():
                    p = Path.cwd() / p
                if p.exists() and cand.note == "":  # 成功读取
                    return cand

    # 2. 精确签名匹配
    if target_signature:
        for cand in candidates:
            if cand.grid_signature == target_signature and cand.note == "":
                return cand

    # 3. 文件名含 landmask/land_mask
    for cand in candidates:
        if cand.note == "":  # 成功读取的候选
            if "landmask" in cand.path.lower() or "land_mask" in cand.path.lower():
                return cand

    # 4. 返回第一个成功读取的候选
    for cand in candidates:
        if cand.note == "":
            return cand

    # 5. 返回第一个候选（即使有问题）
    return candidates[0] if candidates else None


def load_and_align_landmask(
    candidate_or_path: Optional[LandmaskCandidate | str],
    grid: Grid2D,
    method: str = "nearest",
) -> Tuple[Optional[np.ndarray], Dict]:
    """
    加载并对齐 landmask 到目标网格。

    语义归一化：
    - 支持 0/1、1/0、bool、float（>0.5 判 land）、NaN 当 ocean
    - 输出：True = land, False = ocean

    Args:
        candidate_or_path: LandmaskCandidate 对象或文件路径字符串
        grid: 目标 Grid2D 对象
        method: 插值方法 ("nearest" | "linear")

    Returns:
        (landmask_2d, metadata) 其中：
        - landmask_2d: shape 与 grid 相同的 bool 数组，或 None 如果加载失败
        - metadata: 包含原 shape、目标 shape、是否重采样、缓存命中、方法、来源等信息
    """
    meta = {
        "source_path": None,
        "original_shape": None,
        "target_shape": grid.shape(),
        "resampled": False,
        "cache_hit": False,
        "method": method,
        "varname": None,
        "land_fraction": None,
        "nan_count": 0,
        "error": None,
        "warning": None,
    }

    # 解析输入
    if isinstance(candidate_or_path, LandmaskCandidate):
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
    result = _cached_read_landmask_file(str(path), mtime)

    if result is None:
        meta["error"] = "读取文件失败"
        return None, meta

    arr, file_meta = result
    meta["original_shape"] = file_meta.get("shape")
    meta["varname"] = file_meta.get("varname")
    meta["nan_count"] = file_meta.get("nan_count", 0)

    # 语义归一化：转换为 bool (True = land)
    try:
        arr = _normalize_landmask_semantics(arr)
    except Exception as e:
        meta["error"] = f"语义归一化失败: {str(e)}"
        logger.warning(f"[LANDMASK] {meta['error']}")
        return None, meta

    # 检查是否需要重采样
    target_shape = grid.shape()
    if arr.shape == target_shape:
        # 形状已匹配
        land_frac = float(arr.sum()) / arr.size if arr.size > 0 else 0.0
        meta["land_fraction"] = land_frac

        # 检查陆地比例是否合理
        if land_frac < 0.001 or land_frac > 0.98:
            meta["warning"] = (
                f"陆地比例异常: {land_frac:.6f} (期望在 0.001-0.98 之间)"
            )

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
                    arr.astype(float),
                    method="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )

                # 生成目标网格点
                tgt_yy, tgt_xx = np.meshgrid(tgt_y, tgt_x, indexing="ij")
                points = np.stack([tgt_yy, tgt_xx], axis=-1)

                # 插值
                resampled = interp(points)
                # 转换回 bool：>0.5 判为 land
                resampled = resampled > 0.5
                meta["cache_hit"] = False

                land_frac = float(resampled.sum()) / resampled.size if resampled.size > 0 else 0.0
                meta["land_fraction"] = land_frac

                if land_frac < 0.001 or land_frac > 0.98:
                    meta["warning"] = (
                        f"重采样后陆地比例异常: {land_frac:.6f} (期望在 0.001-0.98 之间)"
                    )

                return resampled, meta

            except ImportError:
                logger.warning("[LANDMASK] scipy 不可用，回退到最近邻插值")
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

            land_frac = float(resampled.sum()) / resampled.size if resampled.size > 0 else 0.0
            meta["land_fraction"] = land_frac

            if land_frac < 0.001 or land_frac > 0.98:
                meta["warning"] = (
                    f"重采样后陆地比例异常: {land_frac:.6f} (期望在 0.001-0.98 之间)"
                )

            return resampled, meta

    except Exception as e:
        meta["error"] = f"重采样失败: {str(e)}"
        logger.warning(f"[LANDMASK] {meta['error']}")
        return None, meta

    return None, meta


def _normalize_landmask_semantics(arr: np.ndarray) -> np.ndarray:
    """
    将 landmask 数组归一化为 bool (True = land, False = ocean)。

    支持多种语义：
    - 0/1 编码（0=ocean, 1=land 或反过来）
    - bool 编码
    - float 编码（>0.5 判为 land）
    - NaN 当 ocean

    Args:
        arr: 原始 landmask 数组

    Returns:
        bool 数组 (True = land, False = ocean)
    """
    # 如果已经是 bool，直接返回
    if arr.dtype == bool:
        return arr

    # 转换为 float 便于处理
    arr_float = arr.astype(float)

    # 记录 NaN 位置（NaN 应该当 ocean）
    nan_mask = np.isnan(arr_float)

    # 处理 NaN（先替换为 0，后续会设置为 False）
    arr_float = np.nan_to_num(arr_float, nan=0.0)

    # 检查值的范围和分布
    unique_vals = np.unique(arr_float)

    # 如果只有 0 和 1，判断哪个是 land
    if len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0.0, 1.0])):
        # 简单启发式：计算 1 的比例
        frac_ones = np.mean(arr_float == 1.0)

        # 如果 1 的比例在 10%-40% 之间，认为 1 是 land（合理的陆地比例）
        # 否则认为 0 是 land
        if 0.05 < frac_ones < 0.50:
            result = arr_float > 0.5
        else:
            # 反转：0 是 land
            result = arr_float < 0.5
    else:
        # 对于其他情况，>0.5 判为 land
        result = arr_float > 0.5

    # 将 NaN 位置设置为 False（ocean）
    result[nan_mask] = False

    return result


def compute_grid_signature(grid: Grid2D) -> str:
    """
    计算网格签名，用于 landmask 文件的匹配和缓存。

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

