"""
陆地掩码加载与质量检查模块。

提供陆地掩码加载、统计等功能。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .grid import Grid2D, load_grid_with_landmask
from .landmask_select import (
    LandmaskCandidate,
    scan_landmask_candidates,
    select_best_candidate,
    load_and_align_landmask,
    compute_grid_signature,
)


@dataclass
class RouteLandmaskStats:
    """路线与陆地掩码的统计信息数据类。"""

    total_steps: int
    on_land_steps: int
    on_ocean_steps: int
    first_land_index: int | None
    first_land_latlon: Tuple[float, float] | None


@dataclass
class LandMaskInfo:
    """陆地掩码信息数据类。"""

    grid: Grid2D
    land_mask: np.ndarray  # bool, shape = grid.shape()
    frac_land: float  # 陆地比例
    frac_ocean: float  # 海洋比例
    source: str  # "real" 或 "demo"


def load_landmask_for_grid(
    grid: Grid2D,
    prefer_real: bool = True,
    explicit_path: Optional[str] = None,
    search_dirs: Optional[List[str]] = None,
) -> Tuple[np.ndarray, dict]:
    """
    为指定网格加载并对齐 landmask。

    这是新的统一入口，对标 AIS density 的设计。支持：
    - 候选扫描 → 显式选择/自动匹配 → 必要时最近邻重采样 → 缓存
    - 清晰的诊断信息和修复指引

    Args:
        grid: 目标 Grid2D 对象
        prefer_real: 是否优先加载真实数据（若为 False 则直接返回 demo）
        explicit_path: 显式指定的 landmask 文件路径
        search_dirs: 搜索目录列表（若为 None 使用默认目录）

    Returns:
        (landmask_bool_2d, meta) 元组，其中：
        - landmask_bool_2d: shape 与 grid 相同的 bool 数组 (True = land)
        - meta: 包含以下字段的字典：
            - source_path: 加载的文件路径（或 "demo" 如果回退）
            - original_shape: 原始文件中的 shape
            - target_shape: 目标网格 shape
            - resampled: 是否进行了重采样
            - varname: 使用的变量名
            - land_fraction: 陆地比例
            - fallback_demo: 是否回退到 demo
            - reason: 回退原因（如果有）
            - warning: 警告信息（如陆地比例异常）
    """
    meta = {
        "source_path": None,
        "original_shape": None,
        "target_shape": grid.shape(),
        "resampled": False,
        "varname": None,
        "land_fraction": None,
        "fallback_demo": False,
        "reason": None,
        "warning": None,
    }

    if not prefer_real:
        # 直接返回 demo
        demo_grid, demo_mask = _make_demo_landmask(grid)
        land_frac = float(demo_mask.sum()) / demo_mask.size if demo_mask.size > 0 else 0.0
        meta["source_path"] = "demo"
        meta["land_fraction"] = land_frac
        meta["fallback_demo"] = True
        meta["reason"] = "prefer_real=False"
        return demo_mask, meta

    # 尝试加载真实 landmask
    candidates = scan_landmask_candidates(search_dirs=search_dirs)

    # 选择最佳候选
    target_sig = compute_grid_signature(grid)
    best_candidate = select_best_candidate(
        candidates, target_signature=target_sig, prefer_path=explicit_path
    )

    if best_candidate is None:
        # 没有找到任何候选
        demo_grid, demo_mask = _make_demo_landmask(grid)
        land_frac = float(demo_mask.sum()) / demo_mask.size if demo_mask.size > 0 else 0.0
        meta["source_path"] = "demo"
        meta["land_fraction"] = land_frac
        meta["fallback_demo"] = True
        meta["reason"] = "未找到任何 landmask 候选文件"
        return demo_mask, meta

    # 尝试加载和对齐
    landmask, load_meta = load_and_align_landmask(best_candidate, grid, method="nearest")

    if landmask is None:
        # 加载失败，回退到 demo
        demo_grid, demo_mask = _make_demo_landmask(grid)
        land_frac = float(demo_mask.sum()) / demo_mask.size if demo_mask.size > 0 else 0.0
        meta["source_path"] = "demo"
        meta["land_fraction"] = land_frac
        meta["fallback_demo"] = True
        meta["reason"] = load_meta.get("error", "加载失败")
        return demo_mask, meta

    # 成功加载
    meta.update(load_meta)
    return landmask, meta


def _make_demo_landmask(grid: Grid2D) -> Tuple[Grid2D, np.ndarray]:
    """
    为指定网格生成 demo landmask。

    约定：最右侧 10% 列为陆地。

    Args:
        grid: 目标网格

    Returns:
        (grid, landmask_bool) 元组
    """
    ny, nx = grid.shape()
    land_mask = np.zeros((ny, nx), dtype=bool)
    # 最右侧 10% 列为陆地
    n_land_cols = max(1, nx // 10)
    land_mask[:, -n_land_cols:] = True
    return grid, land_mask


def load_landmask(prefer_real: bool = True) -> LandMaskInfo:
    """
    加载陆地掩码信息。

    内部调用 load_grid_with_landmask，保持向后兼容。

    Args:
        prefer_real: 是否优先加载真实数据

    Returns:
        LandMaskInfo 对象

    Raises:
        ValueError: 如果 land_mask 形状与网格不匹配
    """
    grid, land_mask, meta = load_grid_with_landmask(prefer_real=prefer_real)

    if land_mask.shape != grid.shape():
        raise ValueError(
            f"land_mask shape {land_mask.shape} != grid shape {grid.shape()}"
        )

    total = land_mask.size
    n_land = int(land_mask.sum())
    frac_land = n_land / float(total) if total > 0 else 0.0
    frac_ocean = 1.0 - frac_land

    return LandMaskInfo(
        grid=grid,
        land_mask=land_mask,
        frac_land=frac_land,
        frac_ocean=frac_ocean,
        source=str(meta.get("source", "unknown")),
    )


def _scan_landmask_candidates() -> list[tuple[Path, str]]:
    """
    扫描所有候选的 landmask 文件。
    
    按优先级返回 (path, source_description) 列表：
    1. <DATA_ROOT>/data_processed/env/land_mask.nc
    2. <DATA_ROOT>/data_processed/newenv/land_mask_gebco.nc
    3. <DATA_ROOT>/data_processed/env/land_mask_gebco.nc
    
    Returns:
        [(Path, description), ...] 列表，仅包含存在的文件
    """
    from .env_real import get_data_root
    
    root = get_data_root()
    candidates = [
        (root / "data_processed" / "env" / "land_mask.nc", "env/land_mask.nc"),
        (root / "data_processed" / "newenv" / "land_mask_gebco.nc", "newenv/land_mask_gebco.nc"),
        (root / "data_processed" / "env" / "land_mask_gebco.nc", "env/land_mask_gebco.nc"),
    ]
    
    result = []
    for path, desc in candidates:
        if path.exists():
            result.append((path, desc))
    
    return result


def _try_load_landmask_from_file(
    nc_path: Path,
    grid: Grid2D,
    var_candidates: list[str] = None,
    coord_candidates: list[tuple[str, str]] = None,
) -> Optional[tuple[np.ndarray, dict]]:
    """
    尝试从单个 NetCDF 文件加载 landmask。
    
    Args:
        nc_path: 文件路径
        grid: 目标网格
        var_candidates: 变量名候选列表
        coord_candidates: 坐标名候选列表 [(lat_name, lon_name), ...]
    
    Returns:
        (landmask_array, metadata_dict) 或 None
        metadata_dict 包含: source_path, var_name, shape, coord_names
    """
    try:
        import xarray as xr
    except ImportError:
        return None
    
    if var_candidates is None:
        var_candidates = ["land_mask", "mask", "LANDMASK", "is_land"]
    
    if coord_candidates is None:
        coord_candidates = [("latitude", "longitude"), ("lat", "lon")]
    
    try:
        ds = xr.open_dataset(nc_path, decode_times=False)
    except Exception as e:
        print(f"[LANDMASK] failed to open {nc_path}: {e}")
        return None
    
    try:
        # 找变量
        land_da = None
        var_name_found = None
        for var_name in var_candidates:
            if var_name in ds:
                land_da = ds[var_name]
                var_name_found = var_name
                break
        
        if land_da is None:
            print(
                f"[LANDMASK] variable not found in {nc_path}. "
                f"Candidates: {var_candidates}. "
                f"Available: {list(ds.data_vars.keys())}"
            )
            return None
        
        land_mask = land_da.values.astype(bool)
        ny, nx = grid.shape()
        target_shape = (ny, nx)
        
        # 如果形状已匹配，直接返回
        if land_mask.shape == target_shape:
            print(
                f"[LANDMASK] loaded from {nc_path}: "
                f"var={var_name_found}, shape={land_mask.shape}"
            )
            return land_mask, {
                "source_path": str(nc_path),
                "var_name": var_name_found,
                "shape": land_mask.shape,
                "coord_names": None,
                "resampled": False,
            }
        
        # 尝试坐标重采样
        print(
            f"[LANDMASK] shape mismatch: {land_mask.shape} != {target_shape}, "
            f"attempting coordinate-based resampling..."
        )
        
        resampled = _resample_landmask_by_coords(
            land_mask, nc_path, grid, coord_candidates
        )
        
        if resampled is not None:
            print(
                f"[LANDMASK] resampled to {resampled.shape} using coordinate-based method"
            )
            return resampled, {
                "source_path": str(nc_path),
                "var_name": var_name_found,
                "shape": resampled.shape,
                "coord_names": "lat/lon",
                "resampled": True,
            }
        
        # 回退到简单最近邻
        print(
            f"[LANDMASK] coordinate-based resampling failed, "
            f"falling back to simple nearest-neighbor..."
        )
        resampled = _resample_landmask_simple(land_mask, target_shape)
        
        if resampled is not None:
            print(f"[LANDMASK] resampled to {resampled.shape} using simple method")
            return resampled, {
                "source_path": str(nc_path),
                "var_name": var_name_found,
                "shape": resampled.shape,
                "coord_names": None,
                "resampled": True,
            }
        
        return None
    
    except Exception as e:
        print(f"[LANDMASK] error processing {nc_path}: {e}")
        return None
    finally:
        try:
            ds.close()
        except Exception:
            pass


def _resample_landmask_by_coords(
    land_mask: np.ndarray,
    nc_path: Path,
    grid: Grid2D,
    coord_candidates: list[tuple[str, str]],
) -> Optional[np.ndarray]:
    """
    使用坐标信息进行最近邻重采样。
    
    Args:
        land_mask: 原始 landmask 数组
        nc_path: 数据文件路径（用于读取坐标）
        grid: 目标网格
        coord_candidates: [(lat_name, lon_name), ...] 坐标名候选
    
    Returns:
        重采样后的 landmask，或 None 如果失败
    """
    try:
        import xarray as xr
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    
    try:
        ds = xr.open_dataset(nc_path, decode_times=False)
        
        # 找坐标
        lat_da = None
        lon_da = None
        lat_name = None
        lon_name = None
        
        for lat_cand, lon_cand in coord_candidates:
            if lat_cand in ds.coords and lon_cand in ds.coords:
                lat_da = ds.coords[lat_cand]
                lon_da = ds.coords[lon_cand]
                lat_name = lat_cand
                lon_name = lon_cand
                break
        
        if lat_da is None or lon_da is None:
            print(f"[LANDMASK] coordinates not found in {nc_path}")
            return None
        
        old_lat_vals = lat_da.values
        old_lon_vals = lon_da.values
        
        # 构建原始网格
        old_lon2d, old_lat2d = np.meshgrid(old_lon_vals, old_lat_vals)
        
        # 构建 KDTree
        old_points = np.column_stack([old_lat2d.ravel(), old_lon2d.ravel()])
        tree = cKDTree(old_points)
        
        # 查询新网格中每个点的最近邻
        new_points = np.column_stack([grid.lat2d.ravel(), grid.lon2d.ravel()])
        _, indices = tree.query(new_points)
        
        # 重采样
        ny, nx = grid.shape()
        resampled = land_mask.ravel()[indices].reshape((ny, nx))
        
        return resampled
    
    except Exception as e:
        print(f"[LANDMASK] coordinate-based resampling failed: {e}")
        return None
    finally:
        try:
            ds.close()
        except Exception:
            pass


def _resample_landmask_simple(
    land_mask: np.ndarray,
    target_shape: tuple[int, int],
) -> Optional[np.ndarray]:
    """
    使用简单的线性索引进行最近邻重采样。
    
    Args:
        land_mask: 原始 landmask 数组
        target_shape: 目标形状 (ny, nx)
    
    Returns:
        重采样后的 landmask
    """
    try:
        old_ny, old_nx = land_mask.shape
        ny, nx = target_shape
        
        y_indices = np.round(np.linspace(0, old_ny - 1, ny)).astype(int)
        x_indices = np.round(np.linspace(0, old_nx - 1, nx)).astype(int)
        y_indices = np.clip(y_indices, 0, old_ny - 1)
        x_indices = np.clip(x_indices, 0, old_nx - 1)
        
        resampled = land_mask[np.ix_(y_indices, x_indices)]
        return resampled
    except Exception as e:
        print(f"[LANDMASK] simple resampling failed: {e}")
        return None


def load_real_landmask_from_nc(
    grid: Grid2D,
    nc_path: Optional[Path] = None,
    var_name: str = "land_mask",
    strict: bool = False,
) -> Optional[np.ndarray]:
    """
    尝试从 NetCDF 中载入与 grid 对齐的 landmask。

    这个函数保持向后兼容，内部调用新的 load_landmask_for_grid。

    Args:
        grid: Grid2D 对象，用于确定目标形状
        nc_path: NetCDF 文件路径，若为 None 则自动扫描候选
        var_name: 变量名（已弃用，改用自动候选）
        strict: 若为 True，当加载失败时返回 None；若为 False，回退到 demo

    Returns:
        bool 数组（形状为 grid.shape()），或 None 如果加载失败（仅当 strict=True）
    """
    # 使用新的统一入口
    landmask, meta = load_landmask_for_grid(
        grid, prefer_real=True, explicit_path=str(nc_path) if nc_path else None
    )

    # 如果 strict 模式且回退到 demo，返回 None
    if strict and meta.get("fallback_demo"):
        return None

    # 否则返回 landmask（真实或 demo）
    return landmask


def evaluate_route_against_landmask(
    grid: Grid2D,
    land_mask: np.ndarray,
    route_latlon: List[Tuple[float, float]],
) -> RouteLandmaskStats:
    """
    给定网格、land_mask 和一条 (lat, lon) 路径，统计该路径在 land_mask 上的踩陆情况。

    约定 land_mask == True 表示陆地，不可通行。

    Args:
        grid: Grid2D 对象
        land_mask: bool 数组，True = 陆地，shape = grid.shape()
        route_latlon: [(lat, lon), ...] 路径列表

    Returns:
        RouteLandmaskStats 对象，包含踩陆统计信息
    """
    if not route_latlon:
        # 空路径
        return RouteLandmaskStats(
            total_steps=0,
            on_land_steps=0,
            on_ocean_steps=0,
            first_land_index=None,
            first_land_latlon=None,
        )

    lat2d = grid.lat2d
    lon2d = grid.lon2d
    ny, nx = grid.shape()

    on_land_steps = 0
    on_ocean_steps = 0
    first_land_index = None
    first_land_latlon = None

    for idx, (lat, lon) in enumerate(route_latlon):
        # 使用最近邻的方式映射到 (i, j) 栅格索引
        dist = np.sqrt((lat2d - lat) ** 2 + (lon2d - lon) ** 2)
        i, j = np.unravel_index(np.argmin(dist), dist.shape)

        # 检查是否越界（虽然 unravel_index 不会越界，但保险起见）
        if not (0 <= i < ny and 0 <= j < nx):
            # 越界视为海上
            on_ocean_steps += 1
            continue

        # 检查是否踩陆
        if land_mask[i, j]:
            on_land_steps += 1
            if first_land_index is None:
                first_land_index = idx
                first_land_latlon = (lat, lon)
        else:
            on_ocean_steps += 1

    return RouteLandmaskStats(
        total_steps=len(route_latlon),
        on_land_steps=on_land_steps,
        on_ocean_steps=on_ocean_steps,
        first_land_index=first_land_index,
        first_land_latlon=first_land_latlon,
    )
