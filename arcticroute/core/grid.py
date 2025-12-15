"""
网格与坐标系工具模块。

提供网格加载、坐标管理等功能。
支持从真实数据（land_mask_gebco.nc）或合成 demo grid 加载。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Grid2D:
    """2D 网格数据类，存储纬度和经度坐标。"""

    lat2d: np.ndarray  # 2D, shape (ny, nx)
    lon2d: np.ndarray  # 2D, shape (ny, nx)

    def shape(self) -> tuple[int, int]:
        """返回网格形状 (ny, nx)。"""
        return self.lat2d.shape


def get_project_root() -> Path:
    """获取项目根目录。假设本文件在 arcticroute/core/grid.py。"""
    return Path(__file__).resolve().parents[2]


def get_data_root() -> Path:
    """
    获取数据根目录。

    优先从环境变量 ARCTICROUTE_DATA_ROOT 读取；
    否则默认使用 PROJECT_ROOT/../ArcticRoute_data_backup。
    """
    env = os.getenv("ARCTICROUTE_DATA_ROOT")
    if env:
        return Path(env)
    # 默认认为备份在工程旁边
    return get_project_root().parent / "ArcticRoute_data_backup"


def load_real_grid_from_landmask() -> tuple[Grid2D, np.ndarray] | None:
    """
    尝试从备份目录中加载 landmask 文件，返回 (Grid2D, land_mask)。

    按优先级尝试：
    1. data_processed/env/land_mask.nc
    2. data_processed/newenv/land_mask_gebco.nc
    3. data_processed/env/land_mask_gebco.nc

    land_mask 为 bool 数组，True = 陆地。
    如果文件不存在或结构不符合预期，返回 None。
    """
    try:
        import xarray as xr
    except ImportError:
        print("[GRID] xarray not available, skipping real grid load")
        return None

    data_root = get_data_root()
    
    # 按优先级尝试多个候选路径
    candidate_paths = [
        data_root / "data_processed" / "env" / "land_mask.nc",
        data_root / "data_processed" / "newenv" / "land_mask_gebco.nc",
        data_root / "data_processed" / "env" / "land_mask_gebco.nc",
    ]
    
    nc_path = None
    for cand in candidate_paths:
        if cand.exists():
            nc_path = cand
            break
    
    if nc_path is None:
        print(f"[GRID] no landmask file found in candidates: {[str(c) for c in candidate_paths]}")
        return None

    try:
        ds = xr.open_dataset(nc_path)
    except Exception as e:
        print(f"[GRID] failed to open {nc_path}: {e}")
        return None

    # 尝试识别陆地掩码变量名
    land_var_name_candidates = ["land_mask", "mask", "land"]
    land_da = None
    for name in land_var_name_candidates:
        if name in ds:
            land_da = ds[name]
            break

    if land_da is None:
        print(
            f"[GRID] no land mask variable found in {nc_path}, "
            f"candidates={land_var_name_candidates}"
        )
        return None

    # lat / lon 变量名候选
    lat_name = next(
        (n for n in ["lat", "latitude", "y"] if n in ds.coords or n in ds), None
    )
    lon_name = next(
        (n for n in ["lon", "longitude", "x"] if n in ds.coords or n in ds), None
    )

    if lat_name is None or lon_name is None:
        print(f"[GRID] could not find lat/lon coordinates in {nc_path}")
        return None

    lat_da = ds.coords.get(lat_name)
    if lat_da is None:
        lat_da = ds[lat_name]
    lon_da = ds.coords.get(lon_name)
    if lon_da is None:
        lon_da = ds[lon_name]
    lat = lat_da.values
    lon = lon_da.values

    # 支持 1D 或 2D
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    elif lat.ndim == 2 and lon.ndim == 2:
        lat2d, lon2d = np.broadcast_arrays(lat, lon)
    else:
        print(
            f"[GRID] lat/lon shape not supported: "
            f"lat.ndim={lat.ndim}, lon.ndim={lon.ndim}"
        )
        return None

    land_mask = land_da.values.astype(bool)

    # 确保 land_mask 形状与 lat2d 对齐
    if land_mask.shape != lat2d.shape:
        try:
            land_mask = np.broadcast_to(land_mask, lat2d.shape)
        except ValueError:
            print(
                f"[GRID] land_mask shape {land_mask.shape} not compatible "
                f"with grid {lat2d.shape}"
            )
            return None

    grid = Grid2D(lat2d=lat2d, lon2d=lon2d)
    return grid, land_mask


def make_demo_grid(ny: int = 40, nx: int = 80) -> tuple[Grid2D, np.ndarray]:
    """
    生成一个简单的 2D 网格和 land_mask，用于本地无数据时的 demo / 测试。

    约定：
      - 纬度 65~80N， 经度 0~160E；
      - 最右侧 10 列为陆地。

    Args:
        ny: 纬度方向网格数
        nx: 经度方向网格数

    Returns:
        (Grid2D, land_mask) 元组
    """
    lat_1d = np.linspace(65.0, 80.0, ny)
    lon_1d = np.linspace(0.0, 160.0, nx)
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)

    land_mask = np.zeros((ny, nx), dtype=bool)
    land_mask[:, -10:] = True  # 简单的"右侧陆地"区域

    return Grid2D(lat2d=lat2d, lon2d=lon2d), land_mask


def load_real_grid_from_nc(
    nc_path: Optional[Path] = None,
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> Optional[Grid2D]:
    """
    ??? NetCDF ?????????????? Grid2D?

    - ? nc_path ? None??? data_processed/newenv ? data_processed/env ?????????
    - ?????????????? None??????????? demo??
    """
    try:
        import xarray as xr
    except ImportError:
        print("[GRID] xarray not available, cannot load real grid")
        return None

    if nc_path is None:
        from .config_paths import get_newenv_path

        candidates = [
            get_newenv_path() / "env_clean.nc",
            get_newenv_path() / "grid_spec.nc",
            get_newenv_path() / "land_mask_gebco.nc",
            get_data_root() / "data_processed" / "env" / "env_clean.nc",
            get_data_root() / "data_processed" / "env" / "land_mask.nc",
        ]
        nc_path = next((c for c in candidates if c.exists()), None)
        if nc_path is None:
            print(
                f"[GRID] real grid file not found in candidates: {[str(c) for c in candidates]}"
            )
            return None

    nc_path = Path(nc_path)
    if not nc_path.exists():
        print(f"[GRID] real grid file not found at {nc_path}")
        return None

    try:
        ds = xr.open_dataset(nc_path, decode_times=False)
    except Exception as e:
        print(f"[GRID] failed to open {nc_path}: {e}")
        return None

    try:
        lat_candidates = [lat_name, "latitude", "Latitude", "LAT", "y"]
        lon_candidates = [lon_name, "longitude", "Longitude", "LON", "x"]

        lat_var = None
        for name in lat_candidates:
            lat_var = ds.coords.get(name)
            if lat_var is None:
                lat_var = ds.get(name)
            if lat_var is not None:
                lat_name = name
                break

        lon_var = None
        for name in lon_candidates:
            lon_var = ds.coords.get(name)
            if lon_var is None:
                lon_var = ds.get(name)
            if lon_var is not None:
                lon_name = name
                break

        if lat_var is None or lon_var is None:
            print(
                f"[GRID] could not find lat/lon variables in {nc_path}: "
                f"lat tried={lat_candidates}, lon tried={lon_candidates}"
            )
            return None

        lat = lat_var.values
        lon = lon_var.values

        if lat.ndim == 1 and lon.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon, lat)
        elif lat.ndim == 2 and lon.ndim == 2:
            lat2d, lon2d = np.broadcast_arrays(lat, lon)
        else:
            print(
                f"[GRID] lat/lon shape not supported: lat.ndim={lat.ndim}, lon.ndim={lon.ndim}"
            )
            return None

        grid = Grid2D(lat2d=lat2d, lon2d=lon2d)
        print(f"[GRID] successfully loaded real grid from {nc_path}, shape={grid.shape()}")
        return grid

    except Exception as e:
        print(f"[GRID] error processing grid data from {nc_path}: {e}")
        return None
    finally:
        try:
            ds.close()
        except Exception:
            pass

def load_grid_with_landmask(
    prefer_real: bool = True,
    explicit_landmask_path: Optional[str] = None,
    landmask_search_dirs: Optional[list[str]] = None,
) -> tuple[Grid2D, np.ndarray, dict]:
    """
    加载网格与 land_mask。

    - 若 prefer_real=True 且成功从 land_mask_gebco.nc 读取，则返回真实网格；
    - 否则退回合成 demo grid。

    新增参数支持更灵活的 landmask 加载策略。

    Args:
        prefer_real: 是否优先加载真实数据
        explicit_landmask_path: 显式指定的 landmask 文件路径
        landmask_search_dirs: landmask 搜索目录列表

    Returns:
        (grid, land_mask, meta) 元组，其中 meta 包含：
          - "source": "real" / "demo"
          - "data_root": str(data_root)
          - "landmask_path": 加载的 landmask 文件路径
          - "landmask_resampled": 是否进行了重采样
          - "landmask_land_fraction": 陆地比例
          - "landmask_note": 诊断信息或回退原因
    """
    # 首先加载网格
    if prefer_real:
        real = load_real_grid_from_landmask()
        if real is not None:
            grid, land_mask_old = real
            grid_source = "real"
        else:
            grid, land_mask_old = make_demo_grid()
            grid_source = "demo"
    else:
        grid, land_mask_old = make_demo_grid()
        grid_source = "demo"

    # 然后加载 landmask（使用新的统一接口）
    try:
        from .landmask_select import load_and_align_landmask, scan_landmask_candidates, select_best_candidate, compute_grid_signature
        
        candidates = scan_landmask_candidates(search_dirs=landmask_search_dirs)
        target_sig = compute_grid_signature(grid)
        best_candidate = select_best_candidate(
            candidates, target_signature=target_sig, prefer_path=explicit_landmask_path
        )
        
        if best_candidate is not None:
            land_mask, landmask_meta = load_and_align_landmask(best_candidate, grid, method="nearest")
            if land_mask is not None:
                # 成功加载真实 landmask
                meta = {
                    "source": grid_source,
                    "data_root": str(get_data_root()),
                    "landmask_path": landmask_meta.get("source_path"),
                    "landmask_resampled": landmask_meta.get("resampled", False),
                    "landmask_land_fraction": landmask_meta.get("land_fraction"),
                    "landmask_note": "successfully loaded real landmask",
                }
                return grid, land_mask, meta
        
        # 未找到或加载失败，使用旧的 land_mask
        meta = {
            "source": grid_source,
            "data_root": str(get_data_root()),
            "landmask_path": "fallback (old method)",
            "landmask_resampled": False,
            "landmask_land_fraction": float(land_mask_old.sum()) / land_mask_old.size if land_mask_old.size > 0 else 0.0,
            "landmask_note": "using fallback landmask from old method",
        }
        return grid, land_mask_old, meta
    
    except Exception as e:
        # 异常时回退到旧方法
        meta = {
            "source": grid_source,
            "data_root": str(get_data_root()),
            "landmask_path": "fallback (error)",
            "landmask_resampled": False,
            "landmask_land_fraction": float(land_mask_old.sum()) / land_mask_old.size if land_mask_old.size > 0 else 0.0,
            "landmask_note": f"error in new landmask loading: {str(e)[:100]}",
        }
        return grid, land_mask_old, meta
