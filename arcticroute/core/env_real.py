"""
真实环境数据加载模块。

提供从 NetCDF 文件中加载真实海冰浓度（SIC）等环境数据的功能。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .grid import Grid2D

# ============================================================================
# 路径常量
# ============================================================================

# ???????

def get_data_root() -> Path:
    """Resolve the data root (env override -> backup -> project fallback)."""
    env = os.getenv("ARCTICROUTE_DATA_ROOT")
    if env:
        p = Path(env)
        if p.exists():
            print(f"[DATA] using ARCTICROUTE_DATA_ROOT={p}")
            return p

    here = Path(__file__).resolve().parents[2]  # project root
    candidate = here.parent / "ArcticRoute_data_backup"
    if candidate.exists():
        print(f"[DATA] using backup data root at {candidate}")
        return candidate

    fallback = here / "data_real"
    print(f"[DATA] using fallback data dir {fallback} (may be empty)")
    return fallback
def _get_candidate_dirs() -> list[Path]:
    """
    获取候选的环境数据目录列表。
    
    按优先级排序：
    1. data_processed/newenv
    2. data_processed/env
    3. data_processed
    
    Returns:
        候选目录列表
    """
    base = get_data_root()
    candidates = [
        base / "data_processed" / "newenv",
        base / "data_processed" / "env",
        base / "data_processed",
    ]
    return candidates


def _find_file_in_candidates(filename: str, candidates: list[Path] | None = None) -> Path | None:
    """
    在候选目录中查找文件。
    
    Args:
        filename: 文件名（不包含路径）
        candidates: 候选目录列表，若为 None 则使用 _get_candidate_dirs()
    
    Returns:
        找到的文件路径，或 None 如果未找到
    """
    if candidates is None:
        candidates = _get_candidate_dirs()
    
    for candidate_dir in candidates:
        filepath = candidate_dir / filename
        if filepath.exists():
            return filepath
    
    return None


@dataclass
class RealEnvLayers:
    """????????????????????????"""

    grid: Grid2D | None = None
    sic: Optional[np.ndarray] = None
    wave_swh: Optional[np.ndarray] = None
    land_mask: Optional[np.ndarray] = None
    ice_thickness_m: Optional[np.ndarray] = None
    edl_output: Optional[dict] = None



@dataclass
class EnvFileSet:
    grid_files: list[Path]
    sic_files: list[Path]
    wave_files: list[Path]
    landmask_files: list[Path]


def resolve_env_files_for_ym(ym: str) -> EnvFileSet:
    """
    Resolve candidate real-data files for a given year-month.

    Even if directories are missing, returns empty lists instead of raising.
    """
    root = get_data_root()
    dp = root / "data_processed"

    env_dir = dp / "env"
    newenv_dir = dp / "newenv"

    candidates_grid: list[Path] = []
    if (env_dir / "env_clean.nc").exists():
        candidates_grid.append(env_dir / "env_clean.nc")

    candidates_sic: list[Path] = []
    if (newenv_dir / "ice_copernicus_sic.nc").exists():
        candidates_sic.append(newenv_dir / "ice_copernicus_sic.nc")

    candidates_wave: list[Path] = []
    if (newenv_dir / "wave_swh.nc").exists():
        candidates_wave.append(newenv_dir / "wave_swh.nc")

    candidates_land: list[Path] = []
    if (env_dir / "land_mask.nc").exists():
        candidates_land.append(env_dir / "land_mask.nc")

    return EnvFileSet(
        grid_files=candidates_grid,
        sic_files=candidates_sic,
        wave_files=candidates_wave,
        landmask_files=candidates_land,
    )

def load_real_sic_for_grid(
    grid: Grid2D,
    nc_path: Optional[Path] = None,
    var_candidates: Tuple[str, ...] = ("sic", "SIC", "ice_concentration"),
    time_index: int = 0,
) -> Optional[RealEnvLayers]:
    """
    尝试从 NetCDF 文件中读取与 grid 对齐的海冰浓度（sic）。

    要求：
    - 返回 sic shape == (grid.ny, grid.nx) 的数组。
    - 若文件缺失 / 变量不存在 / 形状不匹配且无法简单调整，则返回 None（不要 raise）。
    - 失败时打印一条简短 debug 信息即可。

    说明：
    - 如果数据有 time 维度，例如 (time, y, x)，取给定 time_index 的切片。
    - 如果数据是 (y, x)，则直接使用。

    Args:
        grid: Grid2D 对象，用于确定目标形状
        nc_path: NetCDF 文件路径。若为 None，则默认尝试
                 get_newenv_path() / "ice_copernicus_sic.nc"
        var_candidates: 变量名候选列表，按顺序尝试
        time_index: 若数据有 time 维度，取该索引的切片

    Returns:
        RealEnvLayers 对象（sic 有效），或 None 如果加载失败
    """
    try:
        import xarray as xr
    except ImportError:
        print("[ENV] xarray not available, cannot load real SIC")
        return None

    # 确定文件路径
    if nc_path is None:
        # 尝试多个候选文件名和目录
        candidate_dirs = _get_candidate_dirs()
        candidate_filenames = ["ice_copernicus_sic.nc", "sic.nc"]
        
        nc_path = None
        for filename in candidate_filenames:
            nc_path = _find_file_in_candidates(filename, candidate_dirs)
            if nc_path is not None:
                break
        
        if nc_path is None:
            print(f"[ENV] real SIC not available: file not found in any of {candidate_dirs}, "
                  f"candidates: {candidate_filenames}")
            return None

    nc_path = Path(nc_path)

    if not nc_path.exists():
        print(f"[ENV] real SIC not available: file not found at {nc_path}")
        return None

    try:
        ds = xr.open_dataset(nc_path, decode_times=False)
    except Exception as e:
        print(f"[ENV] real SIC not available: failed to open {nc_path}: {e}")
        return None

    # 尝试找到 sic 变量
    sic_da = None
    for var_name in var_candidates:
        if var_name in ds:
            sic_da = ds[var_name]
            break

    if sic_da is None:
        print(
            f"[ENV] real SIC not available: "
            f"no variable found in {nc_path}, candidates={var_candidates}"
        )
        return None

    try:
        sic = sic_da.values
        ny, nx = grid.shape()

        # 处理维度
        if sic.ndim == 3:
            # 假设为 (time, y, x)
            if time_index >= sic.shape[0]:
                print(
                    f"[ENV] real SIC not available: "
                    f"time_index {time_index} out of range [0, {sic.shape[0]})"
                )
                return None
            sic = sic[time_index, :, :]
        elif sic.ndim != 2:
            print(
                f"[ENV] real SIC not available: "
                f"unexpected sic dimensions {sic.ndim}, expected 2 or 3"
            )
            return None

        # 检查形状
        if sic.shape != (ny, nx):
            print(
                f"[ENV] real SIC not available: "
                f"sic shape {sic.shape} != grid shape ({ny}, {nx})"
            )
            return None

        # 数据类型转换和裁剪
        sic = np.asarray(sic, dtype=float)

        # 假设原始数据可能是 0..100 或 0..1，尝试自动检测
        # 如果最大值 > 1.5，则假设是 0..100，除以 100
        if np.nanmax(sic) > 1.5:
            sic = sic / 100.0

        # 裁剪到 0..1
        sic = np.clip(sic, 0.0, 1.0)

        print(
            f"[ENV] successfully loaded real SIC from {nc_path}, "
            f"shape={sic.shape}, range=[{np.nanmin(sic):.3f}, {np.nanmax(sic):.3f}]"
        )

        return RealEnvLayers(sic=sic)

    except Exception as e:
        print(f"[ENV] real SIC not available: error processing data: {e}")
        return None
    finally:
        try:
            ds.close()
        except Exception:
            pass


def load_real_grid_from_data_real(ym: str) -> Optional[Grid2D]:
    """
    从 data_real/{ym}/ 目录中的数据文件提取网格坐标。
    
    优先尝试从 sic_{ym}.nc 或 wave_{ym}.nc 中读取 lat/lon。
    
    Args:
        ym: 年月字符串（格式 "YYYYMM"）
    
    Returns:
        Grid2D 对象，或 None 如果加载失败
    """
    try:
        import xarray as xr
    except ImportError:
        print("[ENV] xarray not available, cannot load real grid from data_real")
        return None
    
    # 尝试从 SIC 文件读取
    sic_path = Path(__file__).resolve().parents[2] / "data_real" / ym / f"sic_{ym}.nc"
    if sic_path.exists():
        try:
            ds = xr.open_dataset(sic_path, decode_times=False)
            lat = ds.coords.get("latitude")
            lon = ds.coords.get("longitude")
            
            if lat is not None and lon is not None:
                lat_vals = lat.values
                lon_vals = lon.values
                
                # 构建 2D 网格
                if lat_vals.ndim == 1 and lon_vals.ndim == 1:
                    lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
                elif lat_vals.ndim == 2 and lon_vals.ndim == 2:
                    lat2d, lon2d = np.broadcast_arrays(lat_vals, lon_vals)
                else:
                    print(f"[ENV] unexpected lat/lon dimensions in {sic_path}")
                    return None
                
                grid = Grid2D(lat2d=lat2d, lon2d=lon2d)
                print(f"[ENV] successfully loaded real grid from {sic_path}, shape={grid.shape()}")
                return grid
        except Exception as e:
            print(f"[ENV] failed to load grid from {sic_path}: {e}")
    
    # 尝试从 wave 文件读取
    wave_path = Path(__file__).resolve().parents[2] / "data_real" / ym / f"wave_{ym}.nc"
    if wave_path.exists():
        try:
            ds = xr.open_dataset(wave_path, decode_times=False)
            lat = ds.coords.get("latitude")
            lon = ds.coords.get("longitude")
            
            if lat is not None and lon is not None:
                lat_vals = lat.values
                lon_vals = lon.values
                
                # 构建 2D 网格
                if lat_vals.ndim == 1 and lon_vals.ndim == 1:
                    lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
                elif lat_vals.ndim == 2 and lon_vals.ndim == 2:
                    lat2d, lon2d = np.broadcast_arrays(lat_vals, lon_vals)
                else:
                    print(f"[ENV] unexpected lat/lon dimensions in {wave_path}")
                    return None
                
                grid = Grid2D(lat2d=lat2d, lon2d=lon2d)
                print(f"[ENV] successfully loaded real grid from {wave_path}, shape={grid.shape()}")
                return grid
        except Exception as e:
            print(f"[ENV] failed to load grid from {wave_path}: {e}")
    
    print(f"[ENV] could not load real grid from data_real/{ym}/")
    return None


def load_real_env_for_grid(
    grid: Grid2D | None = None,
    nc_sic_path: Optional[Path] = None,
    nc_wave_path: Optional[Path] = None,
    nc_ice_thickness_path: Optional[Path] = None,
    sic_var_candidates: Tuple[str, ...] = ("sic", "SIC", "ice_concentration"),
    wave_var_candidates: Tuple[str, ...] = ("wave_swh", "swh", "SWH"),
    ice_thickness_var_candidates: Tuple[str, ...] = ("sithick", "sit", "ice_thickness", "ice_thk"),
    time_index: int = 0,
    ym: Optional[str] = None,
) -> Optional[RealEnvLayers]:
    """
    ?????????????????? sic???wave_swh ????????? land_mask????????? R2 ????????????????????????????????????
    ???????????????data_processed/env ??? data_processed/newenv???
    ????????????????????????????????? None???
    """
    try:
        import xarray as xr
    except ImportError:
        print("[ENV] xarray not available, cannot load real environment data")
        return None

    resolved = resolve_env_files_for_ym(ym or "")
    sic_files = [Path(nc_sic_path)] if nc_sic_path else resolved.sic_files
    wave_files = [Path(nc_wave_path)] if nc_wave_path else resolved.wave_files
    landmask_files = resolved.landmask_files
    grid_files = resolved.grid_files

    if not sic_files and not grid_files:
        print(f"[ENV] no real grid/SIC files found for ym={ym}, returning None")
        return None

    try:
        grid_obj = grid
        if grid_obj is None or (sic_files or grid_files or wave_files):
            # 选择用于推断网格的来源文件：优先 sic，其次 wave，最后独立 grid 文件
            def _first_existing(paths: list[Path]) -> Path | None:
                for p in paths:
                    try:
                        if Path(p).exists():
                            return Path(p)
                    except Exception:
                        continue
                return None

            grid_source = (
                _first_existing(list(sic_files))
                or _first_existing(list(wave_files))
                or _first_existing(list(grid_files))
            )

            if grid_source is not None:
                with xr.open_dataset(grid_source, decode_times=False) as ds_grid:
                    lat_da = None
                    for name in ["latitude", "lat", "LAT", "y"]:
                        if name in ds_grid:
                            lat_da = ds_grid[name]
                            break
                    lon_da = None
                    for name in ["longitude", "lon", "LON", "x"]:
                        if name in ds_grid:
                            lon_da = ds_grid[name]
                            break
                    if lat_da is None or lon_da is None:
                        # 无有效经纬度，保留现有 grid_obj 或稍后回退
                        lat2d = lon2d = None
                    else:
                        lat_vals = lat_da.values
                        lon_vals = lon_da.values
                        if lat_vals.ndim == 1 and lon_vals.ndim == 1:
                            lat2d, lon2d = np.meshgrid(lat_vals, lon_vals, indexing="ij")
                        else:
                            lat2d, lon2d = lat_vals, lon_vals
                        if grid_obj is None or grid_obj.shape() != lat2d.shape:
                            grid_obj = Grid2D(lat2d=lat2d, lon2d=lon2d)
            # 若仍没有 grid_obj，则保留传入的 grid（可能为 None），由后续逻辑决定回退

        sic = None
        if sic_files:
            sic_path = sic_files[0]
            try:
                with xr.open_dataset(sic_path, decode_times=False) as ds_sic:
                    sic_da = None
                    for name in ("sic", "SIC", "ice_concentration"):
                        if name in ds_sic:
                            sic_da = ds_sic[name]
                            break
                    if sic_da is not None:
                        sic_raw = sic_da.values
                        if sic_raw.ndim == 3:
                            sic_raw = sic_raw[min(time_index, sic_raw.shape[0] - 1), :, :]
                        sic = np.asarray(sic_raw, dtype=float)
                        if np.nanmax(sic) > 1.5:
                            sic = sic / 100.0
            except Exception:
                # sic 不可用时保持 None
                sic = None

        wave = None
        if wave_files:
            wave_path = Path(wave_files[0])
            try:
                if wave_path.exists():
                    with xr.open_dataset(wave_path, decode_times=False) as ds_wave:
                        wave_da = None
                        for name in ("wave_swh", "swh", "SWH"):
                            if name in ds_wave:
                                wave_da = ds_wave[name]
                                break
                        if wave_da is not None:
                            wave_raw = wave_da.values
                            if wave_raw.ndim == 3:
                                wave_raw = wave_raw[min(time_index, wave_raw.shape[0] - 1), :, :]
                            wave = np.asarray(wave_raw, dtype=float)
            except Exception:
                wave = None

        land_mask = None
        if landmask_files and grid_obj is not None:
            try:
                from .landmask import load_real_landmask_from_nc

                land_mask = load_real_landmask_from_nc(
                    grid_obj, nc_path=landmask_files[0], var_name="land_mask"
                )
            except Exception:
                land_mask = None

        # 尝试加载冰厚（可选）
        ice_thickness = None
        if nc_ice_thickness_path is not None and grid_obj is not None:
            try:
                with xr.open_dataset(nc_ice_thickness_path, decode_times=False) as ds_ice:
                    ice_da = None
                    for name in ice_thickness_var_candidates:
                        if name in ds_ice:
                            ice_da = ds_ice[name]
                            break
                    if ice_da is not None:
                        ice_raw = ice_da.values
                        if ice_raw.ndim == 3:
                            ice_raw = ice_raw[min(time_index, ice_raw.shape[0] - 1), :, :]
                        ice_thickness = np.asarray(ice_raw, dtype=float)
                        # 对齐形状
                        if ice_thickness.shape != grid_obj.shape():
                            print(
                                f"[ENV] warning: ice_thickness shape {ice_thickness.shape} != grid shape {grid_obj.shape()}, skipping ice_thickness"
                            )
                            ice_thickness = None
            except Exception as e:
                print(f"[ENV] warning: failed to load ice_thickness: {e}")
                ice_thickness = None

        # 若冰厚几乎全 NaN，视作缺失
        if ice_thickness is not None:
            try:
                finite_frac = float(np.isfinite(ice_thickness).mean())
                if finite_frac < 0.001:
                    print(
                        f"[ENV] ice_thickness is mostly NaN (finite_frac={finite_frac:.6f}), treating as missing"
                    )
                    ice_thickness = None
            except Exception:
                ice_thickness = None

        # 如果既没有 sic 也没有 wave 数据，则返回 None（符合测试预期）
        if sic is None and wave is None:
            print(f"[ENV] no sic/wave available for ym={ym}; returning None")
            return None

        env = RealEnvLayers(
            grid=grid_obj,
            sic=sic,
            wave_swh=wave,
            land_mask=land_mask,
            ice_thickness_m=ice_thickness,
            edl_output=None,
        )

        if grid_obj is not None:
            ny, nx = grid_obj.shape()
            sic_status = "OK" if sic is not None else "None"
            wave_status = "OK" if wave is not None else "None"
            land_status = "OK" if land_mask is not None else "None"
            ice_status = "OK" if ice_thickness is not None else "None"
            print(
                f"[ENV] loaded real env for ym={ym}: grid={ny}x{nx}, "
                f"sic={sic_status}, wave={wave_status}, land={land_status}, ice_thickness={ice_status}"
            )
        return env

    except Exception as e:
        print(f"[ENV] failed to load real env for ym={ym}: {e}; returning None")
        return None

