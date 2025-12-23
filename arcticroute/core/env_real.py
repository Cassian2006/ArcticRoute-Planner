"""
真实环境数据加载模块。

提供从 NetCDF 文件中加载真实海冰浓度（SIC）等环境数据的功能。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
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
    ice_drift: Optional[np.ndarray] = None  # 漂移速度幅值（m/s）
    bathymetry_depth_m: Optional[np.ndarray] = None  # 水深（正值，米）
    edl_output: Optional[dict] = None
    meta: dict = field(default_factory=dict)



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
    nc_drift_path: Optional[Path] = None,
    nc_bathymetry_path: Optional[Path] = None,
    sic_var_candidates: Tuple[str, ...] = ("sic", "SIC", "ice_concentration"),
    wave_var_candidates: Tuple[str, ...] = ("wave_swh", "swh", "SWH"),
    ice_thickness_var_candidates: Tuple[str, ...] = ("sithick", "sit", "ice_thickness", "ice_thk"),
    drift_u_var_candidates: Tuple[str, ...] = ("uice", "u_drift", "drift_u", "ice_drift_u", "u"),
    drift_v_var_candidates: Tuple[str, ...] = ("vice", "v_drift", "drift_v", "ice_drift_v", "v"),
    drift_speed_var_candidates: Tuple[str, ...] = ("ice_drift", "drift", "drift_speed"),
    bathymetry_var_candidates: Tuple[str, ...] = ("elevation", "z", "depth", "bathymetry", "Band1"),
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

    env_meta: dict = {
        "paths": {},
        "status": {},
        "reasons": {},
    }

    if not sic_files and not grid_files and nc_sic_path is None and nc_wave_path is None:
        print(f"[ENV] no real grid/SIC files found for ym={ym}, returning None")
        env_meta["reasons"]["sic"] = "未找到 sic 或 grid 文件"
        return None

    try:
        grid_obj = grid
        grid_reason = None
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
                or _first_existing([Path(nc_ice_thickness_path)] if nc_ice_thickness_path else [])
                or _first_existing([Path(nc_drift_path)] if nc_drift_path else [])
                or _first_existing([Path(nc_bathymetry_path)] if nc_bathymetry_path else [])
                or _first_existing(list(grid_files))
            )

            if grid_source is not None:
                env_meta["paths"]["grid"] = str(grid_source)
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
                        grid_reason = "grid 文件缺少 latitude/longitude"
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
            else:
                grid_reason = "未找到可用于推断经纬度的文件"
        if grid_reason:
            env_meta["reasons"]["grid"] = grid_reason

        sic = None
        sic_reason = None
        if sic_files:
            sic_path = Path(sic_files[0])
            env_meta["paths"]["sic"] = str(sic_path)
            try:
                with xr.open_dataset(sic_path, decode_times=False) as ds_sic:
                    sic_da = None
                    for name in sic_var_candidates:
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
                    else:
                        sic_reason = "变量未找到"
            except Exception as e:
                sic_reason = f"加载 sic 失败: {e}"
        else:
            sic_reason = "未提供 sic 文件"
        if sic is None and sic_reason:
            env_meta["reasons"]["sic"] = sic_reason

        wave = None
        wave_reason = None
        if wave_files:
            wave_path = Path(wave_files[0])
            env_meta["paths"]["wave_swh"] = str(wave_path)
            try:
                if wave_path.exists():
                    with xr.open_dataset(wave_path, decode_times=False) as ds_wave:
                        wave_da = None
                        for name in wave_var_candidates:
                            if name in ds_wave:
                                wave_da = ds_wave[name]
                                break
                        if wave_da is not None:
                            wave_raw = wave_da.values
                            if wave_raw.ndim == 3:
                                wave_raw = wave_raw[min(time_index, wave_raw.shape[0] - 1), :, :]
                            wave = np.asarray(wave_raw, dtype=float)
                        else:
                            wave_reason = "变量未找到"
            except Exception as e:
                wave_reason = f"加载 wave 失败: {e}"
        else:
            wave_reason = "未提供 wave 文件"
        if wave is None and wave_reason:
            env_meta["reasons"]["wave_swh"] = wave_reason

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
        ice_reason = None
        if nc_ice_thickness_path is not None and grid_obj is not None:
            env_meta["paths"]["sit"] = str(nc_ice_thickness_path)
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
                            ice_reason = "shape 不匹配"
                            ice_thickness = None
                    else:
                        ice_reason = "变量未找到"
            except Exception as e:
                print(f"[ENV] warning: failed to load ice_thickness: {e}")
                ice_reason = f"加载失败: {e}"
                ice_thickness = None
        elif nc_ice_thickness_path is None:
            ice_reason = "未提供 sit 文件"
        else:
            ice_reason = "缺少网格，无法加载冰厚"

        # 若冰厚几乎全 NaN，视作缺失
        if ice_thickness is not None:
            try:
                finite_frac = float(np.isfinite(ice_thickness).mean())
                if finite_frac < 0.001:
                    print(
                        f"[ENV] ice_thickness is mostly NaN (finite_frac={finite_frac:.6f}), treating as missing"
                    )
                    ice_reason = "值几乎全为 NaN"
                    ice_thickness = None
            except Exception:
                ice_thickness = None

        if ice_reason:
            env_meta["reasons"]["sit"] = ice_reason

        # 漂移（可选）
        drift = None
        drift_reason = None
        if nc_drift_path is not None and grid_obj is not None:
            drift_path = Path(nc_drift_path)
            env_meta["paths"]["drift"] = str(drift_path)
            try:
                with xr.open_dataset(drift_path, decode_times=False) as ds_drift:
                    drift_u_da = None
                    drift_v_da = None
                    for name in drift_u_var_candidates:
                        if name in ds_drift:
                            drift_u_da = ds_drift[name]
                            break
                    for name in drift_v_var_candidates:
                        if name in ds_drift:
                            drift_v_da = ds_drift[name]
                            break

                    if drift_u_da is not None and drift_v_da is not None:
                        u = drift_u_da.values
                        v = drift_v_da.values
                        if u.ndim == 3:
                            u = u[min(time_index, u.shape[0] - 1), :, :]
                        if v.ndim == 3:
                            v = v[min(time_index, v.shape[0] - 1), :, :]
                        if u.shape == v.shape == grid_obj.shape():
                            drift = np.sqrt(np.square(u) + np.square(v)).astype(float)
                        else:
                            drift_reason = f"drift shape mismatch {u.shape} / {v.shape} vs {grid_obj.shape()}"
                    else:
                        drift_speed_da = None
                        for name in drift_speed_var_candidates:
                            if name in ds_drift:
                                drift_speed_da = ds_drift[name]
                                break
                        if drift_speed_da is not None:
                            drift_raw = drift_speed_da.values
                            if drift_raw.ndim == 3:
                                drift_raw = drift_raw[min(time_index, drift_raw.shape[0] - 1), :, :]
                            if drift_raw.shape == grid_obj.shape():
                                drift = np.asarray(drift_raw, dtype=float)
                            else:
                                drift_reason = f"drift speed shape {drift_raw.shape} != grid {grid_obj.shape()}"
                        else:
                            drift_reason = "未找到 drift 变量 (u/v 或 drift_speed)"
            except Exception as e:
                drift_reason = f"加载 drift 失败: {e}"
        elif nc_drift_path is None:
            drift_reason = "未提供 drift 文件"
        elif grid_obj is None:
            drift_reason = "缺少网格，无法加载 drift"
        if drift_reason:
            env_meta["reasons"]["drift"] = drift_reason

        # 水深（可选，优先 NC）
        bathy = None
        bathy_reason = None
        if nc_bathymetry_path is not None and grid_obj is not None:
            bathy_path = Path(nc_bathymetry_path)
            env_meta["paths"]["bathymetry"] = str(bathy_path)
            try:
                with xr.open_dataset(bathy_path, decode_times=False) as ds_bathy:
                    bathy_da = None
                    for name in bathymetry_var_candidates:
                        if name in ds_bathy:
                            bathy_da = ds_bathy[name]
                            break
                    if bathy_da is not None:
                        bathy_raw = bathy_da.values
                        if bathy_raw.ndim == 3:
                            bathy_raw = bathy_raw[min(time_index, bathy_raw.shape[0] - 1), :, :]
                        bathy = np.asarray(bathy_raw, dtype=float)
                        if bathy.shape != grid_obj.shape():
                            bathy_reason = f"bathymetry shape {bathy.shape} != grid {grid_obj.shape()}"
                            bathy = None
                        else:
                            # elevation: 海洋为负值，转换为正的水深（米），陆地设为 0
                            bathy = np.where(np.isfinite(bathy), np.where(bathy < 0, -bathy, 0.0), np.nan)
                    else:
                        bathy_reason = "未找到 bathymetry 变量"
            except Exception as e:
                bathy_reason = f"加载 bathymetry 失败: {e}"
        elif nc_bathymetry_path is None:
            bathy_reason = "未提供 bathymetry"
        elif grid_obj is None:
            bathy_reason = "缺少网格，无法加载 bathymetry"
        if bathy_reason:
            env_meta["reasons"]["bathymetry"] = bathy_reason

        # 如果全部缺失则返回 None
        if (
            sic is None
            and wave is None
            and ice_thickness is None
            and drift is None
            and bathy is None
        ):
            env_meta["reasons"]["env"] = "sic/wave/sit/drift/bathymetry 均缺失"
            print(f"[ENV] no sic/wave/sit/drift/bathymetry available for ym={ym}; returning None")
            return None

        env_meta["status"] = {
            "sic_loaded": sic is not None,
            "wave_loaded": wave is not None,
            "sit_loaded": ice_thickness is not None,
            "drift_loaded": drift is not None,
            "bathymetry_loaded": bathy is not None,
            "land_mask_loaded": land_mask is not None,
        }

        env = RealEnvLayers(
            grid=grid_obj,
            sic=sic,
            wave_swh=wave,
            land_mask=land_mask,
            ice_thickness_m=ice_thickness,
            ice_drift=drift,
            bathymetry_depth_m=bathy,
            edl_output=None,
            meta=env_meta,
        )

        if grid_obj is not None:
            ny, nx = grid_obj.shape()
            sic_status = "OK" if sic is not None else "None"
            wave_status = "OK" if wave is not None else "None"
            land_status = "OK" if land_mask is not None else "None"
            ice_status = "OK" if ice_thickness is not None else "None"
            drift_status = "OK" if drift is not None else "None"
            bathy_status = "OK" if bathy is not None else "None"
            print(
                f"[ENV] loaded real env for ym={ym}: grid={ny}x{nx}, "
                f"sic={sic_status}, wave={wave_status}, land={land_status}, "
                f"ice_thickness={ice_status}, drift={drift_status}, bathy={bathy_status}"
            )
        return env

    except Exception as e:
        print(f"[ENV] failed to load real env for ym={ym}: {e}; returning None")
        return None

