"""
CMEMS (Copernicus Marine) 数据加载模块

从 NetCDF 文件中加载海冰浓度（SIC）和有效波高（SWH）数据，
并将其重采样到当前网格。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None

logger = logging.getLogger(__name__)


def find_latest_nc(outdir: Path | str, pattern: str = "*.nc") -> Optional[Path]:
    """
    在指定目录中查找最新的 NetCDF 文件。
    
    Args:
        outdir: 搜索目录
        pattern: 文件名模式（默认 "*.nc"）
    
    Returns:
        最新的 NetCDF 文件路径，或 None 如果未找到
    """
    outdir = Path(outdir)
    if not outdir.exists():
        logger.warning(f"输出目录不存在: {outdir}")
        return None
    
    nc_files = list(outdir.glob(pattern))
    if not nc_files:
        logger.warning(f"在 {outdir} 中未找到匹配 {pattern} 的文件")
        return None
    
    # 按修改时间排序，返回最新的
    latest = max(nc_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"找到最新的 NetCDF 文件: {latest}")
    return latest


def load_sic_from_nc(path: Path | str) -> Tuple[np.ndarray, dict]:
    """
    从 NetCDF 文件中加载海冰浓度（SIC）数据。
    
    Args:
        path: NetCDF 文件路径
    
    Returns:
        (sic_2d, metadata) 元组
        - sic_2d: 2D 海冰浓度数组 (0-100 或 0-1)
        - metadata: 包含坐标、属性等的字典
    
    Raises:
        ImportError: 如果 xarray 未安装
        FileNotFoundError: 如果文件不存在
    """
    if xr is None:
        raise ImportError("需要安装 xarray 来加载 NetCDF 文件: pip install xarray")
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    
    logger.info(f"加载 SIC 数据: {path}")
    
    try:
        ds = xr.open_dataset(path)
        
        # 尝试找到 SIC 变量（可能的名称）
        sic_var_names = ["sic", "sea_ice_concentration", "ice_conc", "concentration"]
        sic_var = None
        sic_var_name = None
        
        for name in sic_var_names:
            if name in ds.data_vars:
                sic_var = ds[name]
                sic_var_name = name
                break
        
        if sic_var is None:
            # 如果没找到，列出可用的变量
            available = list(ds.data_vars.keys())
            raise ValueError(f"未找到 SIC 变量。可用变量: {available}")
        
        # 提取数据
        sic_2d = sic_var.values
        
        # 如果是 3D（时间+空间），取最后一个时间步
        if sic_2d.ndim == 3:
            logger.info(f"SIC 是 3D 数据，形状 {sic_2d.shape}，取最后一个时间步")
            sic_2d = sic_2d[-1, :, :]
        
        # 规范化到 0-1 范围（如果需要）
        if sic_2d.max() > 1.5:
            logger.info("SIC 数据范围 > 1.5，假设为 0-100，转换为 0-1")
            sic_2d = sic_2d / 100.0
        
        # 构建元数据
        metadata = {
            "variable": sic_var_name,
            "shape": sic_2d.shape,
            "min": float(np.nanmin(sic_2d)),
            "max": float(np.nanmax(sic_2d)),
            "dtype": str(sic_2d.dtype),
        }
        
        # 尝试提取坐标信息
        if "lon" in ds.coords:
            metadata["lon"] = ds.coords["lon"].values
        if "lat" in ds.coords:
            metadata["lat"] = ds.coords["lat"].values
        if "longitude" in ds.coords:
            metadata["lon"] = ds.coords["longitude"].values
        if "latitude" in ds.coords:
            metadata["lat"] = ds.coords["latitude"].values
        
        # 尝试提取时间信息
        if "time" in ds.coords:
            metadata["time"] = str(ds.coords["time"].values[-1])
        
        ds.close()
        
        logger.info(f"成功加载 SIC 数据，形状: {sic_2d.shape}")
        return sic_2d, metadata
    
    except Exception as e:
        logger.error(f"加载 SIC 数据失败: {e}")
        raise


def load_swh_from_nc(path: Path | str) -> Tuple[np.ndarray, dict]:
    """
    从 NetCDF 文件中加载有效波高（SWH）数据。
    
    Args:
        path: NetCDF 文件路径
    
    Returns:
        (swh_2d, metadata) 元组
        - swh_2d: 2D 有效波高数组（单位：米）
        - metadata: 包含坐标、属性等的字典
    
    Raises:
        ImportError: 如果 xarray 未安装
        FileNotFoundError: 如果文件不存在
    """
    if xr is None:
        raise ImportError("需要安装 xarray 来加载 NetCDF 文件: pip install xarray")
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    
    logger.info(f"加载 SWH 数据: {path}")
    
    try:
        ds = xr.open_dataset(path)
        
        # 尝试找到 SWH 变量（可能的名称）
        swh_var_names = [
            "sea_surface_wave_significant_height",
            "swh",
            "hs",
            "significant_wave_height",
            "wave_height",
        ]
        swh_var = None
        swh_var_name = None
        
        for name in swh_var_names:
            if name in ds.data_vars:
                swh_var = ds[name]
                swh_var_name = name
                break
        
        if swh_var is None:
            # 如果没找到，列出可用的变量
            available = list(ds.data_vars.keys())
            raise ValueError(f"未找到 SWH 变量。可用变量: {available}")
        
        # 提取数据
        swh_2d = swh_var.values
        
        # 如果是 3D（时间+空间），取最后一个时间步
        if swh_2d.ndim == 3:
            logger.info(f"SWH 是 3D 数据，形状 {swh_2d.shape}，取最后一个时间步")
            swh_2d = swh_2d[-1, :, :]
        
        # 构建元数据
        metadata = {
            "variable": swh_var_name,
            "shape": swh_2d.shape,
            "min": float(np.nanmin(swh_2d)),
            "max": float(np.nanmax(swh_2d)),
            "dtype": str(swh_2d.dtype),
            "unit": "m",  # 假设单位为米
        }
        
        # 尝试提取坐标信息
        if "lon" in ds.coords:
            metadata["lon"] = ds.coords["lon"].values
        if "lat" in ds.coords:
            metadata["lat"] = ds.coords["lat"].values
        if "longitude" in ds.coords:
            metadata["lon"] = ds.coords["longitude"].values
        if "latitude" in ds.coords:
            metadata["lat"] = ds.coords["latitude"].values
        
        # 尝试提取时间信息
        if "time" in ds.coords:
            metadata["time"] = str(ds.coords["time"].values[-1])
        
        ds.close()
        
        logger.info(f"成功加载 SWH 数据，形状: {swh_2d.shape}")
        return swh_2d, metadata
    
    except Exception as e:
        logger.error(f"加载 SWH 数据失败: {e}")
        raise


def align_to_grid(
    data_2d: np.ndarray,
    source_coords: dict,
    target_grid,
    method: str = "nearest",
) -> np.ndarray:
    """
    将数据重采样到目标网格。
    
    Args:
        data_2d: 源数据数组
        source_coords: 源数据的坐标信息（包含 'lon' 和 'lat'）
        target_grid: 目标网格对象（需要有 lon/lat 属性）
        method: 重采样方法（'nearest', 'linear' 等）
    
    Returns:
        重采样后的数据数组
    """
    if xr is None:
        logger.warning("xarray 未安装，跳过重采样，返回原始数据")
        return data_2d
    
    try:
        # 创建源数据的 xarray DataArray
        if "lon" in source_coords and "lat" in source_coords:
            source_lon = source_coords["lon"]
            source_lat = source_coords["lat"]
            
            # 处理 1D 坐标
            if source_lon.ndim == 1 and source_lat.ndim == 1:
                source_da = xr.DataArray(
                    data_2d,
                    coords={"lat": source_lat, "lon": source_lon},
                    dims=["lat", "lon"],
                )
            else:
                # 2D 坐标，直接使用
                source_da = xr.DataArray(
                    data_2d,
                    coords={"lat": (("y", "x"), source_lat), "lon": (("y", "x"), source_lon)},
                    dims=["y", "x"],
                )
        else:
            logger.warning("源坐标信息不完整，跳过重采样")
            return data_2d
        
        # 获取目标网格的坐标
        target_lon = target_grid.lons  # 假设网格有 lons/lats 属性
        target_lat = target_grid.lats
        
        # 使用 xarray 的 interp 方法进行重采样
        aligned = source_da.interp(
            lon=target_lon,
            lat=target_lat,
            method=method,
        )
        
        logger.info(f"数据重采样完成，新形状: {aligned.shape}")
        return aligned.values
    
    except Exception as e:
        logger.warning(f"重采样失败: {e}，返回原始数据")
        return data_2d


def load_sic_with_fallback(
    ym: str,
    prefer_nextsim: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    加载 SIC 数据，优先尝试 nextsim HM，失败则回退到观测数据。
    
    Phase 10.5 策略：
    - nextsim HM 作为"可用则优先"的增强数据源
    - 观测数据作为稳定的主要数据源
    - 自动 fallback，无需用户干预
    
    Args:
        ym: 年月字符串 (YYYYMM)
        prefer_nextsim: 是否优先尝试 nextsim（默认 True）
    
    Returns:
        (sic_2d, metadata)
        metadata 包含 'source' 字段，值为 'nextsim_hm' 或 'cmems_obs-si'
    
    Raises:
        Exception: 如果所有数据源都失败
    """
    if prefer_nextsim:
        try:
            logger.info(f"尝试加载 nextsim HM 数据 (ym={ym})")
            sic_2d, meta = load_sic_from_nextsim_hm(ym)
            meta["source"] = "nextsim_hm"
            meta["fallback_used"] = False
            logger.info(f"成功使用 nextsim HM 数据源 (ym={ym})")
            return sic_2d, meta
        except Exception as e:
            logger.warning(f"nextsim HM 加载失败: {e}，回退到观测数据")
    
    # 回退到观测数据
    logger.info(f"加载观测数据 (ym={ym})")
    sic_2d, meta = load_sic_from_nc_obs(ym)
    meta["source"] = "cmems_obs-si"
    meta["fallback_used"] = True if prefer_nextsim else False
    logger.info(f"使用观测数据源 (ym={ym})")
    return sic_2d, meta


def load_sic_from_nextsim_hm(path_or_ym: str) -> Tuple[np.ndarray, dict]:
    """
    从 nextsim HM 数据集加载 SIC 数据。
    
    Args:
        path_or_ym: NetCDF 文件路径或年月字符串 (YYYYMM)
    
    Returns:
        (sic_2d, metadata)
    
    Raises:
        FileNotFoundError: 如果文件不存在
        ValueError: 如果数据集不包含 SIC 变量
    """
    # 如果输入是年月字符串，需要先下载或查找文件
    # 这里简化为直接加载文件
    if len(path_or_ym) == 6 and path_or_ym.isdigit():
        # 假设文件在特定目录
        path = Path(f"data_real/cmems/nextsim_hm/{path_or_ym}_sic.nc")
        if not path.exists():
            raise FileNotFoundError(
                f"nextsim HM 数据文件不存在: {path}. "
                "请先运行 cmems_download.py 下载数据。"
            )
    else:
        path = Path(path_or_ym)
    
    logger.info(f"加载 nextsim HM SIC 数据: {path}")
    
    if xr is None:
        raise ImportError("需要安装 xarray 来加载 NetCDF 文件: pip install xarray")
    
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    
    try:
        ds = xr.open_dataset(path)
        
        # nextsim HM 的 SIC 变量名可能是 'sic' 或 'sea_ice_concentration'
        sic_var_names = ["sic", "sea_ice_concentration", "ice_conc"]
        sic_var = None
        sic_var_name = None
        
        for name in sic_var_names:
            if name in ds.data_vars:
                sic_var = ds[name]
                sic_var_name = name
                break
        
        if sic_var is None:
            available = list(ds.data_vars.keys())
            raise ValueError(f"未找到 SIC 变量。可用变量: {available}")
        
        # 提取数据
        sic_2d = sic_var.values
        
        # 如果是 3D（时间+空间），取最后一个时间步
        if sic_2d.ndim == 3:
            logger.info(f"SIC 是 3D 数据，形状 {sic_2d.shape}，取最后一个时间步")
            sic_2d = sic_2d[-1, :, :]
        
        # 规范化到 0-1 范围
        if sic_2d.max() > 1.5:
            logger.info("SIC 数据范围 > 1.5，假设为 0-100，转换为 0-1")
            sic_2d = sic_2d / 100.0
        
        # 构建元数据
        metadata = {
            "variable": sic_var_name,
            "shape": sic_2d.shape,
            "min": float(np.nanmin(sic_2d)),
            "max": float(np.nanmax(sic_2d)),
            "dtype": str(sic_2d.dtype),
            "dataset": "cmems_mod_arc_phy_anfc_nextsim_hm",
        }
        
        # 尝试提取坐标信息
        if "lon" in ds.coords:
            metadata["lon"] = ds.coords["lon"].values
        if "lat" in ds.coords:
            metadata["lat"] = ds.coords["lat"].values
        if "longitude" in ds.coords:
            metadata["lon"] = ds.coords["longitude"].values
        if "latitude" in ds.coords:
            metadata["lat"] = ds.coords["latitude"].values
        
        # 尝试提取时间信息
        if "time" in ds.coords:
            metadata["time"] = str(ds.coords["time"].values[-1])
        
        ds.close()
        
        logger.info(f"成功加载 nextsim HM SIC 数据，形状: {sic_2d.shape}")
        return sic_2d, metadata
    
    except Exception as e:
        logger.error(f"加载 nextsim HM SIC 数据失败: {e}")
        raise


def load_sic_from_nc_obs(path_or_ym: str) -> Tuple[np.ndarray, dict]:
    """
    从观测数据集加载 SIC 数据（cmems_obs-si）。
    
    这是 load_sic_from_nc 的别名，用于明确表示使用观测数据。
    
    Args:
        path_or_ym: NetCDF 文件路径或年月字符串 (YYYYMM)
    
    Returns:
        (sic_2d, metadata)
    """
    if len(path_or_ym) == 6 and path_or_ym.isdigit():
        # 假设文件在特定目录
        path = Path(f"data_real/cmems/obs-si/{path_or_ym}_sic.nc")
        if not path.exists():
            raise FileNotFoundError(
                f"观测数据文件不存在: {path}. "
                "请先运行 cmems_download.py 下载数据。"
            )
    else:
        path = path_or_ym
    
    return load_sic_from_nc(path)

