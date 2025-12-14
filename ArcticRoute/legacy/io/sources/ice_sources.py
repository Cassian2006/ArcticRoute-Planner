from __future__ import annotations

import os
import glob
from typing import Tuple, Dict, Optional, List

import numpy as np
import xarray as xr
import streamlit as st

# 统一分块策略：time=12，其它维=256

def _compute_chunks(ds: xr.Dataset) -> dict:
    chunks: Dict[str, int] = {}
    for dim, size in ds.sizes.items():
        if dim.lower() == "time":
            chunks[dim] = int(min(12, max(1, size)))
        else:
            chunks[dim] = int(min(256, max(1, size)))
    return chunks


def _get_lon_name(ds: xr.Dataset) -> Optional[str]:
    for name in ["lon", "longitude", "x"]:
        if name in ds.coords or name in ds:
            return name
    for name in ds.data_vars:
        if name.lower() in ("lon", "longitude"):
            return name
    return None


def _get_lat_name(ds: xr.Dataset) -> Optional[str]:
    for name in ["lat", "latitude", "y"]:
        if name in ds.coords or name in ds:
            return name
    for name in ds.data_vars:
        if name.lower() in ("lat", "latitude"):
            return name
    return None


def _normalize_lon(ds: xr.Dataset) -> xr.Dataset:
    lon_name = _get_lon_name(ds)
    if lon_name is None:
        return ds
    lon = ds[lon_name]
    if lon.ndim == 1:
        try:
            max_lon = float(lon.max().compute()) if hasattr(lon, "compute") else float(lon.max())
        except Exception:
            max_lon = float(lon.max())
        if max_lon > 180:
            new_lon = ((lon + 180) % 360) - 180
            ds = ds.assign_coords({lon_name: new_lon})
            try:
                ds = ds.sortby(lon_name)
            except Exception:
                pass
    return ds


def _slice_bbox(ds: xr.Dataset, bbox: Tuple[float, float, float, float]) -> xr.Dataset:
    """
    bbox: (lat_min, lat_max, lon_min, lon_max)
    支持 1D 或 2D lat/lon。若缺失lat/lon，则直接返回。
    """
    lat_name = _get_lat_name(ds)
    lon_name = _get_lon_name(ds)
    if lat_name is None or lon_name is None:
        return ds

    lat = ds[lat_name]
    lon = ds[lon_name]

    lat_min, lat_max, lon_min, lon_max = bbox

    if lon.ndim == 1 and lat.ndim == 1:
        # 处理升降序
        lat_slice = slice(lat_min, lat_max) if lat[0] < lat[-1] else slice(lat_max, lat_min)
        ds_sub = ds.sel({lat_name: lat_slice})
        if lon_min <= lon_max:
            lon_slice = slice(lon_min, lon_max) if ds_sub[lon_name][0] < ds_sub[lon_name][-1] else slice(lon_max, lon_min)
            ds_sub = ds_sub.sel({lon_name: lon_slice})
        else:
            lon1 = slice(lon_min, 180) if ds_sub[lon_name][0] < ds_sub[lon_name][-1] else slice(180, lon_min)
            lon2 = slice(-180, lon_max) if ds_sub[lon_name][0] < ds_sub[lon_name][-1] else slice(lon_max, -180)
            ds_sub = xr.concat([ds_sub.sel({lon_name: lon1}), ds_sub.sel({lon_name: lon2})], dim=lon_name)
        return ds_sub
    else:
        # 曲线格网 2D
        latmask = (lat >= lat_min) & (lat <= lat_max)
        if lon_min <= lon_max:
            lonmask = (lon >= lon_min) & (lon <= lon_max)
            mask = latmask & lonmask
            return ds.where(mask, drop=True)
        else:
            lonmask1 = (lon >= lon_min) & (lon <= 180)
            lonmask2 = (lon >= -180) & (lon <= lon_max)
            mask = latmask & (lonmask1 | lonmask2)
            return ds.where(mask, drop=True)


def _ensure_time_coord(ds: xr.Dataset) -> xr.Dataset:
    # 常见时间名别名
    for cand in ["time", "t", "Time", "TIME"]:
        if cand in ds.coords or cand in ds.dims:
            if cand != "time":
                try:
                    ds = ds.rename({cand: "time"})
                except Exception:
                    pass
            break
    return ds


def _ensure_vars(ds: xr.Dataset) -> xr.Dataset:
    """把常见变量名映射到 sic, sit。若缺失则填充 NaN 变量。"""
    new_vars: Dict[str, xr.DataArray] = {}
    # SIC: siconc, ci, ice_conc 之类
    sic_name = None
    for cand in ["sic", "siconc", "ci", "ice_conc", "sea_ice_concentration"]:
        if cand in ds.data_vars:
            sic_name = cand
            break
    if sic_name is not None and sic_name != "sic":
        new_vars["sic"] = ds[sic_name]
    elif sic_name is None and "sic" not in ds:
        # 后面再创建空 NaN（等识别到 y/x）
        pass

    # SIT: sithick_corr, sithick, sit, l2_thickness 等
    sit_name = None
    for cand in ["sit", "sithick_corr", "sithick", "sea_ice_thickness", "ice_thickness"]:
        if cand in ds.data_vars:
            sit_name = cand
            break
    if sit_name is not None and sit_name != "sit":
        new_vars["sit"] = ds[sit_name]

    if new_vars:
        ds = ds.assign(new_vars)

    # 若仍缺失，创建 NaN 变量（匹配 y/x 和 time）
    ydim = "y" if "y" in ds.dims else ("lat" if "lat" in ds.dims else None)
    xdim = "x" if "x" in ds.dims else ("lon" if "lon" in ds.dims else None)

    time_len = ds.sizes.get("time", 0)
    if ydim and xdim:
        shape = (time_len, ds.sizes[ydim], ds.sizes[xdim])
        coords = {}
        if "time" in ds.coords:
            coords["time"] = ds["time"]
        coords[ydim] = ds[ydim]
        coords[xdim] = ds[xdim]
        if "sic" not in ds:
            sic_da = xr.DataArray(np.full(shape, np.nan, dtype="float32"), dims=("time", ydim, xdim), coords=coords)
            ds = ds.assign({"sic": sic_da})
        if "sit" not in ds:
            sit_da = xr.DataArray(np.full(shape, np.nan, dtype="float32"), dims=("time", ydim, xdim), coords=coords)
            ds = ds.assign({"sit": sit_da})
    else:
        # 无法构造，忽略（后续 transpose 可能报错，交给上层）
        if "sic" not in ds:
            ds = ds.assign({"sic": xr.full_like(next(iter(ds.data_vars.values())), np.nan)})
        if "sit" not in ds:
            ds = ds.assign({"sit": xr.full_like(next(iter(ds.data_vars.values())), np.nan)})

    return ds


def _postprocess_units_and_range(ds: xr.Dataset) -> xr.Dataset:
    # sic 归一化到 [0,1]
    if "sic" in ds:
        vmax = ds["sic"].max(skipna=True)
        try:
            vmax_val = float(vmax.compute()) if hasattr(vmax, "compute") else float(vmax)
        except Exception:
            vmax_val = float(vmax)
        if np.isfinite(vmax_val) and vmax_val > 1.0 + 1e-6:
            ds["sic"] = ds["sic"] / 100.0
        ds["sic"] = ds["sic"].clip(0.0, 1.0)
        ds["sic"].attrs["units"] = "1"

    # sit 转米
    if "sit" in ds:
        units = (ds["sit"].attrs.get("units", "") or "").lower().strip()
        if units in ("cm", "centimeter", "centimetre", "centimeters", "centimetres"):
            ds["sit"] = ds["sit"] / 100.0
            ds["sit"].attrs["units"] = "m"
        elif units in ("m", "meter", "metre", "meters", "metres", ""):
            ds["sit"].attrs["units"] = "m"
        # 额外清理不合理极值
        ds["sit"] = ds["sit"].where(np.isfinite(ds["sit"]))
    return ds


def _ensure_yx(ds: xr.Dataset) -> xr.Dataset:
    # 常见 y/x 命名
    rename_map = {}
    if "latitude" in ds.dims and "lat" not in ds.dims:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.dims and "lon" not in ds.dims:
        rename_map["longitude"] = "lon"
    if rename_map:
        ds = ds.rename(rename_map)

    # 如存在 lat/lon 且无 y/x，以 lat->y, lon->x 作为栅格索引
    if "y" not in ds.dims and "lat" in ds.dims:
        ds = ds.rename({"lat": "y"})
    if "x" not in ds.dims and "lon" in ds.dims:
        ds = ds.rename({"lon": "x"})

    # 变量维度统一顺序 (time,y,x) 若可能
    for v in ["sic", "sit"]:
        if v in ds:
            dims = list(ds[v].dims)
            desired = [d for d in ["time", "y", "x"] if d in dims]
            # 把其余维度附在末尾，避免丢失
            desired += [d for d in dims if d not in desired]
            ds[v] = ds[v].transpose(*desired)
    return ds


def _open_mfdataset(paths: List[str]) -> xr.Dataset:
    # 先打开一个文件推断分块
    first = xr.open_dataset(paths[0], chunks={}, decode_times=True)
    chunks = _compute_chunks(first)
    first.close()

    ds = xr.open_mfdataset(
        paths,
        combine="by_coords",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        chunks=chunks,
        parallel=True,
        decode_times=True,
        preprocess=lambda d: _ensure_time_coord(d),
    )
    return ds


@st.cache_data(show_spinner=False)
def load_nsidc_sic(path_glob: str, bbox: Tuple[float, float, float, float]) -> xr.Dataset:
    """
    读取 NSIDC 月均 SIC，返回标准 Dataset，包含变量：
    - sic: (time,y,x) 0..1
    - sit: (time,y,x) 全 NaN（NSIDC 本函数不提供厚度）
    结果坐标尽量包含 time, y, x；若文件含 lat/lon 或极射投影坐标，会尽量保留。
    """
    paths = sorted(glob.glob(path_glob))
    if len(paths) == 0:
        raise FileNotFoundError(f"No files matched: {path_glob}")

    ds = _open_mfdataset(paths)
    ds = _ensure_vars(ds)
    ds = _ensure_yx(ds)
    ds = _normalize_lon(ds)
    ds = _postprocess_units_and_range(ds)

    # NSIDC 只保留 sic，sit 填 NaN（已在 _ensure_vars 中保障）

    # 裁剪 BBOX（若存在经纬度）
    ds = _slice_bbox(ds, bbox)

    # 仅保留需要的变量与标准顺序
    keep = {"sic": ds["sic"]}
    if "sit" in ds:
        keep["sit"] = ds["sit"]
    ds = xr.Dataset(keep)

    # 统一维度顺序
    for v in ds.data_vars:
        dims = list(ds[v].dims)
        desired = [d for d in ["time", "y", "x"] if d in dims]
        desired += [d for d in dims if d not in desired]
        ds[v] = ds[v].transpose(*desired)

    return ds


@st.cache_data(show_spinner=False)
def load_cmems_sic_sit(path_glob: str, bbox: Tuple[float, float, float, float]) -> xr.Dataset:
    """
    读取 CMEMS 月均海冰数据，返回标准 Dataset：
    - sic: (time,y,x) 0..1
    - sit: (time,y,x) 单位 m
    自动识别时间轴，规范维度为 (time,y,x)，并裁剪到 bbox。
    """
    paths = sorted(glob.glob(path_glob))
    if len(paths) == 0:
        raise FileNotFoundError(f"No files matched: {path_glob}")

    ds = _open_mfdataset(paths)
    ds = _ensure_vars(ds)
    ds = _ensure_yx(ds)
    ds = _normalize_lon(ds)
    ds = _postprocess_units_and_range(ds)

    ds = _slice_bbox(ds, bbox)

    keep: Dict[str, xr.DataArray] = {}
    if "sic" in ds:
        keep["sic"] = ds["sic"]
    else:
        # 若 CMEMS 数据不含 sic，创建 NaN
        ydim = "y" if "y" in ds.dims else ("lat" if "lat" in ds.dims else None)
        xdim = "x" if "x" in ds.dims else ("lon" if "lon" in ds.dims else None)
        if ydim and xdim:
            shape = (ds.sizes.get("time", 0), ds.sizes[ydim], ds.sizes[xdim])
            coords = {"time": ds["time"]} if "time" in ds.coords else {}
            coords[ydim] = ds[ydim]; coords[xdim] = ds[xdim]
            keep["sic"] = xr.DataArray(np.full(shape, np.nan, dtype="float32"), dims=("time", ydim, xdim), coords=coords)

    if "sit" in ds:
        keep["sit"] = ds["sit"]
    else:
        # 允许 sit 缺失，后续填 NaN
        ydim = "y" if "y" in ds.dims else ("lat" if "lat" in ds.dims else None)
        xdim = "x" if "x" in ds.dims else ("lon" if "lon" in ds.dims else None)
        if ydim and xdim:
            shape = (ds.sizes.get("time", 0), ds.sizes[ydim], ds.sizes[xdim])
            coords = {"time": ds["time"]} if "time" in ds.coords else {}
            coords[ydim] = ds[ydim]; coords[xdim] = ds[xdim]
            keep["sit"] = xr.DataArray(np.full(shape, np.nan, dtype="float32"), dims=("time", ydim, xdim), coords=coords)

    ds = xr.Dataset(keep)

    # 统一维度顺序
    for v in ds.data_vars:
        dims = list(ds[v].dims)
        desired = [d for d in ["time", "y", "x"] if d in dims]
        desired += [d for d in dims if d not in desired]
        ds[v] = ds[v].transpose(*desired)

    return ds

