from __future__ import annotations

from typing import Tuple

import numpy as np
import xarray as xr


def _rename_latlon_dims(ds: xr.Dataset) -> xr.Dataset:
    ren = {}
    if "latitude" in ds.dims and "y" not in ds.dims:
        ren["latitude"] = "y"
    if "longitude" in ds.dims and "x" not in ds.dims:
        ren["longitude"] = "x"
    return ds.rename(ren) if ren else ds


def _flip_if_descending(ds: xr.Dataset, dim: str, fix_order: bool) -> xr.Dataset:
    if dim not in ds.dims:
        return ds
    # 仅在该维也作为坐标存在且为1D时检查
    if dim in ds.coords and ds[dim].ndim == 1:
        coord = ds[dim].values
        if coord.size >= 2 and coord[1] < coord[0]:
            if not fix_order:
                raise ValueError(f"坐标 {dim} 非单调递增，检测到降序；可使用 --fix-order 进行翻转")
            # 翻转坐标和所有沿该维的数据
            ds = ds.sortby(dim, ascending=True)
    return ds


def _transpose_time_y_x(ds: xr.Dataset) -> xr.Dataset:
    dims_order = [d for d in ("time", "y", "x") if d in ds.dims]
    # 补上其它维度顺序不变
    others = [d for d in ds.dims if d not in dims_order]
    return ds.transpose(*(dims_order + others))


def normalize_coords(ds: xr.Dataset, fix_order: bool = False) -> xr.Dataset:
    """将数据集规范为 time,y,x 维度命名和顺序，并检查 y/x 单调性。

    - 自动将维度 latitude/longitude -> y/x（仅重命名维度，不改投影/数值）
    - 维度顺序调整为 time,y,x 在前
    - 若 y/x 坐标为降序，默认报错；当 fix_order=True 时自动翻转为升序
    - 不强制要求存在 lat/lon 坐标；如有，保留不变
    """
    ds2 = _rename_latlon_dims(ds)
    ds2 = _transpose_time_y_x(ds2)
    # 单调性检查/修复
    for d in ("y", "x"):
        ds2 = _flip_if_descending(ds2, d, fix_order=fix_order)
    return ds2


__all__ = ["normalize_coords"]

















