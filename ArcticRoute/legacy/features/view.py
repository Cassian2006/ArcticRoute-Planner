"""AIS 图层可视化钩子（不侵入 app_min）

提供简易渲染函数：
- render_ais_layer(ds_or_path, cmap="viridis")：渲染 ais_density 或其时间均值

说明：
- 若传入路径，内部以 xarray 打开；若传入 Dataset/DataArray 则直接使用
- 优先变量名：ais_density；若不存在，则对 ais_density_cls 沿 vclass 求和
- 返回 (fig, ax)，由调用方决定如何在 UI 中展示（例如 st.pyplot）
"""
from __future__ import annotations

from typing import Any, Tuple
import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def _pick_da(ds_or_da: Any) -> xr.DataArray:
    if isinstance(ds_or_da, xr.DataArray):
        return ds_or_da
    if isinstance(ds_or_da, xr.Dataset):
        if "ais_density" in ds_or_da:
            return ds_or_da["ais_density"]
        if "ais_density_cls" in ds_or_da:
            return ds_or_da["ais_density_cls"].sum(dim="vclass")
    raise ValueError("输入应为含 ais_density(_cls) 的 xarray 数据或路径")


def render_ais_layer(ds_or_path: Any, cmap: str = "viridis") -> Tuple[Any, Any]:
    """渲染 AIS Density（时间均值）。返回 (fig, ax)。"""
    need_close = False
    if isinstance(ds_or_path, (str, os.PathLike)):
        ds = xr.open_dataset(ds_or_path)  # 调用方负责持久化与关闭；此处短期打开
        need_close = True
    else:
        ds = ds_or_path
    try:
        da = _pick_da(ds)
        arr = da.values
        if arr.ndim == 3:
            arr2 = np.nanmean(arr, axis=0)
        else:
            arr2 = arr
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(arr2, origin="lower", cmap=cmap)
        ax.set_title("AIS Density (mean)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        return fig, ax
    finally:
        if need_close:
            try:
                ds.close()
            except Exception:
                pass


__all__ = ["render_ais_layer"]

