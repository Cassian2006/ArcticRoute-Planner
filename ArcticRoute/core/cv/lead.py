from __future__ import annotations
"""
cv.lead.build — 生成 lead_prob_<ym>.nc（最小可用实现）

- 目标：在网格上提供开阔水裂隙（Leads）概率的轻量通道。
- 数据依赖：若存在卫星镶嵌 sat_mosaic_*.tif，简单做亮度阈值 + 形态学平滑；否则回退为全零。
- 网格/坐标：# REUSE 与 cv.edge.build 相同模板获取策略。
"""
import os
from typing import Any, Dict

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

from .edge import _get_template  # REUSE 模板选择

essential_dir = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "cv_cache")


def build(ym: str, save: bool = True) -> Dict[str, Any]:
    if xr is None:
        raise RuntimeError("xarray required for cv.lead")
    tpl = _get_template(ym)
    Ny = int(tpl.sizes.get("y") or tpl.sizes.get("lat") or tpl.shape[-2])
    Nx = int(tpl.sizes.get("x") or tpl.sizes.get("lon") or tpl.shape[-1])
    arr = np.zeros((Ny, Nx), dtype=np.float32)
    source = "placeholder"
    try:
        import rasterio  # type: ignore
        from scipy.ndimage import median_filter  # type: ignore
        root = os.path.join(os.getcwd(), "ArcticRoute")
        cand = None
        dp = os.path.join(root, "data_processed")
        for name in os.listdir(dp):
            if name.startswith("sat_mosaic_") and name.endswith(".tif"):
                cand = os.path.join(dp, name)
                break
        if cand:
            with rasterio.open(cand) as src:
                img = src.read(1).astype(np.float32)
                vmin, vmax = float(np.nanmin(img)), float(np.nanmax(img))
                if vmax > vmin:
                    img = (img - vmin) / (vmax - vmin)
                else:
                    img = np.zeros_like(img, dtype=np.float32)
                # 简单阈值：亮区近似开水/云；为稳健，取上分位 q90 作为阈值
                q = np.nanpercentile(img, 90.0)
                mask = (img >= q).astype(np.float32)
                mask = median_filter(mask, size=3)
                yy = min(mask.shape[0], Ny)
                xx = min(mask.shape[1], Nx)
                arr[:yy, :xx] = mask[:yy, :xx]
                source = f"threshold:{os.path.basename(cand)}"
    except Exception:
        pass

    da = xr.DataArray(arr, dims=tpl.dims, coords=tpl.coords, name="lead_prob")
    da.attrs.update({"long_name": "Lead probability (proxy)", "units": "1", "norm": "[0,1]", "source": source, "grid_id": "epsg:3413"})
    ds = xr.Dataset({"lead_prob": da})
    os.makedirs(essential_dir, exist_ok=True)
    out_nc = os.path.join(essential_dir, f"lead_prob_{ym}.nc")
    if save:
        ds.to_netcdf(out_nc)
    return {"ym": ym, "out": out_nc, "shape": [Ny, Nx], "source": source}


__all__ = ["build"]



