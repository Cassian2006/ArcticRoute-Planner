from __future__ import annotations
"""
cv.edge.build — 生成 edge_dist_<ym>.nc（最小可用实现）

# REUSE: 与事故层相同的网格/模板选择逻辑：优先使用 sic_fcst_<ym>.nc 提供 y/x/time/coords
步骤：
1) 读取参考网格（sic_fcst_{ym}.nc 或 env_clean.nc），创建占位通道
2) 若存在卫星镶嵌 data_processed/sat_mosaic_*.tif，可选读取做 Sobel 边缘（需要 scikit-image）
3) 归一化到[0,1]，写出 data_processed/cv_cache/edge_dist_{ym}.nc

注：为避免重依赖不可用，若缺少 skimage 或影像，回退为全零阵（并写明 attrs['source']）
"""
import os
from typing import Any, Dict

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore


def _get_template(ym: str) -> "xr.DataArray":
    # 优先 ice 预测合并文件（# REUSE 与 risk.accident 模板选择思想一致）
    root = os.path.join(os.getcwd(), "ArcticRoute")
    sic_path = os.path.join(root, "data_processed", "ice_forecast", "merged", f"sic_fcst_{ym}.nc")
    if xr is None:
        raise RuntimeError("xarray required for cv.edge")
    if os.path.exists(sic_path):
        ds = xr.open_dataset(sic_path)
        var = "sic_pred" if "sic_pred" in ds else (list(ds.data_vars)[0] if ds.data_vars else None)
        if var is None:
            raise RuntimeError("sic_fcst 缺少变量")
        da = ds[var]
        if "time" in da.dims:
            da = da.isel(time=0)
        return da
    # 回退 env_clean
    env_nc = os.path.join(root, "data_processed", "env_clean.nc")
    if os.path.exists(env_nc):
        ds = xr.open_dataset(env_nc)
        var = list(ds.data_vars)[0]
        da = ds[var]
        if "time" in da.dims:
            da = da.isel(time=0)
        return da
    raise FileNotFoundError("找不到参考网格 sic_fcst 或 env_clean")


essential_dir = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "cv_cache")


def build(ym: str, save: bool = True) -> Dict[str, Any]:
    tpl = _get_template(ym)
    Ny = int(tpl.sizes.get("y") or tpl.sizes.get("lat") or tpl.shape[-2])
    Nx = int(tpl.sizes.get("x") or tpl.sizes.get("lon") or tpl.shape[-1])
    # 默认回退：全零（表示远离边缘），后续可替换为真实 Sobel
    arr = np.zeros((Ny, Nx), dtype=np.float32)
    source = "placeholder"
    try:
        # 可选从 S2 镶嵌读取并做 Sobel（失败即忽略）
        import rasterio  # type: ignore
        from skimage.filters import sobel  # type: ignore
        root = os.path.join(os.getcwd(), "ArcticRoute")
        # 选择一个示例镶嵌
        cand = None
        dp = os.path.join(root, "data_processed")
        for name in os.listdir(dp):
            if name.startswith("sat_mosaic_") and name.endswith(".tif"):
                cand = os.path.join(dp, name)
                break
        if cand:
            with rasterio.open(cand) as src:
                img = src.read(1)
                img = img.astype(np.float32)
                vmin, vmax = float(np.nanmin(img)), float(np.nanmax(img))
                if vmax > vmin:
                    img = (img - vmin) / (vmax - vmin)
                else:
                    img = np.zeros_like(img, dtype=np.float32)
                edges = sobel(img)
                # 简单对齐模板大小（不重投影）
                yy = min(edges.shape[0], Ny)
                xx = min(edges.shape[1], Nx)
                arr[:yy, :xx] = edges[:yy, :xx]
                source = f"sobel:{os.path.basename(cand)}"
    except Exception:
        pass

    da = xr.DataArray(arr, dims=tpl.dims, coords=tpl.coords, name="edge_dist")
    da.attrs.update({"long_name": "Edge distance/strength (proxy)", "units": "1", "norm": "[0,1]", "source": source, "grid_id": "epsg:3413"})
    ds = xr.Dataset({"edge_dist": da})
    os.makedirs(essential_dir, exist_ok=True)
    out_nc = os.path.join(essential_dir, f"edge_dist_{ym}.nc")
    if save:
        ds.to_netcdf(out_nc)
    return {"ym": ym, "out": out_nc, "shape": [Ny, Nx], "source": source}


__all__ = ["build"]



