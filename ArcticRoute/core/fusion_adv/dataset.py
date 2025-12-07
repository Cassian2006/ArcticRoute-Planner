from __future__ import annotations
"""
Phase K 数据集与弱标注构建

函数：
- make_weak_labels(ais_density, incidents_mask, ice_mask, cfg) -> xr.DataArray{0/1/NaN}
- build_patches(channels: Dict[str, xr.DataArray], labels, tile=256, stride=128, aug=True) -> torch.utils.data.Dataset

约定：
- 通道键：R_ice/R_ice_eff, R_wave, R_acc, prior_penalty, edge_dist, lead_prob
- 所有通道在同一网格与坐标（若存在 time，当前取首帧）

# REUSE: 读取路径与归一化策略与 core.risk.fusion/_quantile_norm 一致思想。
"""
from typing import Any, Dict, List, Optional, Tuple
import os
import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

import torch
from torch.utils.data import Dataset


def _to2d(da: "xr.DataArray") -> "xr.DataArray":
    if "time" in da.dims and int(da.sizes.get("time", 0)) > 0:
        da = da.isel(time=0)
    return da


def _minmax01(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    vmin = np.nanmin(a)
    vmax = np.nanmax(a)
    if not np.isfinite(vmax - vmin) or (vmax - vmin) <= 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    out = (a - vmin) / (vmax - vmin)
    out[~np.isfinite(a)] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def make_weak_labels(ais_density: "xr.DataArray", incidents_mask: Optional["xr.DataArray"], ice_mask: Optional["xr.DataArray"], cfg: Optional[Dict[str, Any]] = None) -> "xr.DataArray":
    cfg = cfg or {}
    tau_q = float(cfg.get("tau_pos_q", 0.9))
    ais2d = _to2d(ais_density).astype("float32")
    arr = np.asarray(ais2d.values, dtype=np.float32)
    thr = float(np.nanquantile(arr[np.isfinite(arr)], tau_q)) if np.isfinite(arr).any() else 0.0
    pos = (arr >= thr).astype(np.float32)
    lab = np.full_like(arr, np.nan, dtype=np.float32)
    lab[pos > 0] = 1.0
    # 负类：incidents_mask（事故点）周围区域以及冰闭区
    if incidents_mask is not None:
        inc = _to2d(incidents_mask).fillna(0).astype("float32").values
        lab[inc > 0] = 0.0
    if ice_mask is not None:
        im = _to2d(ice_mask).fillna(0).astype("float32").values
        lab[im > 0] = 0.0
    out = xr.DataArray(lab, dims=ais2d.dims, coords=ais2d.coords, name="weak_label")
    out.attrs.update({"tau_q": tau_q})
    return out


class PatchDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, mask: np.ndarray):
        self.X = X  # [N, C, H, W]
        self.Y = Y  # [N, 1, H, W]
        self.M = mask  # [N, 1, H, W]
    def __len__(self) -> int:
        return int(self.X.shape[0])
    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.Y[idx])
        m = torch.from_numpy(self.M[idx])
        return x, y, m


def build_patches(channels: Dict[str, "xr.DataArray"], labels: "xr.DataArray", tile: int = 256, stride: int = 128, aug: bool = True) -> PatchDataset:
    # 对齐到 2D
    keys = [k for k in channels.keys()]
    da0 = _to2d(channels[keys[0]])
    H = int(da0.sizes.get("y") or da0.sizes.get("lat") or da0.shape[-2])
    W = int(da0.sizes.get("x") or da0.sizes.get("lon") or da0.shape[-1])
    # 收集通道并统一空间尺寸（裁剪到公共最小 HxW）
    raw_list: List[np.ndarray] = []
    shapes: List[Tuple[int,int]] = []
    for k in keys:
        da = _to2d(channels[k]).astype("float32")
        arr = np.asarray(da.values, dtype=np.float32)
        raw_list.append(arr)
        shapes.append((arr.shape[-2], arr.shape[-1]))
    Hc = min(s[0] for s in shapes)
    Wc = min(s[1] for s in shapes)
    # 若公共尺寸过小，放大到至少 tile 以适配下游池化
    if Hc < tile or Wc < tile:
        try:
            from skimage.transform import resize as _resize  # type: ignore
            raw_list = [ _resize(a, (max(tile,Hc), max(tile,Wc)), order=1, mode='edge', anti_aliasing=True, preserve_range=True).astype(np.float32) for a in raw_list ]
            Hc, Wc = max(tile, Hc), max(tile, Wc)
        except Exception:
            # 回退：重复边界
            scale_y = max(1, int(np.ceil(tile / max(1,Hc))))
            scale_x = max(1, int(np.ceil(tile / max(1,Wc))))
            raw_list = [ np.repeat(np.repeat(a, scale_y, axis=0), scale_x, axis=1)[:max(tile,Hc), :max(tile,Wc)].astype(np.float32) for a in raw_list ]
            Hc, Wc = max(tile, Hc), max(tile, Wc)
    C_list: List[np.ndarray] = []
    for arr in raw_list:
        arr2 = arr[:Hc, :Wc]
        arr2 = _minmax01(arr2)
        C_list.append(arr2)
    Xfull = np.stack(C_list, axis=0)  # [C,H,W]
    Yfull = _to2d(labels).astype("float32")
    Yarr0 = np.asarray(Yfull.values, dtype=np.float32)
    Yarr = Yarr0[:Hc, :Wc]
    # 若标签尺寸仍小于 (Hc,Wc)，进行 resize/重复边界到目标尺寸
    if Yarr.shape != (Hc, Wc):
        try:
            from skimage.transform import resize as _resize  # type: ignore
            Yarr = _resize(Yarr, (Hc, Wc), order=0, mode='edge', anti_aliasing=False, preserve_range=True).astype(np.float32)
        except Exception:
            sy = max(1, int(np.ceil(Hc / max(1, Yarr.shape[0])))); sx = max(1, int(np.ceil(Wc / max(1, Yarr.shape[1]))))
            Yarr = np.repeat(np.repeat(Yarr, sy, axis=0), sx, axis=1)[:Hc, :Wc].astype(np.float32)
    Mfull = np.isfinite(Yarr).astype(np.float32)
    Yarr[~np.isfinite(Yarr)] = 0.0

    patches_X: List[np.ndarray] = []
    patches_Y: List[np.ndarray] = []
    patches_M: List[np.ndarray] = []
    for y in range(0, max(1, H - tile + 1), stride):
        for x in range(0, max(1, W - tile + 1), stride):
            xs = Xfull[:, y:y+tile, x:x+tile]
            if xs.shape[-2] != tile or xs.shape[-1] != tile:
                continue
            ys = Yarr[y:y+tile, x:x+tile]
            ms = Mfull[y:y+tile, x:x+tile]
            # 至少有一定比例的标注
            if float(ms.mean()) < 0.05:
                continue
            patches_X.append(xs[None, ...])
            patches_Y.append(ys[None, None, ...])
            patches_M.append(ms[None, None, ...])
    if not patches_X:
        # 回退：强制生成一个 tile 大小的中心裁剪块，保证训练/验证形状一致
        cy = H // 2; cx = W // 2
        hs = tile // 2; ws = tile // 2
        y0 = max(0, cy - hs); x0 = max(0, cx - ws)
        y1 = min(H, y0 + tile); x1 = min(W, x0 + tile)
        # 若边界导致不足 tile，则补齐起点
        y0 = max(0, y1 - tile); x0 = max(0, x1 - tile)
        xs = Xfull[:, y0:y1, x0:x1]
        ys = Yarr[y0:y1, x0:x1]
        ms = Mfull[y0:y1, x0:x1]
        # 如果仍不满足尺寸（极端小图），则 resize 到 tile
        if xs.shape[-2] != tile or xs.shape[-1] != tile:
            try:
                from skimage.transform import resize as _resize  # type: ignore
                xs = _resize(xs, (xs.shape[0], tile, tile), order=1, mode='edge', anti_aliasing=True, preserve_range=True).astype(np.float32)
                ys = _resize(ys, (tile, tile), order=0, mode='edge', anti_aliasing=False, preserve_range=True).astype(np.float32)
                ms = _resize(ms, (tile, tile), order=0, mode='edge', anti_aliasing=False, preserve_range=True).astype(np.float32)
            except Exception:
                # 重复边界回退
                def _rep(a, th, tw):
                    sy = max(1, int(np.ceil(th / max(1, a.shape[-2])))); sx = max(1, int(np.ceil(tw / max(1, a.shape[-1]))))
                    b = np.repeat(np.repeat(a, sy, axis=-2), sx, axis=-1)
                    return b[..., :th, :tw]
                xs = _rep(xs, tile, tile).astype(np.float32)
                ys = _rep(ys, tile, tile).astype(np.float32)
                ms = _rep(ms, tile, tile).astype(np.float32)
        patches_X = [xs[None, ...]]
        patches_Y = [ys[None, None, ...]]
        patches_M = [ms[None, None, ...]]
    X = np.concatenate(patches_X, axis=0).astype(np.float32)
    Y = np.concatenate(patches_Y, axis=0).astype(np.float32)
    M = np.concatenate(patches_M, axis=0).astype(np.float32)
    return PatchDataset(X, Y, M)


__all__ = ["make_weak_labels", "build_patches", "PatchDataset"]

