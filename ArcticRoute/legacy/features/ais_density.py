"""AIS 密度栅格化（AIS -> ais_density）

目标：按 (time_bin, y, x[, vclass]) 统计计数，生成稠密栅格。

实现要点：
- 读取分桶与网格索引列：time_bin_idx, iy, ix（由 B-09/B-10 生成）；可选 vclass
- 采用 numpy.add.at 或 np.bincount 进行稀疏聚合到致密栅格
- 结果：
  - 默认：ais_density(time,y,x)
  - 可选：ais_density_cls(time,y,x,vclass) （vclass 维度为字符串/类别，内部通过索引映射处理）
- 可选高斯平滑（关闭为默认）

约束：
- 仅在非 dry-run 写盘，并通过 register_artifact() 登记
- Windows 路径使用 os.path.join
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import xarray as xr  # 需要 xarray

try:
    from scipy.ndimage import gaussian_filter  # type: ignore
except Exception:  # pragma: no cover
    gaussian_filter = None  # type: ignore

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover
    pl = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from ArcticRoute.cache.index_util import register_artifact
from ArcticRoute.io.timebin import make_time_bins
from ArcticRoute.io.grid_index import _load_rect_grid  # type: ignore


def _read_table(path: str):
    if pl is not None:
        try:
            return pl.read_parquet(path)
        except Exception:
            pass
    if pd is not None:
        return pd.read_parquet(path)  # type: ignore
    raise RuntimeError("No dataframe engine available to read parquet")


def _get_col(tbl: Any, name: str):
    if pl is not None and isinstance(tbl, pl.DataFrame):  # type: ignore[attr-defined]
        return tbl.get_column(name).to_numpy()
    return tbl[name].to_numpy()  # type: ignore[index]


def _unique_classes(vals: np.ndarray) -> Tuple[np.ndarray, Dict[Any, int]]:
    uniq = np.unique(vals)
    mapping = {k: i for i, k in enumerate(uniq)}
    return uniq, mapping


def _ensure_output_dir(month: str) -> str:
    base = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "features")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"ais_density_{month}.nc")


def build_ais_density(
    month: str,
    parquet_paths: List[str],
    grid_path: Optional[str] = None,
    time_step: Optional[str] = None,
    by_class: bool = False,
    smooth_sigma: Optional[float] = None,
    out_path: Optional[str] = None,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """从分桶与网格索引后的 AIS 记录生成密度栅格。

    - parquet_paths: 含 time_bin_idx, iy, ix 的 parquet 文件列表（可多文件累加）
    - time_step: 若未提供，则内部按 grid_spec 或默认 6H 生成时间桶
    - by_class: 是否按 vclass 分层输出 ais_density_cls(time,y,x,vclass)
    - smooth_sigma: 可选高斯平滑 sigma（单位为格点），默认关闭
    - out_path: 目标 nc 路径，缺省为 ArcticRoute/data_processed/features/ais_density_YYYYMM.nc
    返回：{"out": path, "vars": [...], "shape": [..], "stats": {...}}
    """
    # 网格与时间轴
    lat1d, lon1d = _load_rect_grid(grid_path)
    Ny, Nx = int(lat1d.shape[0]), int(lon1d.shape[0])
    step = time_step or "6H"
    bins = make_time_bins(month, step=step)
    Nt = max(len(bins) - 1, 1)

    # 聚合计数
    if by_class:
        # 需先扫一遍取 vclass 字典
        classes: List[Any] = []
        for p in parquet_paths:
            tbl = _read_table(p)
            if (pl is not None and isinstance(tbl, pl.DataFrame) and "vclass" in tbl.columns) or (pd is not None and hasattr(tbl, 'columns') and "vclass" in tbl.columns):
                classes.extend(list(np.unique(_get_col(tbl, "vclass"))))
        if not classes:
            by_class = False  # 无 vclass 列，降级到无分层
        else:
            uniq = np.unique(np.array(classes, dtype=object))
            vclass_list = list(uniq)
            vclass_index = {v: i for i, v in enumerate(vclass_list)}
            acc = np.zeros((Nt, Ny, Nx, len(vclass_list)), dtype=np.int32)
    if not by_class:
        acc = np.zeros((Nt, Ny, Nx), dtype=np.int32)
        vclass_list = []
        vclass_index = {}

    valid_rows = 0
    for p in parquet_paths:
        tbl = _read_table(p)
        if pl is not None and isinstance(tbl, pl.DataFrame):  # type: ignore[attr-defined]
            cols = set(tbl.columns)
        else:
            cols = set(tbl.columns)  # type: ignore[attr-defined]
        required = {"time_bin_idx", "iy", "ix"}
        if not required.issubset(cols):
            continue
        tbi = _get_col(tbl, "time_bin_idx").astype(np.int64)
        iy = _get_col(tbl, "iy").astype(np.int64)
        ix = _get_col(tbl, "ix").astype(np.int64)
        mask = (tbi >= 0) & (tbi < Nt) & (iy >= 0) & (iy < Ny) & (ix >= 0) & (ix < Nx)
        if by_class and "vclass" in cols:
            cls_raw = _get_col(tbl, "vclass")
            # 对未识别类别（不在 vclass_index 的）进行过滤
            cls_idx = np.array([vclass_index.get(c, -1) for c in cls_raw], dtype=np.int64)
            mask = mask & (cls_idx >= 0)
            tbi2 = tbi[mask]
            iy2 = iy[mask]
            ix2 = ix[mask]
            ci2 = cls_idx[mask]
            valid_rows += int(mask.sum())
            # 使用 add.at 聚合
            np.add.at(acc, (tbi2, iy2, ix2, ci2), 1)  # type: ignore[index]
        else:
            tbi2 = tbi[mask]
            iy2 = iy[mask]
            ix2 = ix[mask]
            valid_rows += int(mask.sum())
            np.add.at(acc, (tbi2, iy2, ix2), 1)  # type: ignore[index]

    # 可选平滑
    if smooth_sigma and float(smooth_sigma) > 0.0 and gaussian_filter is not None:
        if by_class:
            for k in range(acc.shape[-1]):
                acc[..., k] = gaussian_filter(acc[..., k].astype(float), sigma=float(smooth_sigma)).astype(np.int32)
        else:
            acc = gaussian_filter(acc.astype(float), sigma=float(smooth_sigma)).astype(np.int32)

    # 构建 Dataset 与坐标
    time_edges = bins  # 左闭右开，bin 索引对齐左边界
    time_left = np.array([np.datetime64(int(t) * 10**9, 'ns') for t in time_edges[:-1]])
    ds_vars: Dict[str, xr.DataArray] = {}
    coords = {
        "time": ("time", time_left),
        "y": ("y", lat1d.astype(np.float32)),
        "x": ("x", lon1d.astype(np.float32)),
    }
    if by_class:
        ds_vars["ais_density_cls"] = xr.DataArray(acc, dims=("time", "y", "x", "vclass"),
                                                   coords={**coords, "vclass": ("vclass", np.array(vclass_list, dtype=object))})
        # 同时提供合计图层
        total = acc.sum(axis=-1)
        ds_vars["ais_density"] = xr.DataArray(total, dims=("time", "y", "x"), coords=coords)
    else:
        ds_vars["ais_density"] = xr.DataArray(acc, dims=("time", "y", "x"), coords=coords)

    ds = xr.Dataset(ds_vars)
    ds["ais_density"].attrs.update({"long_name": "AIS density (counts)", "units": "count"})
    if by_class:
        ds["ais_density_cls"].attrs.update({"long_name": "AIS density by class (counts)", "units": "count"})

    # 写盘
    out_nc = out_path or _ensure_output_dir(month)
    if not dry_run:
        comp = dict(zlib=True, complevel=4)
        enc = {var: {**comp, "chunksizes": (len(ds["time"]), min(128, len(ds["y"])), min(256, len(ds["x"]))) } for var in ds.data_vars}
        ds.attrs.update({
            "run_id": os.environ.get("RUN_ID", ""),
            "layer": "ais_density",
            "version": "0.1",
            "month": month,
            "time_step": step,
        })
        os.makedirs(os.path.dirname(out_nc), exist_ok=True)
        ds.to_netcdf(out_nc, encoding=enc)
        try:
            register_artifact(run_id=os.environ.get("RUN_ID", ""), kind="ais_density", path=out_nc, attrs={"month": month, "by_class": bool(by_class)})
        except Exception:
            pass

    return {"out": out_nc, "vars": list(ds.data_vars.keys()), "shape": {k: int(ds.sizes[k]) for k in ds.sizes}, "valid_rows": valid_rows}


__all__ = ["build_ais_density"]

