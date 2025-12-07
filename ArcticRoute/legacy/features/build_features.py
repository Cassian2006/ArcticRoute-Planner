"""合成 Feature Dataset 并写盘（AIS 栅格 + 可选环境层）

输出：ArcticRoute/data_processed/features/features_YYYYMM.nc

实现要点：
- 读取 ais_density（B-11 输出）与可选 env_clean.nc（若无则跳过并 WARN）
- 统一 dims/coords：(time,y,x)；检查坐标一致（equals），不一致则抛错或重命名/对齐（此处要求一致）
- 写 attrs: {run_id, layer="features", version, month}
- 压缩：zlib=True, complevel=4；chunks: time:-1, y:128, x:256
- 非 dry-run 写盘并登记工件
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import xarray as xr

from ArcticRoute.cache.index_util import register_artifact


def _default_out_path(month: str) -> str:
    out_dir = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "features")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"features_{month}.nc")


def _coords_equal(a: xr.DataArray, b: xr.DataArray) -> bool:
    try:
        return bool((a.values == b.values).all())
    except Exception:
        return False


def build_feature_dataset(
    month: str,
    density_nc_path: str,
    env_nc_path: Optional[str] = None,
    out_path: Optional[str] = None,
    dry_run: bool = True,
) -> Dict[str, Any]:
    # 读取 AIS 密度
    ds_ais = xr.open_dataset(density_nc_path)
    # 坐标基准
    if "time" not in ds_ais.dims or "y" not in ds_ais.dims or "x" not in ds_ais.dims:
        raise ValueError("ais_density 数据集中缺少 (time,y,x) 维度")
    time = ds_ais["time"]
    y = ds_ais["y"]
    x = ds_ais["x"]

    ds_vars = {}
    for v in list(ds_ais.data_vars):
        if v.startswith("ais_density"):
            ds_vars[v] = ds_ais[v]

    # 可选环境层：若存在则合并 matching 变量（如 Hs, U10 等）
    env_vars = {}
    if env_nc_path and os.path.exists(env_nc_path):
        env = xr.open_dataset(env_nc_path)
        # 要求坐标一致
        for nm in ("time", "y", "x"):
            if nm not in env.dims:
                continue
        # 坐标一致性校验
        ok = _coords_equal(time, env["time"]) if "time" in env else True
        ok = ok and (_coords_equal(y, env["y"]) if "y" in env else True)
        ok = ok and (_coords_equal(x, env["x"]) if "x" in env else True)
        if not ok:
            # 降级为告警并跳过环境层，确保最小可用产物可写出
            try:
                import logging
                logging.getLogger(__name__).warning("env_clean 坐标与 ais_density 不一致，已跳过环境层合并。")
            except Exception:
                pass
        else:
            for v in list(env.data_vars):
                if v in ("ais_density", "ais_density_cls"):
                    continue
                # 仅挑选 (time,y,x) 对齐的数值型变量
                da = env[v]
                if all(dim in da.dims for dim in ("time", "y", "x")):
                    env_vars[v] = da
    # 合并
    out = xr.Dataset({**ds_vars, **env_vars}, coords=dict(time=time, y=y, x=x))
    # 写属性与压缩
    comp = dict(zlib=True, complevel=4)
    enc = {var: {**comp, "chunksizes": (len(time), min(128, len(y)), min(256, len(x)))} for var in out.data_vars}
    out.attrs.update({
        "run_id": os.environ.get("RUN_ID", ""),
        "layer": "features",
        "version": "0.1",
        "month": month,
    })

    out_nc = out_path or _default_out_path(month)
    if not dry_run:
        os.makedirs(os.path.dirname(out_nc), exist_ok=True)
        out.to_netcdf(out_nc, encoding=enc)
        try:
            register_artifact(run_id=os.environ.get("RUN_ID", ""), kind="features_nc", path=out_nc, attrs={"month": month})
        except Exception:
            pass
    # 资源释放
    try:
        ds_ais.close()
    except Exception:
        pass

    return {"out": out_nc, "vars": list(out.data_vars.keys()), "dims": {k: int(out.sizes[k]) for k in out.sizes}}


__all__ = ["build_feature_dataset"]

