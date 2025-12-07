import os
import sys
from pathlib import Path
import argparse
import xarray as xr
import numpy as np
import pandas as pd

# 确保项目根目录在 sys.path
try:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except Exception:
    pass

DEF_BASE = Path("ArcticRoute")/"data_processed"/"ice_forecast"/"merged"


def _ensure_yx(ds: xr.Dataset) -> xr.Dataset:
    ren = {}
    if "latitude" in ds.dims and "y" not in ds.dims:
        ren["latitude"] = "y"
    if "longitude" in ds.dims and "x" not in ds.dims:
        ren["longitude"] = "x"
    if ren:
        ds = ds.rename(ren)
    # 维度顺序 time,y,x
    if "time" in ds.dims:
        target_dims = [d for d in ["time", "y", "x"] if d in ds.dims]
        ds = ds.transpose(*target_dims, ...)
    else:
        target_dims = [d for d in ["y", "x"] if d in ds.dims]
        ds = ds.transpose(*target_dims, ...)
    return ds


def _pick_sic(ds: xr.Dataset) -> xr.DataArray:
    if "sic_pred" in ds.data_vars:
        v = ds["sic_pred"]
    else:
        # 兼容备选名
        for k in ["sic","siconc","ci","ice_conc","sea_ice_concentration"]:
            if k in ds.data_vars:
                v = ds[k]
                break
        else:
            raise KeyError("未发现海冰浓度变量(sic_pred/sic/siconc/ci/ice_conc/sea_ice_concentration)")
    # 若为百分比，归一化
    try:
        vmax = float(v.max().compute().item()) if hasattr(v, "compute") else float(v.max().item())
    except Exception:
        vmax = float(v.max())
    if vmax > 1.0 + 1e-6:
        v = v / 100.0
    v = v.clip(0.0, 1.0)
    return v


def _sic_to_cost(sic: xr.DataArray, t0: float, t1: float, gamma: float, alpha: float) -> xr.DataArray:
    # 分段线性+幂次
    # sic < t0 -> 0
    # t0 <= sic < t1 -> alpha * ((sic - t0)/(t1 - t0))**gamma
    # sic >= t1 -> 1
    # NaN 作为高代价处理为 1
    sic_clipped = sic.clip(0.0, 1.0)
    denom = max(1e-12, float(t1 - t0))
    scaled = ((sic_clipped - t0)/denom).clip(0.0, 1.0)
    mid = alpha * (scaled ** gamma)
    cost = xr.where(sic_clipped < t0, 0.0, xr.where(sic_clipped < t1, mid, 1.0))
    # NaN -> 1.0 高代价
    cost = cost.fillna(1.0)
    # 数值稳定与范围裁剪
    cost = cost.clip(0.0, 1.0)
    cost.name = "ice_cost"
    return cost


essential_enc_coords = {"time": {"zlib": False}, "y": {"zlib": False}, "x": {"zlib": False}}

def _open_with_retry(path: Path, retry: int = 0):
    last = None
    for k in range(max(0, int(retry)) + 1):
        try:
            return xr.open_dataset(path)
        except Exception as e:
            last = e
            if k < retry:
                import time as _t
                _t.sleep(0.5)
            else:
                raise last


def map_month(ym: str, base_merged: Path, t0: float, t1: float, gamma: float, alpha: float, retry_open: int = 0) -> str:
    in_path = base_merged / f"sic_fcst_{ym}.nc"
    if not in_path.exists():
        raise FileNotFoundError(f"未找到输入文件: {in_path}")
    with _open_with_retry(in_path, retry_open) as ds:
        ds = _ensure_yx(ds)
        sic = _pick_sic(ds)

        cost = _sic_to_cost(sic, t0=t0, t1=t1, gamma=gamma, alpha=alpha)
        out = xr.Dataset(dict(ice_cost=cost))

        # 继承 time 坐标，确保对齐
        out = out.assign_coords({k: ds.coords[k] for k in ["time","y","x"] if k in ds.coords})

        # attrs
        out = out.assign_attrs({
            "source": "ArcticRoute P1",
            "method": f"t0={t0},t1={t1},gamma={gamma},alpha={alpha}",
        })

        # chunk + 编码
        if {"y","x"}.issubset(out.dims):
            out = out.chunk({"time": 1, "y": 256, "x": 256})
            chunks = tuple([out.sizes.get(d, 1) if d != "time" else 1 for d in ["time","y","x"] if d in out.dims])
        else:
            chunks = None

        enc = {"ice_cost": {"zlib": True, "complevel": 4}}
        if chunks is not None:
            enc["ice_cost"]["chunksizes"] = chunks
        for c, cfg in essential_enc_coords.items():
            if c in out.coords:
                enc[c] = cfg

        out_dir = base_merged
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"ice_cost_{ym}.nc"
        out.to_netcdf(out_path, encoding=enc)
    return str(out_path)


def build_parser():
    ap = argparse.ArgumentParser(description="P1-04-02 冰情→代价映射（COST）")
    ap.add_argument("--ym", required=True, help="目标年月 YYYYMM")
    ap.add_argument("--base-merged", default=str(DEF_BASE), help="merged 目录：data_processed/ice_forecast/merged")
    ap.add_argument("--t0", type=float, default=0.10)
    ap.add_argument("--t1", type=float, default=0.75)
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--alpha", type=float, default=0.6)
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    ym = args.ym
    base = Path(args.base_merged)
    out = map_month(ym, base, t0=args.t0, t1=args.t1, gamma=args.gamma, alpha=args.alpha)
    print("cost_output:", out)


if __name__ == "__main__":
    main()

