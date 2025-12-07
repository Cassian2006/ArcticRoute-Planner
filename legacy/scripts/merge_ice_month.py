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

DEF_BASE = Path("ArcticRoute")/"data_processed"/"ice_forecast"


def _ensure_yx(ds: xr.Dataset) -> xr.Dataset:
    ren = {}
    if "latitude" in ds.dims and "y" not in ds.dims:
        ren["latitude"] = "y"
    if "longitude" in ds.dims and "x" not in ds.dims:
        ren["longitude"] = "x"
    if ren:
        ds = ds.rename(ren)
    # 只保留需要的维度顺序
    if "time" in ds.dims:
        target_dims = [d for d in ["time", "y", "x"] if d in ds.dims]
        ds = ds.transpose(*target_dims, ...)
    else:
        target_dims = [d for d in ["y", "x"] if d in ds.dims]
        ds = ds.transpose(*target_dims, ...)
    return ds


def _pick_var(ds: xr.Dataset) -> xr.Dataset:
    # 目标变量 sic_pred
    if "sic_pred" in ds.data_vars:
        v = ds["sic_pred"]
    else:
        # 兼容常见命名
        cand = ["sic", "siconc", "ci", "ice_conc", "sea_ice_concentration"]
        name = None
        for k in cand:
            if k in ds.data_vars:
                name = k
                break
        if name is None:
            raise KeyError("未发现海冰浓度变量(sic_pred/sic/siconc/ci/ice_conc/sea_ice_concentration)")
        v = ds[name]
    # 归一化到[0,1]
    try:
        vmax = float(v.max().compute().item()) if hasattr(v, "compute") else float(v.max().item())
    except Exception:
        vmax = float(v.max())
    if vmax > 1.0 + 1e-6:
        v = v / 100.0
    v = v.clip(0.0, 1.0)
    return xr.Dataset(dict(sic_pred=v))


def _select_month(ds: xr.Dataset, ym: str) -> xr.Dataset:
    # 只保留指定年月的那一帧，time 维度长度为 1
    if "time" not in ds.dims:
        return ds.expand_dims(time=[pd.Timestamp(f"{ym}01")])
    y = int(ym[:4])
    m = int(ym[4:6])
    tsel = ds.time.where((ds.time.dt.year == y) & (ds.time.dt.month == m), drop=True)
    if tsel.size == 0:
        # 兜底：若 time 为月初以外的月内任意日，选择该月最近一帧
        mask = (ds.time.dt.year == y) & (ds.time.dt.month == m)
        idx = int(np.argmax(mask.values)) if mask.any() else None
        if idx is None or not bool(mask.any()):
            raise ValueError(f"在源数据中未找到目标月份 {ym} 的时间切片")
        ds = ds.isel(time=idx)
        ds = ds.expand_dims(time=[pd.Timestamp(f"{y:04d}-{m:02d}-01")])
    else:
        ds = ds.sel(time=tsel).isel(time=0)
        # 保留 time 维
        ds = ds.expand_dims(time=[pd.Timestamp(f"{y:04d}-{m:02d}-01")])
    return ds


def _write_nc(ds: xr.Dataset, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = ds.assign_attrs({"source": "ArcticRoute P1"})
    # chunk 与编码
    ds = ds.chunk({"time": 1, "y": 256, "x": 256}) if {"y","x"}.issubset(ds.dims) else ds
    enc = {
        "sic_pred": {"zlib": True, "complevel": 4, "chunksizes": tuple([ds.sizes.get(d, 1) if d != "time" else 1 for d in ["time","y","x"] if d in ds.dims])}
    }
    # 坐标编码（可选保留 chunk 信息）
    for c in ["time", "y", "x"]:
        if c in ds.coords:
            enc[c] = {"zlib": False}
    ds.to_netcdf(out_path, encoding=enc)


def find_input_paths(base_dir: Path, ym: str, use_snapshot: bool = False):
    if use_snapshot:
        # 优先使用快照：merged/_blocks_snapshot/<ym>/ 或 merged/_snapshot_ice_forecast_<ym>.nc
        snap_blocks = base_dir/"merged"/"_blocks_snapshot"/ym
        if snap_blocks.is_dir():
            paths = sorted(str(p) for p in snap_blocks.glob("block_*.nc"))
            if paths:
                return ("A", paths)
        snap_month = base_dir/"merged"/f"_snapshot_ice_forecast_{ym}.nc"
        if snap_month.exists():
            return ("B", str(snap_month))
    # 路径 A：blocks 或 _blocks
    blocks_dirs = [base_dir/"blocks"/ym, base_dir/"_blocks"/ym]
    for bdir in blocks_dirs:
        if bdir.is_dir():
            paths = sorted(str(p) for p in bdir.glob("block_*.nc"))
            if paths:
                return ("A", paths)
    # 路径 B：整月文件
    cand = base_dir / f"ice_forecast_{ym}.nc"
    if cand.exists():
        return ("B", str(cand))
    return (None, None)


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


def merge_month(ym: str, base: Path, use_snapshot: bool = False, retry_open: int = 0):
    mode, src = find_input_paths(base, ym, use_snapshot=use_snapshot)
    if mode is None:
        raise FileNotFoundError(f"未找到输入：{base}/(blocks|_blocks)/{ym}/block_*.nc 或 {base}/ice_forecast_{ym}.nc（也检查了快照）")

    if mode == "A":
        # 多文件合并
        def _pre(ds):
            ds = _ensure_yx(ds)
            ds = _pick_var(ds)
            return ds
        dsm = xr.open_mfdataset(src, combine="by_coords", preprocess=_pre, parallel=True)
        try:
            ds = _ensure_yx(dsm)
            ds = _pick_var(ds)
            ds = _select_month(ds, ym)
        finally:
            try:
                dsm.close()
            except Exception:
                pass
    else:
        with _open_with_retry(Path(src), retry_open) as dso:
            ds = _ensure_yx(dso)
            ds = _pick_var(ds)
            ds = _select_month(ds, ym)

    # 最终变量只保留 sic_pred，维度 time,y,x
    if set(ds.data_vars) != {"sic_pred"}:
        ds = xr.Dataset(dict(sic_pred=ds["sic_pred"]))
    ds = _ensure_yx(ds)

    out_dir = base/"merged"
    out_path = out_dir / f"sic_fcst_{ym}.nc"
    _write_nc(ds, out_path)
    return str(out_path)


def build_parser():
    ap = argparse.ArgumentParser(description="P1-04-01 月产品合并/抽取（MERGE）")
    ap.add_argument("--ym", required=True, help="目标年月 YYYYMM")
    ap.add_argument("--base-dir", default=str(DEF_BASE), help="基础目录 data_processed/ice_forecast")
    ap.add_argument("--use-snapshot", action="store_true", help="优先从 merged/_blocks_snapshot 或 _snapshot_ice_forecast 读取")
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    ym = args.ym
    base = Path(args.base_dir)
    out = merge_month(ym, base, use_snapshot=bool(getattr(args, "use_snapshot", False)))
    print("merged_output:", out)


if __name__ == "__main__":
    main()

