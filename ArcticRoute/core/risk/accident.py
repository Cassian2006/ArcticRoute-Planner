"""事故风险层（R_acc）：KDE + 曝光校正

- build_risk_accident(ym, acc_src, bandwidth_cells=2.0, eps=1.0, dry_run=True)
  读取事故点（CSV/GeoJSON），筛选同月，映射到参考网格，做二维 KDE（高斯平滑近似）得到 λ_acc。
  曝光校正：R_acc = norm_q( λ_acc / (ais_density_sum + eps) )，其中 ais_density_sum 为同月 AIS 密度在 time 维求和。
  无 AIS 密度时退化为纯 KDE（并 WARN）。
  输出：risk_accident_<ym>.nc 与 PNG（首帧）。
  注册：register_artifact(kind="risk_accident").
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    from scipy.ndimage import gaussian_filter  # type: ignore
except Exception:  # pragma: no cover
    gaussian_filter = None  # type: ignore

from ArcticRoute.io.grid_index import _load_rect_grid
from ArcticRoute.cache.index_util import register_artifact


def _parse_month(ts_val: Any) -> Optional[str]:
    # 接受 epoch_s/ms 或 ISO8601 字符串
    try:
        import pandas as _pd  # type: ignore
        if isinstance(ts_val, (int, float)):
            iv = int(ts_val)
            if iv > 10**12:
                iv = iv // 1000
            return _pd.to_datetime(iv, unit="s", utc=True).strftime("%Y%m")
        if isinstance(ts_val, str) and ts_val.strip():
            s = ts_val.strip().replace("Z", "+00:00").replace(" ", "T")
            try:
                dt = _pd.to_datetime(s, utc=True)
            except Exception:
                return None
            return dt.strftime("%Y%m")
    except Exception:
        return None
    return None


def _read_acc_points(path: str) -> List[Tuple[float, float, Optional[str]]]:
    pts: List[Tuple[float, float, Optional[str]]] = []
    if path.lower().endswith(".csv"):
        if pd is None:
            raise RuntimeError("pandas required to read CSV")
        # 更健壮的 CSV 解析：跳过坏行、自动分隔符
        try:
            df = pd.read_csv(path, on_bad_lines='skip')  # pandas>=1.3
        except TypeError:
            df = pd.read_csv(path, error_bad_lines=False, warn_bad_lines=True)  # 兼容旧版

        # 猜测列名
        lat_col = next((c for c in df.columns if str(c).lower() in ("lat","latitude")), None)
        lon_col = next((c for c in df.columns if str(c).lower() in ("lon","lng","longitude")), None)
        ts_col = next((c for c in df.columns if str(c).lower() in ("ts","time","timestamp","datetime","date")), None)
        if not lat_col or not lon_col:
            return pts
        for _, r in df.iterrows():
            try:
                lat = float(r[lat_col]); lon = float(r[lon_col])
                ts = r[ts_col] if ts_col else None
                pts.append((lat, lon, str(ts) if ts is not None else None))
            except Exception:
                continue
        return pts
    # GeoJSON
    try:
        data = json.loads(open(path, "r", encoding="utf-8").read())
        feats = data.get("features", []) if isinstance(data, dict) else []
        for f in feats:
            try:
                geom = f.get("geometry", {})
                if geom.get("type") != "Point":
                    continue
                coords = geom.get("coordinates")
                if not coords or len(coords) < 2:
                    continue
                lon, lat = float(coords[0]), float(coords[1])
                props = f.get("properties", {})
                ts = props.get("ts") or props.get("time") or props.get("timestamp")
                pts.append((lat, lon, str(ts) if ts is not None else None))
            except Exception:
                continue
    except Exception:
        # 尝试简单数组形式 [[lon,lat,ts?], ...]
        try:
            arr = json.loads(open(path, "r", encoding="utf-8").read())
            if isinstance(arr, list):
                for it in arr:
                    if isinstance(it, (list, tuple)) and len(it) >= 2:
                        lon, lat = float(it[0]), float(it[1])
                        ts = it[2] if len(it) > 2 else None
                        pts.append((lat, lon, str(ts) if ts is not None else None))
        except Exception:
            return pts
    return pts


def _quantile_norm(arr: np.ndarray, q_lo: float = 1.0, q_hi: float = 99.0) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    lo = np.nanpercentile(a, q_lo)
    hi = np.nanpercentile(a, q_hi)
    if not np.isfinite(hi - lo) or (hi - lo) <= 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    out = (a - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def build_risk_accident(ym: str, acc_src: str, bandwidth_cells: float = 2.0, eps: float = 1.0, dry_run: bool = True) -> Dict[str, Any]:
    if xr is None:
        raise RuntimeError("xarray required for risk_accident")

    # 参考网格：使用 P1 sic_fcst_<ym>.nc
    sic_path = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "ice_forecast", "merged", f"sic_fcst_{ym}.nc")
    if not os.path.exists(sic_path):
        raise FileNotFoundError(f"缺少 sic_fcst 文件: {sic_path}")
    ref_ds = xr.open_dataset(sic_path)
    # 取模板变量
    var_name = "sic_pred" if "sic_pred" in ref_ds else (list(ref_ds.data_vars)[0] if ref_ds.data_vars else None)
    tpl = ref_ds[var_name] if var_name else None

    # 网格坐标
    lat1d, lon1d = _load_rect_grid(None)
    Ny, Nx = int(lat1d.shape[0]), int(lon1d.shape[0])

    # 事故点读取
    pts = _read_acc_points(acc_src)
    # 同月过滤
    pts_m = []
    for lat, lon, ts in pts:
        m = _parse_month(ts) if ts is not None else None
        if (m is None) or (m == ym):
            pts_m.append((lat, lon))
    # 落网计数
    def _nearest_index(axis_vals: np.ndarray, values: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(axis_vals, values, side='left')
        idx_right = np.clip(idx, 0, len(axis_vals) - 1)
        idx_left = np.clip(idx - 1, 0, len(axis_vals) - 1)
        dist_right = np.abs(axis_vals[idx_right] - values)
        dist_left = np.abs(axis_vals[idx_left] - values)
        choose_left = dist_left <= dist_right
        out = np.where(choose_left, idx_left, idx_right)
        return np.clip(out, 0, len(axis_vals) - 1)

    lam = np.zeros((Ny, Nx), dtype=np.float32)
    if pts_m:
        latv = np.array([p[0] for p in pts_m], dtype=float)
        lonv = np.array([p[1] for p in pts_m], dtype=float)
        iy = _nearest_index(lat1d, latv)
        ix = _nearest_index(lon1d, lonv)
        for iyy, ixx in zip(iy, ix):
            lam[iyy, ixx] += 1.0
        # KDE 近似：高斯平滑（bandwidth 按格点数）
        if gaussian_filter is not None and float(bandwidth_cells) > 0:
            lam = gaussian_filter(lam, sigma=float(bandwidth_cells)).astype(np.float32)
    # 将 lam resize 到模板尺寸，避免与 AIS 栅格不一致
    if tpl is not None:
        ty = int(tpl.isel(time=0).sizes.get("y") if "time" in tpl.dims else tpl.sizes.get("y", lam.shape[0])) if hasattr(tpl, 'sizes') else lam.shape[0]
        tx = int(tpl.isel(time=0).sizes.get("x") if "time" in tpl.dims else tpl.sizes.get("x", lam.shape[1])) if hasattr(tpl, 'sizes') else lam.shape[1]
        if (lam.shape[0] != ty) or (lam.shape[1] != tx):
            try:
                from skimage.transform import resize as _resize  # type: ignore
                lam = _resize(lam, (ty, tx), order=1, mode='edge', anti_aliasing=True, preserve_range=True).astype(np.float32)
            except Exception:
                # 简单重复/裁剪回退
                ry = max(1, int(np.round(ty / lam.shape[0])))
                rx = max(1, int(np.round(tx / lam.shape[1])))
                lam = np.repeat(np.repeat(lam, ry, axis=0), rx, axis=1)[:ty, :tx]
                lam = lam.astype(np.float32)
    # 曝光：读取 AIS 密度
    ais_nc = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "features", f"ais_density_{ym}.nc")
    have_ais = os.path.exists(ais_nc)
    den_sum = None
    if have_ais:
        try:
            ds_ais = xr.open_dataset(ais_nc)
            if "ais_density" in ds_ais:
                den = ds_ais["ais_density"].astype("float32")
                if "time" in den.dims:
                    den_sum = den.sum(dim="time")
                else:
                    den_sum = den
                # 插值到模板
                try:
                    den_sum = den_sum.interp_like(tpl if tpl is not None else den_sum, method="nearest")
                except Exception:
                    pass
            ds_ais.close()
        except Exception:
            den_sum = None
            have_ais = False
    if den_sum is not None:
        den_arr = np.asarray(den_sum.values, dtype=np.float32)
        ratio = lam / (den_arr + float(eps))
    else:
        ratio = lam

    R = _quantile_norm(ratio, 1.0, 99.0)

    # 构建输出 Dataset（对齐模板坐标）
    if tpl is not None:
        da = xr.DataArray(R, dims=tpl.isel(time=0).dims if "time" in tpl.dims else tpl.dims, coords=tpl.isel(time=0).coords if "time" in tpl.dims else tpl.coords, name="R_acc")
    else:
        dims = ("y","x")
        coords = {"y": ("y", lat1d.astype(np.float32)), "x": ("x", lon1d.astype(np.float32))}
        da = xr.DataArray(R, dims=dims, coords=coords, name="R_acc")
    ds_out = xr.Dataset({"R_acc": da})
    ds_out["R_acc"].attrs.update({"long_name": "Accident risk (KDE / exposure)", "units": "1"})
    ds_out = ds_out.assign_attrs({"layer": "risk_accident", "ym": ym, "have_ais": int(have_ais), "bandwidth_cells": float(bandwidth_cells)})

    out_dir = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "risk")
    fig_dir = os.path.join(os.getcwd(), "reports", "figs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    out_nc = os.path.join(out_dir, f"risk_accident_{ym}.nc")
    png_path = os.path.join(fig_dir, f"risk_accident_{ym}.png")

    if not dry_run:
        comp = {"zlib": True, "complevel": 4}
        enc = {"R_acc": {**comp}}
        ds_out.to_netcdf(out_nc, encoding=enc)
        # PNG
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore
            arr = np.asarray(ds_out["R_acc"].values)
            fig = plt.figure(figsize=(7.2,3.2))
            ax = fig.add_subplot(111)
            im = ax.imshow(arr, origin="upper", cmap="inferno", vmin=0.0, vmax=1.0)
            ax.set_title(f"R_acc {ym}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(png_path, dpi=140)
            plt.close(fig)
        except Exception:
            pass
        # register
        try:
            run_id = os.environ.get("RUN_ID", "") or __import__("time").strftime("%Y%m%dT%H%M%S")
        except Exception:
            run_id = ""
        try:
            register_artifact(run_id=run_id, kind="risk_accident", path=out_nc, attrs={"ym": ym, "src": acc_src})
        except Exception:
            pass

    try:
        ref_ds.close()
    except Exception:
        pass

    return {"out": out_nc, "png": png_path, "dry_run": bool(dry_run), "have_ais": int(have_ais), "shape": {k:int(ds_out.sizes[k]) for k in ds_out.sizes}}


__all__ = ["build_risk_accident"]

