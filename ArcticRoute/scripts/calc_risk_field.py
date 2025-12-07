"""Compute risk field layers (ice/accident/etc.) and write to NetCDF.

@role: pipeline
"""

﻿#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert ERA5 single-layer fields (u10/v10/swh/siconc) to an environmental risk layer.

Features:
- Auto-detect NetCDF engine and coordinate names
- Print available data variables
- Auto-scale risk components based on file contents
- Save processed dataset to data_processed/env_clean.nc by default

Usage:
  python scripts/calc_risk_field.py --in data_raw/era5/era5_env_2023_q1.nc [--out data_processed/env_clean.nc]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

DEFAULT_ENV_CFG = Path("config/env.yaml")


def load_scale_config(path: Path = DEFAULT_ENV_CFG) -> tuple[float, float, str]:
    low_pct, high_pct = 5.0, 95.0
    if path.exists():
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            quant_cfg = cfg.get("scale_quantiles", {}) if isinstance(cfg, dict) else {}
            low_pct = float(quant_cfg.get("low", low_pct))
            high_pct = float(quant_cfg.get("high", high_pct))
        except Exception as err:
            print(f"[WARN] 读取 {path} 失败，使用默认分位数: {err}")
    low_pct = max(0.0, min(low_pct, 100.0))
    high_pct = max(0.0, min(high_pct, 100.0))
    if low_pct >= high_pct:
        print("[WARN] scale_quantiles 配置不合法，恢复默认 [5,95]")
        low_pct, high_pct = 5.0, 95.0
    return low_pct / 100.0, high_pct / 100.0, f"[{low_pct:.0f},{high_pct:.0f}]"


def open_dataset_auto_engine(path: Path) -> xr.Dataset:
    candidates = [None, "h5netcdf", "netcdf4", "scipy"]
    available = set(xr.backends.list_engines())
    tried = []
    for engine in candidates:
        if engine and engine not in available:
            continue
        label = engine or "default"
        try:
            kwargs = {"engine": engine} if engine else {}
            ds = xr.open_dataset(path, **kwargs)
            print(f"[INFO] 使用 NetCDF engine: '{label}'")
            return ds
        except Exception as err:
            tried.append((label, err))
    hints = ", ".join(f"{name}: {repr(err)}" for name, err in tried) or "无可用引擎"
    raise RuntimeError(f"[ERR] 无法打开 NetCDF 文件，尝试的引擎 -> {hints}")


def pick_var(ds, candidates, what):
    for name in candidates:
        if name in ds.variables:
            print(f"[INFO] {what}: 使用变量 '{name}'")
            return ds[name]
    raise KeyError(f"[ERR] 找不到 {what}，尝试过：{candidates}\n现有变量：{list(ds.data_vars)}")


def detect_coords(ds):
    lat_name = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else None)
    lon_name = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else None)
    time_name = "time" if "time" in ds.coords else None
    if not all([lat_name, lon_name, time_name]):
        raise KeyError(
            f"[ERR] 坐标缺失。检测到 coords: {list(ds.coords)}，需要包含 time 以及 lat/latitude、lon/longitude"
        )
    print(f"[INFO] 坐标 -> time:'{time_name}', lat:'{lat_name}', lon:'{lon_name}'")
    return time_name, lat_name, lon_name


def xr_clip(da, lo, hi):
    return da.where(da >= lo, lo).where(da <= hi, hi)


def minmax(x, lo, hi):
    span = hi - lo
    if np.isclose(span, 0.0):
        return xr.zeros_like(x)
    return xr_clip((x - lo) / span, 0.0, 1.0)


def mp_mean(vals, weights, p=2.0, dim="var"):
    w = xr.DataArray(weights, dims=[dim])
    return ((w * (vals**p)).sum(dim=dim) / w.sum()) ** (1.0 / p)


def auto_scaled_risk(field, name, q_low=0.05, q_high=0.95):
    def _scalar_quantile(da, q):
        return float(da.quantile(q).values)

    lo = _scalar_quantile(field, q_low)
    hi = _scalar_quantile(field, q_high)
    if np.isclose(lo, hi):
        lo = float(field.min().values)
        hi = float(field.max().values)
    if np.isclose(lo, hi):
        print(f"[WARN] {name} 数值几乎恒定，风险设为 0")
        return xr.zeros_like(field)
    print(f"[INFO] {name} 风险映射范围: lo={lo:.3f}, hi={hi:.3f}")
    return minmax(field, lo, hi)


def main(args):
    in_path = Path(args.infile)
    out_path = Path(args.outfile)
    if not in_path.exists():
        print(f"[ERR] 输入文件不存在：{in_path}")
        sys.exit(1)

    print(f"[INFO] 读取：{in_path}")
    try:
        ds = open_dataset_auto_engine(in_path)
    except RuntimeError as err:
        print(err)
        sys.exit(1)

    print(f"[INFO] 数据变量: {', '.join(ds.data_vars)}")

    q_low, q_high, q_label = load_scale_config()

    t_name, lat_name, lon_name = detect_coords(ds)
    u = pick_var(ds, ["u10", "10m_u_component_of_wind"], "10米U风")
    v = pick_var(ds, ["v10", "10m_v_component_of_wind"], "10米V风")
    swh = pick_var(ds, ["swh", "significant_height_of_combined_wind_waves_and_swell"], "有效波高")
    ice = pick_var(ds, ["siconc", "sea_ice_cover"], "海冰浓度")

    wind = (u**2 + v**2) ** 0.5
    wave = swh

    ice_max = float(ice.max().values)
    if ice_max > 1.1:
        print(f"[WARN] 检测到海冰浓度最大值 {ice_max:.2f} > 1，按百分比处理并除以 100")
        icec = ice * 0.01
    else:
        icec = ice

    risk_wind = auto_scaled_risk(wind, "wind_speed", q_low=q_low, q_high=q_high)
    risk_wave = auto_scaled_risk(wave, "wave_height", q_low=q_low, q_high=q_high)
    risk_ice = auto_scaled_risk(icec, "ice_conc", q_low=q_low, q_high=q_high)

    risk_env = mp_mean(
        xr.concat([risk_wind, risk_wave, risk_ice], dim="var"),
        weights=[0.45, 0.35, 0.20],
        p=2.0,
        dim="var",
    ).astype("float32")

    attrs = dict(
        description="ERA5 environmental risk layer (auto-detected variables and coordinates)",
        scale_quantiles=q_label,
    )

    ds_out = xr.Dataset(
        data_vars=dict(
            wind_speed=wind.astype("float32"),
            wave_height=wave.astype("float32"),
            ice_conc=icec.astype("float32"),
            risk_wind=risk_wind.astype("float32"),
            risk_wave=risk_wave.astype("float32"),
            risk_ice=risk_ice.astype("float32"),
            risk_env=risk_env,
        ),
        coords={
            "time": ds[t_name].copy(),
            "latitude": ds[lat_name].copy(),
            "longitude": ds[lon_name].copy(),
        },
        attrs=attrs,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {k: {"zlib": True, "complevel": 4} for k in ds_out.data_vars}
    ds_out.to_netcdf(out_path, encoding=encoding)
    print(f"[OK] 已保存：{out_path}")
    ds.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--out", dest="outfile", default="data_processed/env_clean.nc")
    main(parser.parse_args())
