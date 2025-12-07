#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualization of accident overlays vs routes for exploratory analysis.

@role: analysis
"""

"""
事故样本与风险场对比分析脚本。

流程：
1. 读取事故记录（data_raw/incidents.csv）；
2. 映射至 env_clean.nc 网格，取得对应 risk_env；
3. 绘制风险栅格 + 事故叠加，以及事故/非事故风险分布直方图；
4. 输出 docs/accident_compare.png，并打印关键统计指标。
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

REQUIRED_COLUMNS = {"mmsi", "time_utc", "lat", "lon", "incident"}


def load_incidents(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"事故数据不存在: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise KeyError(f"事故数据缺少必要字段: {sorted(missing)}")
    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["incident"] = pd.to_numeric(df["incident"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["time_utc", "lat", "lon"])
    if df.empty:
        raise ValueError("事故数据为空")
    return df


def map_to_risk(df: pd.DataFrame, ds: xr.Dataset) -> pd.DataFrame:
    risk_da = ds["risk_env"]
    times = xr.DataArray(df["time_utc"].values.astype("datetime64[ns]"), dims="points")
    lat_da = xr.DataArray(df["lat"].values.astype("float32"), dims="points")
    lon_da = xr.DataArray(df["lon"].values.astype("float32"), dims="points")
    selected = risk_da.sel(time=times, latitude=lat_da, longitude=lon_da, method="nearest")

    df = df.copy()
    df["risk_env"] = selected.values.astype("float32")
    df["risk_time"] = pd.to_datetime(selected.coords["time"].values)
    df["risk_lat"] = selected.coords["latitude"].values.astype("float32")
    df["risk_lon"] = selected.coords["longitude"].values.astype("float32")
    return df


def build_incident_density(df: pd.DataFrame, ds: xr.Dataset) -> np.ndarray:
    lat_vals = ds["latitude"].values
    lon_vals = ds["longitude"].values
    lat_map: Dict[float, int] = {round(float(v), 4): idx for idx, v in enumerate(lat_vals)}
    lon_map: Dict[float, int] = {round(float(v), 4): idx for idx, v in enumerate(lon_vals)}

    density = np.zeros((len(lat_vals), len(lon_vals)), dtype="float32")
    df_pos = df[df["incident"] == 1]
    for _, row in df_pos.iterrows():
        lat_idx = lat_map.get(round(float(row["risk_lat"]), 4))
        lon_idx = lon_map.get(round(float(row["risk_lon"]), 4))
        if lat_idx is None or lon_idx is None:
            continue
        density[lat_idx, lon_idx] += 1.0

    if density.max() > 0:
        density /= density.max()
    return density


def plot_overlay(df: pd.DataFrame, ds: xr.Dataset, density: np.ndarray, out_path: Path, bins: int = 20):
    risk_mean = ds["risk_env"].mean(dim="time")
    lat_vals = ds["latitude"].values
    lon_vals = ds["longitude"].values

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))

    mesh = ax0.pcolormesh(lon_vals, lat_vals, risk_mean, shading="auto", cmap="viridis")
    plt.colorbar(mesh, ax=ax0, label="平均 risk_env")
    if np.any(density > 0):
        ax0.contourf(lon_vals, lat_vals, density, levels=[0.1, 0.3, 0.6, 1.0], cmap="Reds", alpha=0.4)
    ax0.scatter(
        df.loc[df["incident"] == 1, "lon"],
        df.loc[df["incident"] == 1, "lat"],
        c="red",
        s=25,
        label="事故样本",
        edgecolors="white",
    )
    ax0.scatter(
        df.loc[df["incident"] == 0, "lon"],
        df.loc[df["incident"] == 0, "lat"],
        c="cyan",
        s=20,
        alpha=0.5,
        label="非事故样本",
        edgecolors="none",
    )
    ax0.set_title("平均风险场与事故叠加")
    ax0.set_xlabel("经度")
    ax0.set_ylabel("纬度")
    ax0.legend(loc="upper right")

    incident_risk = df[df["incident"] == 1]["risk_env"].dropna()
    normal_risk = df[df["incident"] == 0]["risk_env"].dropna()
    ax1.hist(normal_risk, bins=bins, alpha=0.7, label=f"非事故 (n={len(normal_risk)})", color="#4daf4a")
    ax1.hist(incident_risk, bins=bins, alpha=0.7, label=f"事故 (n={len(incident_risk)})", color="#e41a1c")
    ax1.set_xlabel("risk_env")
    ax1.set_ylabel("频数")
    ax1.set_title("事故 vs 非事故 风险分布")
    ax1.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="事故样本与风险场对比分析")
    parser.add_argument("--incidents", default="data_raw/incidents.csv", help="事故 CSV 路径")
    parser.add_argument("--env", default="data_processed/env_clean.nc", help="风险场 NetCDF 路径")
    parser.add_argument("--out", default="docs/accident_compare.png", help="输出图像路径")
    parser.add_argument("--hist-bins", type=int, default=20, help="直方图分箱数")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    def resolve_input(raw: str) -> Path:
        path = Path(raw)
        if path.is_absolute():
            return path
        return project_root / raw

    incidents_path = resolve_input(args.incidents)
    env_path = resolve_input(args.env)
    out_path = resolve_input(args.out)

    try:
        df = load_incidents(incidents_path)
    except Exception as err:
        print(f"[ERR] 读取事故数据失败: {err}")
        sys.exit(1)

    try:
        ds = xr.open_dataset(env_path)
    except Exception as err:
        print(f"[ERR] 打开风险场失败: {err}")
        sys.exit(1)

    if "risk_env" not in ds.data_vars:
        print("[ERR] 风险场缺少 risk_env 变量")
        sys.exit(1)

    df = map_to_risk(df, ds)
    density = build_incident_density(df, ds)

    plot_overlay(df, ds, density, out_path, bins=args.hist_bins)

    stats = df.groupby("incident")["risk_env"].agg(["mean", "median", "std", "count"])
    mean_incident = stats.loc[1, "mean"] if 1 in stats.index else float("nan")
    mean_normal = stats.loc[0, "mean"] if 0 in stats.index else float("nan")
    delta = mean_incident - mean_normal if np.isfinite(mean_incident) and np.isfinite(mean_normal) else float("nan")

    print(f"[OK] 图像已保存: {out_path}")
    total_incident = int((df["incident"] == 1).sum())
    total_normal = int((df["incident"] == 0).sum())
    print(f"[STAT] 样本总数: {len(df)} (事故 {total_incident}, 非事故 {total_normal})")
    print(f"[STAT] 事故风险均值: {mean_incident:.3f}")
    print(f"[STAT] 非事故风险均值: {mean_normal:.3f}")
    if np.isfinite(delta):
        print(f"[STAT] 均值差 (事故-非事故): {delta:.3f}")


if __name__ == "__main__":
    main()
