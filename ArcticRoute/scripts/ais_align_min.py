#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal AIS alignment routine to map tracks onto the model grid.

@role: pipeline
"""

"""
AIS 航迹与风险场对齐脚本（最小示例）。

功能：
- 读取 data_raw/ais/ 下的所有 CSV（mmsi,time_utc,lat,lon,sog,cog）
- 数据清洗、按船舶重采样到 10 分钟频率
- 将轨迹映射到 env_clean.nc 的风险栅格，取最近时间的 risk_env
- 导出 data_processed/ais_aligned.parquet
- 输出 docs/ais_vs_risk_t0.png（叠加轨迹与 t=0 风险图）
- 打印统计信息（样本数、轨迹条数、风险 NaN 占比）
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "Noto Sans CJK SC"]
plt.rcParams["axes.unicode_minus"] = False

COLUMN_ALIASES = {
    "mmsi": ["mmsi", "MMSI"],
    "time_utc": ["time_utc", "timestamp", "time", "message_stamp", "time_stamp"],
    "lat": ["lat", "latitude"],
    "lon": ["lon", "longitude"],
    "sog": ["sog", "speed_over_ground", "sog_knot"],
    "cog": ["cog", "course_over_ground"],
}
REQUIRED_COLS = ["mmsi", "time_utc", "lat", "lon"]
OPTIONAL_COLS = ["sog", "cog"]


def list_csv_files(ais_dir: Path) -> list[Path]:
    """列出 AIS 目录下的全部 CSV 文件。"""
    files = sorted(ais_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"未在目录 {ais_dir} 找到任何 CSV 文件")
    return files


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将列名映射到统一命名，缺失的可选列会补充为空值。"""
    lower_map = {col: col.strip().lower() for col in df.columns}
    df = df.rename(columns=lower_map)

    rename_pairs = {}
    for target, candidates in COLUMN_ALIASES.items():
        for name in candidates:
            key = name.strip().lower()
            if key in df.columns:
                rename_pairs[key] = target
                break
    df = df.rename(columns=rename_pairs)

    missing_required = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_required:
        raise ValueError(f"缺少必要列: {missing_required}")

    for col in OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df[REQUIRED_COLS + OPTIONAL_COLS]


def load_and_clean(files: list[Path]) -> pd.DataFrame:
    """读取并清洗 AIS 原始数据。"""
    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            iterator = None
            if path.stat().st_size > 200 * 1024 * 1024:
                iterator = pd.read_csv(path, chunksize=200_000, low_memory=False)
            else:
                iterator = [pd.read_csv(path, low_memory=False)]
        except Exception as err:
            raise ValueError(f"读取 {path} 失败: {err}") from err

        for chunk in iterator:
            try:
                chunk = _standardize_columns(chunk)
            except ValueError as err:
                raise ValueError(f"{path} {err}") from err

            chunk["mmsi"] = chunk["mmsi"].astype(str).str.strip()
            chunk["time_utc"] = pd.to_datetime(chunk["time_utc"], errors="coerce", dayfirst=True)
            chunk["lat"] = pd.to_numeric(chunk["lat"], errors="coerce")
            chunk["lon"] = pd.to_numeric(chunk["lon"], errors="coerce")
            chunk["sog"] = pd.to_numeric(chunk["sog"], errors="coerce")
            chunk["cog"] = pd.to_numeric(chunk["cog"], errors="coerce")

            # 仅保留北极附近的轨迹点以降低数据量
            chunk = chunk[chunk["lat"].between(60, 90) & chunk["lon"].between(-180, 180)]

            if chunk.empty:
                continue
            frames.append(chunk)

    if not frames:
        raise ValueError("所有 AIS 文件在清洗后均为空")

    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["mmsi", "time_utc", "lat", "lon"])
    data = data[(data["lat"].between(-90, 90)) & (data["lon"].between(-180, 180))]
    data["mmsi"] = data["mmsi"].astype(str)
    data = data.sort_values(["mmsi", "time_utc"])
    return data


def resample_tracks(df: pd.DataFrame, freq: str = "10min") -> pd.DataFrame:
    """按船舶重采样到指定频率。"""

    pieces: list[pd.DataFrame] = []
    for mmsi_value, group in df.groupby("mmsi"):
        group = group.sort_values("time_utc").drop_duplicates("time_utc")
        group = group.set_index("time_utc")
        resampled = group.resample(freq).nearest(limit=1)
        resampled["mmsi"] = str(mmsi_value)
        pieces.append(resampled.reset_index())
    if not pieces:
        return df.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def attach_risk(df: pd.DataFrame, ds: xr.Dataset) -> pd.DataFrame:
    """在 DataFrame 上添加风险值与匹配后的网格坐标。"""
    if "risk_env" not in ds.data_vars:
        raise KeyError("env_clean.nc 缺少 risk_env 变量")
    risk_da = ds["risk_env"]

    times = xr.DataArray(df["time_utc"].values.astype("datetime64[ns]"), dims="points")
    lat_da = xr.DataArray(df["lat"].values.astype("float32"), dims="points")
    lon_da = xr.DataArray(df["lon"].values.astype("float32"), dims="points")

    selected = risk_da.sel(
        time=times,
        latitude=lat_da,
        longitude=lon_da,
        method="nearest",
    )

    df = df.copy()
    df["risk_env"] = selected.values.astype("float32")
    df["risk_time"] = pd.to_datetime(selected.coords["time"].values)
    df["grid_lat"] = selected.coords["latitude"].values.astype("float32")
    df["grid_lon"] = selected.coords["longitude"].values.astype("float32")
    return df


def plot_overlay(ds: xr.Dataset, df: pd.DataFrame, out_png: Path) -> None:
    """绘制风险场（t=0）与 AIS 轨迹叠加图。"""
    risk_t0 = ds["risk_env"].isel(time=0)
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    mesh = ax.pcolormesh(lon, lat, risk_t0, shading="auto", cmap="viridis")
    plt.colorbar(mesh, ax=ax, label="risk_env (t=0)")

    for mmsi, g in df.groupby("mmsi"):
        ax.plot(g["lon"], g["lat"], marker="o", markersize=3, linewidth=1, label=f"MMSI {mmsi}")

    ax.set_title("AIS 航迹与风险场（t=0）")
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="AIS 航迹与风险栅格对齐脚本")
    parser.add_argument("--ais-dir", default="data_raw/ais", help="AIS 原始 CSV 目录")
    parser.add_argument("--nc", default="data_processed/env_clean.nc", help="风险场 NetCDF 文件")
    parser.add_argument("--out", default="data_processed/ais_aligned.parquet", help="对齐结果输出 Parquet 路径")
    parser.add_argument("--fig", default="docs/ais_vs_risk_t0.png", help="风险叠加图输出路径")
    parser.add_argument("--freq", default="10min", help="重采样时间分辨率")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    def resolve_input(raw: str) -> Path:
        path = Path(raw)
        if path.is_absolute():
            return path
        alt = project_root / raw
        if alt.exists():
            return alt
        return path

    def resolve_output(raw: str) -> Path:
        path = Path(raw)
        if path.is_absolute():
            return path
        return project_root / raw

    ais_dir = resolve_input(args.ais_dir)
    nc_path = resolve_input(args.nc)
    out_path = resolve_output(args.out)
    fig_path = resolve_output(args.fig)

    if not ais_dir.exists():
        print(f"[ERR] AIS 目录不存在: {ais_dir}")
        sys.exit(1)
    if not nc_path.exists():
        print(f"[ERR] 风险场文件不存在: {nc_path}")
        sys.exit(1)

    try:
        files = list_csv_files(ais_dir)
    except FileNotFoundError as err:
        print(f"[ERR] {err}")
        sys.exit(1)

    print(f"[INFO] 读取 {len(files)} 个 AIS 文件")
    df_raw = load_and_clean(files)
    print(f"[INFO] 清洗后样本数: {len(df_raw)}")

    df_resampled = resample_tracks(df_raw, freq=args.freq)
    print(f"[INFO] 重采样后样本数: {len(df_resampled)}")

    try:
        ds = xr.open_dataset(nc_path)
    except Exception as err:
        print(f"[ERR] 打开风险场失败: {err}")
        sys.exit(1)

    df_aligned = attach_risk(df_resampled, ds)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df_aligned.to_parquet(out_path, index=False)
    except Exception as err:
        print(f"[ERR] 写入 Parquet 失败: {err}")
        sys.exit(1)

    plot_overlay(ds, df_resampled, fig_path)
    ds.close()

    total_samples = len(df_aligned)
    vessel_count = df_aligned["mmsi"].nunique()
    nan_ratio = float(df_aligned["risk_env"].isna().mean())

    print(f"[OK] 对齐结果已保存: {out_path}")
    print(f"[OK] 风险叠加图已保存: {fig_path}")
    print(f"[STAT] 样本数: {total_samples}")
    print(f"[STAT] 轨迹条数（唯一样本数）: {vessel_count}")
    print(f"[STAT] risk_env NaN 占比: {nan_ratio:.4f}")


if __name__ == "__main__":
    main()
