#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Derive corridor probability grid from aligned AIS tracks (DBSCAN + rasterize).

@role: pipeline
"""

"""
根据 AIS 对齐数据提取航线走廊概率场。

流程：
1. 读取 data_processed/ais_aligned.parquet；
2. 使用 DBSCAN（球面距离近似）聚类航迹点；
3. 将聚类密度映射到 env_clean.nc 相同网格，生成 corridor_prob.nc；
4. 输出统计信息（样本数、有效簇、走廊覆盖率）。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

try:
    from sklearn.cluster import DBSCAN
except Exception as err:  # pragma: no cover
    DBSCAN = None
    _IMPORT_ERR = err
else:
    _IMPORT_ERR = None


EARTH_RADIUS_M = 6_371_000.0


def load_parquet(path: Path) -> pd.DataFrame:
    """读取 AIS 对齐结果。"""
    if not path.exists():
        raise FileNotFoundError(f"AIS 对齐文件不存在: {path}")
    df = pd.read_parquet(path)
    required_cols = {"lat", "lon", "time_utc", "risk_time", "grid_lat", "grid_lon", "mmsi"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"缺少必要字段: {sorted(missing)}")
    df = df.dropna(subset=["lat", "lon"])
    if df.empty:
        raise ValueError("数据为空，无法生成走廊")
    return df


def run_dbscan(df: pd.DataFrame, eps_km: float, min_samples: int) -> np.ndarray:
    """使用 DBSCAN（haversine 距离）聚类。"""
    if DBSCAN is None:  # pragma: no cover
        raise RuntimeError(f"缺少 scikit-learn，导入错误: {_IMPORT_ERR}")

    coords = np.radians(df[["lat", "lon"]].to_numpy(dtype="float64"))
    eps_rad = eps_km * 1000.0 / EARTH_RADIUS_M

    model = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine", algorithm="ball_tree")
    labels = model.fit_predict(coords)
    return labels


def build_corridor_grid(df: pd.DataFrame, labels: np.ndarray, env_ds: xr.Dataset) -> xr.Dataset:
    """将聚类后的样本映射到环境网格，生成 corridor_prob。"""
    time_coord = env_ds["time"].values
    lat_coord = env_ds["latitude"].values
    lon_coord = env_ds["longitude"].values

    lat_map = {round(float(v), 4): idx for idx, v in enumerate(lat_coord)}
    lon_map = {round(float(v), 4): idx for idx, v in enumerate(lon_coord)}

    # 初始计数
    counts = np.zeros((len(time_coord), len(lat_coord), len(lon_coord)), dtype="float32")

    valid_mask = labels >= 0
    if not np.any(valid_mask):
        corridor_prob = xr.DataArray(
            counts,
            dims=("time", "latitude", "longitude"),
            coords={"time": env_ds["time"], "latitude": env_ds["latitude"], "longitude": env_ds["longitude"]},
        )
        return xr.Dataset({"corridor_prob": corridor_prob})

    df_valid = df.loc[valid_mask].copy()
    df_valid["cluster"] = labels[valid_mask]

    # 将 risk_time 映射到最近环境时间索引
    df_valid["risk_time"] = pd.to_datetime(df_valid["risk_time"])
    time_array = time_coord.astype("datetime64[ns]")

    def nearest_time_index(ts: pd.Timestamp) -> int:
        diffs = np.abs(time_array - ts.to_datetime64())
        return int(diffs.argmin())

    for _, row in df_valid.iterrows():
        t_idx = nearest_time_index(row["risk_time"])
        lat_idx = lat_map.get(round(float(row["grid_lat"]), 4))
        lon_idx = lon_map.get(round(float(row["grid_lon"]), 4))
        if lat_idx is None or lon_idx is None:
            continue
        counts[t_idx, lat_idx, lon_idx] += 1.0

    max_count = counts.max()
    if max_count > 0:
        prob = (counts / max_count).astype("float32")
    else:
        prob = counts

    corridor_prob = xr.DataArray(
        prob,
        dims=("time", "latitude", "longitude"),
        coords={"time": env_ds["time"], "latitude": env_ds["latitude"], "longitude": env_ds["longitude"]},
    )
    corridor_prob.attrs["description"] = "AIS 走廊概率（按聚类密度归一化）"
    return xr.Dataset({"corridor_prob": corridor_prob})


def save_corridor(ds: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {"corridor_prob": {"zlib": True, "complevel": 4}}
    ds.to_netcdf(path, encoding=encoding)


def main():
    parser = argparse.ArgumentParser(description="根据 AIS 数据生成航线走廊概率场")
    parser.add_argument("--ais", default="data_processed/ais_aligned.parquet", help="AIS 对齐 parquet 路径")
    parser.add_argument("--env", default="data_processed/env_clean.nc", help="环境风险文件")
    parser.add_argument("--out", default="data_processed/corridor_prob.nc", help="输出 NetCDF 路径")
    parser.add_argument("--eps-km", type=float, default=25.0, help="DBSCAN 半径（公里）")
    parser.add_argument("--min-samples", type=int, default=3, help="DBSCAN 最小样本数")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    def resolve_input(raw: str) -> Path:
        p = Path(raw)
        return p if p.is_absolute() else project_root / raw

    ais_path = resolve_input(args.ais)
    env_path = resolve_input(args.env)
    out_path = resolve_input(args.out)

    try:
        df = load_parquet(ais_path)
    except Exception as err:
        print(f"[ERR] 读取 AIS 数据失败: {err}")
        sys.exit(1)

    try:
        labels = run_dbscan(df, eps_km=args.eps_km, min_samples=args.min_samples)
    except Exception as err:
        print(f"[ERR] DBSCAN 聚类失败: {err}")
        sys.exit(1)

    try:
        env_ds = xr.open_dataset(env_path)
    except Exception as err:
        print(f"[ERR] 打开环境文件失败: {err}")
        sys.exit(1)

    corridor_ds = build_corridor_grid(df, labels, env_ds)
    save_corridor(corridor_ds, out_path)
    env_ds.close()

    total_samples = len(df)
    valid_samples = int((labels >= 0).sum())
    cluster_count = len(set(labels[labels >= 0]))
    coverage_ratio = 0.0
    if valid_samples > 0:
        coverage_ratio = valid_samples / total_samples

    print(f"[OK] 走廊概率文件: {out_path}")
    print(f"[STAT] 样本总数: {total_samples}")
    print(f"[STAT] 有效簇数量: {cluster_count}")
    print(f"[STAT] 被识别为走廊的样本数: {valid_samples} ({coverage_ratio:.2%})")


if __name__ == "__main__":
    main()
