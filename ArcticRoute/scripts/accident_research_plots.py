#!/usr/bin/env python3
"""Research plots for accident risk vs background and route distance to hotspots.

@role: analysis
"""

"""
ACC-6 | 事故 vs 环境风险与路线距离关系研究图。

- 对比事故点与背景网格 risk_env 分布
- 计算规划路线各节点到事故热点的距离分布
- 输出图表与简短中文结论
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_INCIDENTS = PROJECT_ROOT / "data_processed" / "incidents_aligned.parquet"
DEFAULT_ENV = PROJECT_ROOT / "data_processed" / "env_clean.nc"
DEFAULT_ACC_STATIC = PROJECT_ROOT / "data_processed" / "accident_density_static.nc"
DEFAULT_ROUTE_REPORT = PROJECT_ROOT / "outputs" / "run_report_acc_s.json"
DEFAULT_ROUTE_GEOJSON = PROJECT_ROOT / "outputs" / "route_acc_s.geojson"
DEFAULT_HIST_PATH = PROJECT_ROOT / "docs" / "accident_risk_hist.png"
DEFAULT_BOX_PATH = PROJECT_ROOT / "docs" / "route_acc_distance_box.png"


def resolve_path(path: Path, search_roots: list[Path]) -> Path:
    if path.is_absolute():
        return path
    for root in search_roots:
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    return (search_roots[0] / path).resolve()


def load_incidents(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.dropna(subset=["tidx", "ii", "jj"])
    df = df.astype({"tidx": "int64", "ii": "int64", "jj": "int64"})
    return df


def load_env(path: Path) -> xr.Dataset:
    return xr.open_dataset(path)


def sample_background(
    env_ds: xr.Dataset,
    tidx: int,
    sample_size: int,
    random_state: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    risk_slice = env_ds["risk_env"].isel(time=tidx).values
    lat_len, lon_len = risk_slice.shape
    total = lat_len * lon_len
    sample_size = min(sample_size, total)
    indices = random_state.choice(total, size=sample_size, replace=False)
    ii = indices // lon_len
    jj = indices % lon_len
    return ii, jj


def gather_risk_samples(
    env_ds: xr.Dataset,
    incidents: pd.DataFrame,
    sample_size: int,
    random_state: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if incidents.empty:
        raise ValueError("无事故点可用于对比分析")

    # 只选在 env 时间范围内的事故
    time_len = env_ds.dims["time"]
    incidents = incidents[incidents["tidx"].between(0, time_len - 1)]
    if incidents.empty:
        raise ValueError("事故点不在 env 时间范围内")

    incidents_sample = incidents.sample(n=min(sample_size, len(incidents)), random_state=random_state)
    accident_risks = []
    for _, row in incidents_sample.iterrows():
        tidx = int(row["tidx"])
        ii = int(row["ii"])
        jj = int(row["jj"])
        val = float(env_ds["risk_env"].isel(time=tidx, latitude=ii, longitude=jj).values)
        accident_risks.append(val)

    # 背景采样使用事故的时间分布
    background_risks = []
    for _, row in incidents_sample.iterrows():
        tidx = int(row["tidx"])
        ii_bg, jj_bg = sample_background(env_ds, tidx, 1, random_state)
        val = float(env_ds["risk_env"].isel(time=tidx, latitude=ii_bg[0], longitude=jj_bg[0]).values)
        background_risks.append(val)

    return np.array(accident_risks, dtype="float32"), np.array(background_risks, dtype="float32")


def detect_hotspots(acc_ds: xr.Dataset, threshold: float = 0.7) -> np.ndarray:
    data = acc_ds["acc_density_static"].values if "acc_density_static" in acc_ds else None
    if data is None:
        # fall back to generic name
        data = acc_ds["accident_density"].values
    mask = data >= threshold
    lat, lon = np.where(mask)
    return np.vstack([lat, lon]).T  # shape (N, 2)


def load_route_geometry(report_path: Path, geojson_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if geojson_path.exists():
        geojson = json.loads(geojson_path.read_text(encoding="utf-8"))
        coords = geojson["features"][0]["geometry"]["coordinates"]
        lon = np.array([pt[0] for pt in coords], dtype="float64")
        lat = np.array([pt[1] for pt in coords], dtype="float64")
        return lat, lon
    # fallback to report path indices with env lat/lon later (requires predictor info)
    data = json.loads(report_path.read_text(encoding="utf-8"))
    raise FileNotFoundError("路线 GeoJSON 不存在，无法绘制距离分布")


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2
    return 2 * radius * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def compute_hotspot_distance_distribution(
    route_lat: np.ndarray,
    route_lon: np.ndarray,
    hotspot_lat: np.ndarray,
    hotspot_lon: np.ndarray,
) -> np.ndarray:
    if hotspot_lat.size == 0:
        return np.array([], dtype="float32")

    distances = []
    for lat_pt, lon_pt in zip(route_lat, route_lon):
        dist = haversine_km(lat_pt, lon_pt, hotspot_lat, hotspot_lon)
        distances.append(np.min(dist))
    return np.array(distances, dtype="float32")


def plot_risk_histogram(acc_risks: np.ndarray, bg_risks: np.ndarray, output_path: Path) -> dict[str, float]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    bins = np.linspace(0, 1, 30)
    plt.hist(bg_risks, bins=bins, alpha=0.6, label="背景网格", color="#1f77b4", density=True)
    plt.hist(acc_risks, bins=bins, alpha=0.6, label="事故点", color="#ff7f0e", density=True)
    plt.xlabel("risk_env")
    plt.ylabel("Probability Density")
    plt.title("Accident vs Background risk_env Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] 保存风险分布图: {output_path}")

    acc_high = np.mean(acc_risks >= 0.75)
    bg_high = np.mean(bg_risks >= 0.75)
    return {"acc_high": float(acc_high), "bg_high": float(bg_high)}


def plot_distance_box(distances: np.ndarray, output_path: Path) -> dict[str, float]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    if distances.size > 0:
        plt.boxplot(distances, vert=True, labels=["Route→Hotspot"])
    else:
        plt.boxplot([0], vert=True, labels=["No Hotspot"], widths=0.5)
    plt.ylabel("Nearest Hotspot Distance (km)")
    plt.title("Route vs Accident Hotspots Distance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[OK] 保存距离箱线图: {output_path}")

    if distances.size == 0:
        return {"median": math.nan, "p90": math.nan}
    return {"median": float(np.median(distances)), "p90": float(np.percentile(distances, 90))}


def summarize(acc_stats: dict, distance_stats: dict) -> str:
    acc_high = acc_stats.get("acc_high", 0.0)
    bg_high = acc_stats.get("bg_high", 0.0)
    uplift = (acc_high - bg_high) * 100
    distance_median = distance_stats.get("median")
    distance_p90 = distance_stats.get("p90")

    if math.isnan(distance_median):
        distance_text = "路线附近暂无显著事故热点"
    else:
        distance_text = f"路线节点与热点的中位距离约 {distance_median:.1f} km，90 分位 {distance_p90:.1f} km"

    return (
        f"事故点落在 risk_env≥0.75 的概率比背景高 {uplift:.1f}%。"
        f" {distance_text}。"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="ACC-6 accident research plots")
    parser.add_argument("--inc-aligned", type=Path, default=DEFAULT_INCIDENTS)
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--acc-static", type=Path, default=DEFAULT_ACC_STATIC)
    parser.add_argument("--route-report", type=Path, default=DEFAULT_ROUTE_REPORT)
    parser.add_argument("--route-geojson", type=Path, default=DEFAULT_ROUTE_GEOJSON)
    parser.add_argument("--hist", type=Path, default=DEFAULT_HIST_PATH)
    parser.add_argument("--box", type=Path, default=DEFAULT_BOX_PATH)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--hotspot-threshold", type=float, default=0.7)
    args = parser.parse_args()

    incidents_path = resolve_path(args.inc_aligned, [PROJECT_ROOT, REPO_ROOT])
    env_path = resolve_path(args.env, [PROJECT_ROOT, REPO_ROOT])
    acc_path = resolve_path(args.acc_static, [PROJECT_ROOT, REPO_ROOT])
    route_report_path = resolve_path(args.route_report, [PROJECT_ROOT, REPO_ROOT])
    route_geojson_path = resolve_path(args.route_geojson, [PROJECT_ROOT, REPO_ROOT])
    hist_path = resolve_path(args.hist, [PROJECT_ROOT, REPO_ROOT])
    box_path = resolve_path(args.box, [PROJECT_ROOT, REPO_ROOT])

    rng = np.random.default_rng(2025)
    incidents_df = load_incidents(incidents_path)
    env_ds = load_env(env_path)

    acc_risks, bg_risks = gather_risk_samples(
        env_ds,
        incidents_df,
        sample_size=args.sample_size,
        random_state=rng,
    )
    acc_stats = plot_risk_histogram(acc_risks, bg_risks, hist_path)

    acc_static_ds = xr.open_dataset(acc_path)
    hotspot_mask = detect_hotspots(acc_static_ds, threshold=args.hotspot_threshold)
    hotspot_lat = hotspot_lon = None
    if hotspot_mask.size > 0:
        lat_vals = env_ds["latitude"].values
        lon_vals = env_ds["longitude"].values
        hotspot_lat = lat_vals[hotspot_mask[:, 0]]
        hotspot_lon = lon_vals[hotspot_mask[:, 1]]
    else:
        hotspot_lat = np.array([])
        hotspot_lon = np.array([])

    route_lat, route_lon = load_route_geometry(route_report_path, route_geojson_path)
    distance_arr = compute_hotspot_distance_distribution(route_lat, route_lon, hotspot_lat, hotspot_lon)
    distance_stats = plot_distance_box(distance_arr, box_path)

    summary = summarize(acc_stats, distance_stats)
    print(f"[结论] {summary}")

    env_ds.close()
    acc_static_ds.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
