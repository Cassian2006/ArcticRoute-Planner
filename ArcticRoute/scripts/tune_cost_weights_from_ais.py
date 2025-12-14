# -*- coding: utf-8 -*-
"""
根据历史 AIS 航线离线调参，搜索多组 cost 权重，使规划路线更接近历史航线。

用法示例：
  python ArcticRoute/scripts/tune_cost_weights_from_ais.py --ym 202412 \
      --routes-dir ArcticRoute/data_processed/hist_routes \
      --out ArcticRoute/config/weight_profiles_from_ais.yaml \
      --max-combos 60 --topk 3

输入：
  - 样本航线（CSV/NC）：文件名任意，列/变量含经纬度（lat/lon/Latitude/Longitude）
    - CSV: 列名自动识别
    - NC: 变量/坐标名自动识别（latitude/lat, longitude/lon），优先 1D

搜索空间（内置，可通过 --random-sample N 随机采样以避免组合爆炸）：
  ice:      [0.3, 0.6, 1.0]
  accident: [0.0, 0.3, 0.6]
  interact: [0.0, 0.3, 0.6]
  wave:     [0.0, 0.3, 0.6]
  prior:    [0.0, 0.3, 0.6]

输出：
  - YAML: profiles: { ais_default/top1/top2/... }

注：不修改任何核心代码，调用 planner_service.run_planning_pipeline，并通过 kwargs 传入 w_interact / use_newenv_for_cost / w_wave 等。
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from ArcticRoute.core import planner_service as ps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROUTES_DIR = PROJECT_ROOT / "ArcticRoute" / "data_processed" / "hist_routes"
DEFAULT_OUT = PROJECT_ROOT / "ArcticRoute" / "config" / "weight_profiles_from_ais.yaml"


# -------------------------
# 地理工具
# -------------------------

def haversine(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    R = 6371.0
    rad = np.pi / 180.0
    dlat = (lat2 - lat1) * rad
    dlon = (lon2 - lon1) * rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * rad) * np.cos(lat2 * rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def _cumulative_dist_km(path_lonlat: np.ndarray) -> np.ndarray:
    if path_lonlat.shape[0] <= 1:
        return np.array([0.0])
    lat = path_lonlat[:, 0]
    lon = path_lonlat[:, 1]
    d = haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s


def resample_polyline(latlon: np.ndarray, n: int = 200) -> np.ndarray:
    """
    将路径按弧长均匀重采样到 n 个点（线性插值在经纬度空间，近似足够）。
    输入 latlon: (N,2) [lat, lon]
    返回 (n,2)
    """
    if latlon is None or len(latlon) == 0:
        return np.zeros((n, 2), dtype=float)
    latlon = np.asarray(latlon, dtype=float)
    if latlon.ndim != 2 or latlon.shape[1] != 2:
        raise ValueError("latlon 形状应为 (N,2)")
    s = _cumulative_dist_km(latlon)
    if s[-1] <= 0:
        return np.repeat(latlon[:1, :], n, axis=0)
    t = np.linspace(0.0, s[-1], n)
    lat = np.interp(t, s, latlon[:, 0])
    lon = np.interp(t, s, latlon[:, 1])
    return np.stack([lat, lon], axis=1)


def route_distance(path1_latlon: List[List[float]] | np.ndarray, path2_latlon: List[List[float]] | np.ndarray, n: int = 200) -> float:
    p1 = np.asarray(path1_latlon, dtype=float)
    p2 = np.asarray(path2_latlon, dtype=float)
    if p1.size == 0 or p2.size == 0:
        return float("inf")
    p1r = resample_polyline(p1, n=n)
    p2r = resample_polyline(p2, n=n)
    d = haversine(p1r[:, 1], p1r[:, 0], p2r[:, 1], p2r[:, 0])
    return float(np.nanmean(d))


# -------------------------
# 数据加载
# -------------------------

def _detect_lat_lon_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cand_lat = ["lat", "latitude", "LAT", "Latitude"]
    cand_lon = ["lon", "longitude", "LON", "Longitude"]
    lat_col = next((c for c in cand_lat if c in df.columns), None)
    lon_col = next((c for c in cand_lon if c in df.columns), None)
    if lat_col is None or lon_col is None:
        raise ValueError(f"未找到经纬度列: {list(df.columns)}")
    return lat_col, lon_col


def _load_route_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    lat_col, lon_col = _detect_lat_lon_columns(df)
    lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy()
    lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy()
    m = np.isfinite(lat) & np.isfinite(lon)
    lat = lat[m]
    lon = lon[m]
    return np.stack([lat, lon], axis=1)


def _detect_lat_lon_in_ds(ds: xr.Dataset) -> Tuple[str, str]:
    cand_lat = ["lat", "latitude", "LAT", "Latitude"]
    cand_lon = ["lon", "longitude", "LON", "Longitude"]
    latn = next((c for c in cand_lat if c in ds.variables or c in ds.coords), None)
    lonn = next((c for c in cand_lon if c in ds.variables or c in ds.coords), None)
    if latn is None or lonn is None:
        raise ValueError(f"未找到经纬度坐标: vars={list(ds.data_vars)} coords={list(ds.coords)}")
    return latn, lonn


def _load_route_nc(path: Path) -> np.ndarray:
    with xr.open_dataset(path) as ds:
        latn, lonn = _detect_lat_lon_in_ds(ds)
        lat = np.asarray(ds[latn].values).reshape(-1)
        lon = np.asarray(ds[lonn].values).reshape(-1)
        m = np.isfinite(lat) & np.isfinite(lon)
        return np.stack([lat[m], lon[m]], axis=1)


def load_ais_route_latlon(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".csv":
        return _load_route_csv(path)
    if path.suffix.lower() in (".nc", ".netcdf"):
        return _load_route_nc(path)
    raise ValueError(f"不支持的文件类型: {path}")


def load_ais_route_to_path_ij(env: ps.EnvironmentContext, route_path: Path) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    latlon = load_ais_route_latlon(route_path)
    if latlon.size == 0:
        raise ValueError(f"{route_path} 路径为空")
    # 取首末点作为起终点
    s_lat, s_lon = float(latlon[0, 0]), float(latlon[0, 1])
    g_lat, g_lon = float(latlon[-1, 0]), float(latlon[-1, 1])
    s_ij = ps.latlon_to_ij(env, s_lat, s_lon)
    g_ij = ps.latlon_to_ij(env, g_lat, g_lon)
    if s_ij is None or g_ij is None:
        raise ValueError("无法将起止点映射到网格，请检查经纬度范围与 env 网格")
    return latlon, s_ij, g_ij


# -------------------------
# 规划与打分
# -------------------------

def run_planner_with_weights(ym: str, start_ij: Tuple[int, int], goal_ij: Tuple[int, int], weight: Dict[str, float]) -> Tuple[ps.EnvironmentContext, ps.RouteResult]:
    w_ice = float(weight.get("ice", 1.0))
    w_accident = float(weight.get("accident", 0.0))
    w_interact = float(weight.get("interact", 0.0))
    w_wave = float(weight.get("wave", 0.0))
    prior_w = float(weight.get("prior", 0.0))
    use_newenv_for_cost = bool(w_wave > 0.0)

    env_ctx, route_result = ps.run_planning_pipeline(
        ym=ym,
        start_ij=start_ij,
        goal_ij=goal_ij,
        w_ice=w_ice,
        w_accident=w_accident,
        prior_weight=prior_w,
        allow_diagonal=True,
        heuristic="euclidean",
        eco_enabled=False,
        profile_name="balanced",
        w_interact=w_interact,
        use_newenv_for_cost=use_newenv_for_cost,
        w_wave=w_wave,
    )
    return env_ctx, route_result


def grid_search_weights(ym: str, route_files: List[Path], candidate_space: Dict[str, List[float]], max_combos: int = 60, topk: int = 3, seed: int = 42) -> List[Tuple[Dict[str, float], float]]:
    random.seed(seed)
    # 生成候选组合
    keys = list(candidate_space.keys())
    grids = list(itertools.product(*[candidate_space[k] for k in keys]))
    combos = [dict(zip(keys, vals)) for vals in grids]
    if len(combos) > max_combos:
        combos = random.sample(combos, k=max_combos)

    # 预加载环境（用于起终点投射）
    # 使用默认权重，不影响后续 run_planner
    env_ctx = ps.load_environment(ym=ym)

    # 将样本路径投射到网格，取起止点
    samples: List[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]] = []
    for rp in route_files:
        try:
            latlon, s_ij, g_ij = load_ais_route_to_path_ij(env_ctx, rp)
            samples.append((latlon, s_ij, g_ij))
            print(f"[SAMPLE] {rp.name}: start={s_ij} goal={g_ij} pts={len(latlon)}")
        except Exception as e:
            print(f"[SAMPLE] 跳过 {rp.name}: {e}")

    if not samples:
        raise RuntimeError("没有可用的样本航线")

    results: List[Tuple[Dict[str, float], float]] = []
    for idx, w in enumerate(combos):
        scores = []
        for latlon, s_ij, g_ij in samples:
            try:
                _, rr = run_planner_with_weights(ym, s_ij, g_ij, w)
                if not rr.reachable or not rr.path_lonlat:
                    scores.append(1e9)
                    continue
                dist = route_distance(latlon, rr.path_lonlat, n=200)
                scores.append(dist)
            except Exception as e:
                print(f"[EVAL] 组合{idx}报错：{e}")
                scores.append(1e9)
        score = float(np.nanmean(scores)) if scores else 1e9
        results.append((w, score))
        print(f"[EVAL] {idx+1}/{len(combos)} weight={w} score_km={score:.2f}")

    # 选出最优 topk
    results.sort(key=lambda x: x[1])
    return results[:topk]


# -------------------------
# 主程序
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ym", required=True, help="YYYYMM")
    ap.add_argument("--routes-dir", default=str(DEFAULT_ROUTES_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--max-combos", type=int, default=60)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--random-seed", type=int, default=42)
    args = ap.parse_args()

    ym = str(args.ym)
    routes_dir = Path(args.routes_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 收集样本路径文件
    route_files = sorted(list(routes_dir.glob("*.csv")) + list(routes_dir.glob("*.nc")))
    if not route_files:
        print(f"[TUNE] 未在 {routes_dir} 找到样本航线（*.csv/*.nc），退出")
        return

    candidate_space = {
        "ice": [0.3, 0.6, 1.0],
        "accident": [0.0, 0.3, 0.6],
        "interact": [0.0, 0.3, 0.6],
        "wave": [0.0, 0.3, 0.6],
        "prior": [0.0, 0.3, 0.6],
    }

    top = grid_search_weights(
        ym=ym,
        route_files=route_files,
        candidate_space=candidate_space,
        max_combos=int(args.max_combos),
        topk=int(args.topk),
        seed=int(args.random_seed),
    )

    if not top:
        print("[TUNE] 未获得有效结果")
        return

    # 写出 YAML
    profiles = {}
    base_name = f"ais_{ym}"
    for i, (w, sc) in enumerate(top, 1):
        name = f"{base_name}_top{i}" if i > 1 else f"{base_name}_default"
        profiles[name] = {
            "ice": float(w.get("ice", 1.0)),
            "accident": float(w.get("accident", 0.0)),
            "interact": float(w.get("interact", 0.0)),
            "wave": float(w.get("wave", 0.0)),
            "prior": float(w.get("prior", 0.0)),
            "score_km": float(sc),
        }

    obj = {"profiles": profiles}
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)
    print(f"[TUNE] 写出 {out_path}，共 {len(profiles)} 个 profile")


if __name__ == "__main__":
    main()

