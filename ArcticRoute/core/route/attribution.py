from __future__ import annotations

# Phase H | Segment attribution core
# - sample_field_along_path
# - segment_contributions
#
# 依赖：xarray/numpy/matplotlib（仅当绘图时）
# 复用：
# - haversine_m / integrate helpers 来自 ArcticRoute.core.route.metrics  # REUSE
# - 图层加载逻辑复用 ArcticRoute.core.route.scan._open_da/_load_layers       # REUSE

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

from ArcticRoute.core.route.metrics import haversine_m  # REUSE
from ArcticRoute.core.route.scan import _open_da as _open_da_scan  # REUSE
from ArcticRoute.core.route.scan import _load_layers as _load_layers_scan  # REUSE

REPO_ROOT = Path(__file__).resolve().parents[3]
ROUTES_DIR = REPO_ROOT / "ArcticRoute" / "data_processed" / "routes"
REPORT_DIR = REPO_ROOT / "ArcticRoute" / "reports" / "d_stage" / "phaseH"


@dataclass
class SegmentContribution:
    i: int
    start: Tuple[float, float]  # (lon, lat)
    end: Tuple[float, float]
    distance_m: float
    values: Dict[str, float]  # 原始场取值均值（端点平均）
    contrib: Dict[str, float]  # 该段贡献（含权重）


def _grid_coords(da: "xr.DataArray") -> Tuple[np.ndarray, np.ndarray]:
    # 支持 1D/2D lat/lon 坐标；若为2D则取首列/首行近似为1D轴
    lat = None; lon = None
    for latn in ("lat", "latitude"):
        if latn in da.coords:
            v = np.asarray(da.coords[latn].values)
            lat = v[:, 0] if v.ndim == 2 else v
            break
    for lonn in ("lon", "longitude"):
        if lonn in da.coords:
            v = np.asarray(da.coords[lonn].values)
            lon = v[0, :] if v.ndim == 2 else v
            break
    # 回退：若找不到 lat/lon，但存在 y/x 1D 坐标，则用其值近似
    if lat is None and "y" in da.coords and da.coords["y"].ndim == 1:
        lat = np.asarray(da.coords["y"].values)
    if lon is None and "x" in da.coords and da.coords["x"].ndim == 1:
        lon = np.asarray(da.coords["x"].values)
    if lat is None:
        raise KeyError("lat/latitude not found in dataarray")
    if lon is None:
        raise KeyError("lon/longitude not found in dataarray")
    return lat, lon


def _slice_2d(da: "xr.DataArray") -> np.ndarray:
    arr = da
    if "time" in arr.dims and arr.sizes.get("time", 0) > 0:
        arr = arr.isel(time=0)
    A = np.asarray(arr.values, dtype=float)
    if A.ndim == 2:
        return A
    A2 = np.squeeze(A)
    if A2.ndim > 2:
        axes = tuple(range(0, A2.ndim - 2))
        A2 = A2.mean(axis=axes)
    return A2


def sample_field_along_path(field: "xr.DataArray", path_lonlat: Sequence[Tuple[float, float]]) -> np.ndarray:
    """对路径上每个点进行近邻采样，返回值数组，长度等于路径点数。

    采样策略：简单网格近邻（保持与 metrics.integrate_field_along_path 一致）。  # REUSE
    """
    if xr is None:
        raise RuntimeError("xarray required")
    try:
        lat, lon = _grid_coords(field)
        A = _slice_2d(field)
        vals: List[float] = []
        for (lon1, lat1) in path_lonlat:
            iy = int(np.clip(np.searchsorted(lat, lat1) - 1, 0, len(lat) - 1))
            ix = int(np.clip(np.searchsorted(lon, lon1) - 1, 0, len(lon) - 1))
            v = float(np.nan_to_num(A[iy, ix], nan=0.0))
            vals.append(v)
        return np.asarray(vals, dtype=float)
    except Exception:
        # 回退：无法获取经纬度坐标时返回全零，保证流程可继续
        return np.zeros((len(path_lonlat),), dtype=float)


def segment_contributions(path_lonlat: Sequence[Tuple[float, float]], fields: Dict[str, "xr.DataArray"], *, weights: Optional[Dict[str, float]] = None) -> List[SegmentContribution]:
    """按段计算贡献：ΔJ_s,i = w_s * 0.5*(f_s(pi)+f_s(pi+1)) * d(pi,pi+1)

    参数：
    - path_lonlat: [(lon,lat), ...]
    - fields: 字典，如 {"risk": da_risk, "prior": da_prior, "interact": da_interact}
    - weights: 对应权重，例如 {"risk": beta, "prior": w_p, "interact": w_c, "distance": w_d}

    返回：每段的 SegmentContribution，包含各源的原始取值均值与贡献值。
    """
    W = dict(weights or {})
    # 默认距离权重 1.0
    if "distance" not in W:
        W["distance"] = 1.0
    # 预采样所有场在各路径点处的值
    samples: Dict[str, np.ndarray] = {}
    for k, da in fields.items():
        if da is None:
            continue
        samples[k] = sample_field_along_path(da, path_lonlat)
    out: List[SegmentContribution] = []
    for i in range(len(path_lonlat) - 1):
        lon1, lat1 = path_lonlat[i]
        lon2, lat2 = path_lonlat[i + 1]
        dist_m = haversine_m(lat1, lon1, lat2, lon2) * float(W.get("distance", 1.0))
        vals: Dict[str, float] = {}
        contrib: Dict[str, float] = {}
        for k, arr in samples.items():
            v = 0.5 * (float(arr[i]) + float(arr[i + 1]))
            vals[k] = v
            w = float(W.get(k, 1.0 if k == "risk" else 0.0))
            contrib[k] = v * dist_m * w
        # 显式加入距离项贡献，便于与规划器目标对齐（w_d * 距离）
        if float(W.get("distance", 1.0)) != 0.0:
            contrib["distance"] = float(W.get("distance", 1.0)) * dist_m
        out.append(SegmentContribution(i=i, start=(lon1, lat1), end=(lon2, lat2), distance_m=dist_m, values=vals, contrib=contrib))
    return out


def _read_geojson_features(path: Path) -> Dict[str, List[List[Tuple[float, float]]]]:
    """读取 FeatureCollection，将不同 mode 的 LineString 按模态分组；返回 {mode: [path1, path2,...]}。
    若文件是单条 LineString（无 FeatureCollection），则默认 mode="sea"。
    识别 Point 的 transfer_penalty，总和返回在 special 键中：{"transfer": [penalty,...]}。
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, List[List[Tuple[float, float]]]] = {}
    specials: Dict[str, List[float]] = {"transfer": []}
    feats = data.get("features") if isinstance(data, dict) else None
    if isinstance(feats, list) and feats:
        for f in feats:
            g = f.get("geometry") or {}
            props = f.get("properties") or {}
            gtype = g.get("type")
            if gtype == "LineString":
                mode = str(props.get("mode", "sea")).lower()
                coords = g.get("coordinates") or []
                path_xy = [(float(lon), float(lat)) for lon, lat in coords]
                out.setdefault(mode, []).append(path_xy)
            elif gtype == "Point":
                # 读取换乘点惩罚
                pen = props.get("transfer_penalty")
                if pen is not None:
                    try:
                        specials["transfer"].append(float(pen))
                    except Exception:
                        pass
        out.setdefault("_special", []).append([])  # 占位
        out["_special_transfer"] = [[(0.0,0.0) for _ in specials["transfer"]]] if specials["transfer"] else []
        out["_special_values"] = [[p for p in specials["transfer"]]] if specials["transfer"] else []
        return out
    # 非 FeatureCollection 情况：尝试 geometry 顶层
    if isinstance(data, dict) and data.get("type") == "LineString":
        coords = data.get("coordinates") or []
        out.setdefault("sea", []).append([(float(lon), float(lat)) for lon, lat in coords])
        return out
    # 尝试简单坐标数组
    if isinstance(data, dict) and data.get("type") == "FeatureCollection":
        raise ValueError("Empty FeatureCollection")
    if isinstance(data, list):
        out.setdefault("sea", []).append([(float(lon), float(lat)) for lon, lat in data])
        return out
    raise ValueError("No LineString found in geojson")


def _git_sha() -> str:
    import subprocess
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL, text=True).strip()
        return sha
    except Exception:
        return "unknown"


def _hash_obj(obj: object) -> str:
    import hashlib
    m = hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    return m.hexdigest()[:16]


def explain_route(route_geojson: Path, ym: str, out_dir: Optional[Path] = None, *, weights: Optional[Dict[str, float]] = None) -> Dict[str, str]:
    """主流程：加载图层 → 读取路线（可多模态） → 计算分段贡献 → 写 JSON + PNG + meta

    返回：{"json": json_path, "png": png_path}
    """
    if xr is None:
        raise RuntimeError("xarray required")
    out_dir = Path(out_dir or REPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 载入层（与 Phase G 保持一致）
    layers = _load_layers_scan(ym, risk_source="fused")  # REUSE
    risk = layers.get("risk")
    prior = layers.get("prior")
    interact = layers.get("interact")

    # 解析多模态要素
    paths_by_mode = _read_geojson_features(route_geojson)

    rows: List[Dict[str, float]] = []
    totals: Dict[str, float] = {"distance_m": 0.0}
    seg_index = 0

    # 定义每模态使用的字段与权重键
    def mode_weights(mode: str) -> Dict[str, float]:
        W = dict(weights or {})
        out: Dict[str, float] = {
            "distance": float(W.get(f"distance_{mode}", W.get("distance", 1.0)))
        }
        if mode == "sea":
            out.update({
                "risk": float(W.get("risk_sea", W.get("risk", 1.0))),
                "prior": float(W.get("prior", 0.0)),
                "interact": float(W.get("interact", 0.0)),
            })
        else:
            # rail/road 默认只有距离项（若未来有 rail/road 风险层，可在此扩展）
            out.update({
                "risk": float(W.get(f"risk_{mode}", 0.0)),
            })
        return out

    for mode, paths in list(paths_by_mode.items()):
        if mode.startswith("_special"):
            continue
        for pts in paths:
            # 构建该模态的字段
            fields: Dict[str, xr.DataArray] = {}
            if mode == "sea":
                if risk is not None:
                    fields["risk"] = risk
                if prior is not None:
                    fields["prior"] = prior
                if interact is not None:
                    fields["interact"] = interact
            # 计算该段贡献
            segs = segment_contributions(pts, fields, weights=mode_weights(mode))
            for s in segs:
                row = {"i": seg_index, "mode": mode, "distance_m": s.distance_m}
                for k, v in s.values.items():
                    row[f"v_{mode}_{k}"] = float(v)
                for k, v in s.contrib.items():
                    key = f"c_{mode}_{k}" if k in ("risk","prior","interact","distance") else f"c_{k}"
                    row[key] = float(v)
                    totals[key] = totals.get(key, 0.0) + float(v)
                totals["distance_m"] += float(s.distance_m)
                rows.append(row)
                seg_index += 1

    # 处理换乘惩罚
    special_vals = paths_by_mode.get("_special_values") or []
    if special_vals:
        tr = special_vals[0]
        pen_w = float((weights or {}).get("transfer", 1.0))
        for p in tr:
            c = float(p) * pen_w
            key = "c_transfer"
            totals[key] = totals.get(key, 0.0) + c
            rows.append({"i": seg_index, "mode": "transfer", key: c, "distance_m": 0.0})
            seg_index += 1

    run_id = time.strftime("%Y%m%dT%H%M%S")
    json_path = out_dir / f"route_attr_{ym}_{route_geojson.stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"ym": ym, "route": str(route_geojson), "weights": dict(weights or {}), "segments": rows, "totals": totals}, f, ensure_ascii=False, indent=2)
    # meta
    with open(str(json_path) + ".meta.json", "w", encoding="utf-8") as f:
        meta = {
            "logical_id": json_path.name,
            "inputs": [str(route_geojson)],
            "run_id": run_id,
            "git_sha": _git_sha(),
            "config_hash": _hash_obj(weights or {}),
        }
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 生成堆叠条形图（按模态聚合主要项）
    try:
        import matplotlib.pyplot as plt  # type: ignore
        modes = sorted({r.get("mode","sea") for r in rows})
        keys = []
        for m in modes:
            for k in ("risk","prior","interact","distance"):
                keys.append(f"c_{m}_{k}")
        if any("c_transfer" in r for r in rows):
            keys.append("c_transfer")
        x = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(11, 4))
        bottom = np.zeros(len(rows))
        import itertools
        color_map = {}
        palette = itertools.cycle(["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2"]) 
        for key in keys:
            y = np.array([float(r.get(key, 0.0)) for r in rows])
            if key not in color_map:
                color_map[key] = next(palette)
            ax.bar(x, y, bottom=bottom, label=key, color=color_map[key])
            bottom = bottom + y
        ax.set_title(f"Multimodal route attribution {ym}")
        ax.set_xlabel("segment")
        ax.set_ylabel("contribution")
        ax.legend(ncol=4, fontsize=8)
        png_path = out_dir / f"route_attr_{ym}_{route_geojson.stem}.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        # meta
        with open(str(png_path) + ".meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "logical_id": png_path.name,
                "inputs": [str(json_path)],
                "run_id": run_id,
                "git_sha": _git_sha(),
                "config_hash": _hash_obj(weights or {}),
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return {"json": str(json_path), "png": str(out_dir / f"route_attr_{ym}_{route_geojson.stem}.png")}


__all__ = ["sample_field_along_path", "segment_contributions", "explain_route"]

