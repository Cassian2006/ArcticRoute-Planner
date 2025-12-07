# -*- coding: utf-8 -*-
from __future__ import annotations
"""
ArcticRoute/scripts/clean_prior_routes.py

按更严格的规则清洗历史主航线，并构建与环境网格对齐的 corridor_prob.nc。

目标：
- 显著降低“历史主航线路径落陆比例”（目标 <1%）
- 生成的 corridor_prob.nc 无 NaN，且非近似常量（有正常的 min/max/spread）

数据定位（兼容多路径）：
- 历史航线输入：优先使用 ArcticRoute/data_processed/prior/prior_routes_raw.geojson；
  若不存在，则回退到 ArcticRoute/data_processed/prior/prior_routes_clean.geojson；
  若仍不存在，再尝试 prior_centerlines/prior_*.geojson（保持兼容）。
- 陆地掩膜：ArcticRoute/data_processed/newenv/land_mask_gebco.nc （变量名包含 land/mask 的变量，语义：1=land, 0=ocean）
- 参考网格：优先 ArcticRoute/data_processed/newenv/ice_copernicus_sic.nc；
  若不存在则使用 ArcticRoute/data_processed/env/env_clean.nc
- 输出：
  - 清洗后历史航线：ArcticRoute/data_processed/prior/prior_routes_clean.geojson
  - 走廊概率场：ArcticRoute/data_processed/corridor_prob.nc（与自检路径一致）

日志：
- 清洗前后路线数量
- 抽样 2000 点统计：on_land / ocean / oob 比例
- corridor 构建：total_hits、shape、min/max/spread
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import geopandas as gpd  # type: ignore
except Exception:  # pragma: no cover
    gpd = None  # type: ignore

try:
    from shapely.geometry import LineString, MultiLineString, Point  # type: ignore
except Exception:  # pragma: no cover
    LineString = None  # type: ignore
    MultiLineString = None  # type: ignore
    Point = None  # type: no cover

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

logger = logging.getLogger("PRIOR_CLEAN")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "ArcticRoute" / "data_processed"
PRIOR_DIR = DATA_DIR / "prior"
NEWENV_DIR = DATA_DIR / "newenv"
ENV_DIR = DATA_DIR / "env"

DEFAULT_OUT_PRIOR = PRIOR_DIR / "prior_routes_clean.geojson"
DEFAULT_OUT_CORRIDOR = DATA_DIR / "corridor_prob.nc"  # 自检检查的路径


# -----------------------------
# 工具函数：坐标与索引
# -----------------------------

def _detect_lat_lon_names(obj: xr.Dataset | xr.DataArray) -> Tuple[str, str]:
    cand_lat = ("latitude", "lat", "y")
    cand_lon = ("longitude", "lon", "x")
    # 统一获取可查询集合
    coords = set(getattr(obj, "coords", {}).keys())
    dims_attr = getattr(obj, "dims", ())
    if isinstance(dims_attr, dict):
        dims = set(dims_attr.keys())
    elif isinstance(dims_attr, (tuple, list)):
        dims = set(dims_attr)
    else:
        dims = set()
    vars_set = set(getattr(obj, "variables", {}).keys()) if hasattr(obj, "variables") else set()
    for a in cand_lat:
        if a in coords or a in dims or a in vars_set:
            latn = a
            break
    else:
        raise KeyError("未找到 latitude 维度/坐标")
    for b in cand_lon:
        if b in coords or b in dims or b in vars_set:
            lonn = b
            break
    else:
        raise KeyError("未找到 longitude 维度/坐标")
    return latn, lonn


def _as_1d_coords(ds: xr.Dataset | xr.DataArray, latn: str, lonn: str) -> Tuple[np.ndarray, np.ndarray]:
    has_vars = hasattr(ds, "variables")
    coords_keys = set(getattr(ds, "coords", {}).keys())
    # 取纬度坐标
    if latn in coords_keys or (has_vars and latn in ds.variables):
        lat = np.asarray((ds[latn].values if hasattr(ds[latn], "values") else ds[latn]))
    else:
        # ds.dims 对 Dataset 是映射，对 DataArray 是元组
        dims = getattr(ds, "dims", None)
        if isinstance(dims, dict):
            # 无法直接从维度获取坐标值，只能构造索引
            lat = np.arange(int(dims.get(latn, 0)), dtype=float)
        else:
            lat = np.asarray(dims if dims is not None else [])
    # 取经度坐标
    if lonn in coords_keys or (has_vars and lonn in ds.variables):
        lon = np.asarray((ds[lonn].values if hasattr(ds[lonn], "values") else ds[lonn]))
    else:
        dims = getattr(ds, "dims", None)
        if isinstance(dims, dict):
            lon = np.arange(int(dims.get(lonn, 0)), dtype=float)
        else:
            lon = np.asarray(dims if dims is not None else [])
    # 若为 2D 网格，退化为取唯一值
    if lat.ndim > 1:
        lat = np.unique(lat)
    if lon.ndim > 1:
        lon = np.unique(lon)
    return lat.astype(float), lon.astype(float)


def _nearest_index(axis: np.ndarray, value: float) -> int:
    """支持升/降序数组的最近邻索引。"""
    if axis.size == 0:
        return 0
    asc = axis[0] <= axis[-1]
    if not asc:
        # 转为升序索引再映射回去
        idx = np.searchsorted(axis[::-1], value)
        idx = len(axis) - 1 - np.clip(idx, 0, len(axis) - 1)
    else:
        idx = np.searchsorted(axis, value)
        idx = np.clip(idx - 1, 0, len(axis) - 1)
    return int(idx)


# -----------------------------
# 陆地掩膜与海域判断
# -----------------------------

def load_land_mask_is_ocean() -> Tuple[Callable[[float, float], bool], xr.Dataset, str, str]:
    """返回 (is_ocean(lon,lat), land_mask_ds, latn, lonn)。
    land_mask: 变量名包含 land/mask 的变量；语义 1=land, 0=ocean。
    """
    if xr is None:
        raise RuntimeError("xarray 不可用")
    lm_path = NEWENV_DIR / "land_mask_gebco.nc"
    if not lm_path.exists():
        raise FileNotFoundError(f"缺失陆地掩膜: {lm_path}")
    ds = xr.open_dataset(lm_path)
    # 选择变量
    var = None
    for k in list(ds.data_vars):
        nm = k.lower()
        if "land" in nm or "mask" in nm:
            var = ds[k]
            break
    if var is None and ds.data_vars:
        # 回退：取首个变量并二值化
        var = next(iter(ds.data_vars.values()))
        var = (var > 0).astype("int8")
    latn, lonn = _detect_lat_lon_names(var)
    lat, lon = _as_1d_coords(var, latn, lonn)

    def is_ocean(lon_deg: float, lat_deg: float) -> bool:
        i = _nearest_index(lat, float(lat_deg))
        j = _nearest_index(lon, float(lon_deg))
        try:
            v = float(var.values[i, j])
        except Exception:
            v = float(var.sel({latn: lat[i], lonn: lon[j]}, method="nearest").values)
        # 按 1=land,0=ocean
        return v <= 0.5

    return is_ocean, ds, latn, lonn


# -----------------------------
# 读取与清洗历史航线
# -----------------------------

def _find_prior_inputs() -> List[Path]:
    cand: List[Path] = []
    p_raw = PRIOR_DIR / "prior_routes_raw.geojson"
    p_clean = PRIOR_DIR / "prior_routes_clean.geojson"
    if p_raw.exists():
        cand.append(p_raw)
    elif p_clean.exists():
        cand.append(p_clean)
    # 兼容旧中心线
    for p in [
        PRIOR_DIR / "centerlines" / "prior_centerlines_202412.geojson",
        PRIOR_DIR / "prior_centerlines_all_wgs84.geojson",
    ]:
        if p.exists():
            cand.append(p)
    if not cand and PRIOR_DIR.exists():
        cand.extend(sorted(PRIOR_DIR.glob("prior_*.geojson")))
    return cand


def _load_routes_geoms(paths: Sequence[Path]) -> List[LineString | MultiLineString]:
    geoms: List[LineString | MultiLineString] = []
    if not paths:
        return geoms
    if gpd is None:
        raise RuntimeError("需要 geopandas 以读取/写出 GeoJSON")

    for p in paths:
        try:
            gdf = gpd.read_file(p)
            # 统一到 WGS84
            try:
                if gdf.crs and gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs(epsg=4326)
            except Exception:
                pass
            for g in gdf.geometry:
                if g is None or g.is_empty:
                    continue
                if isinstance(g, (LineString, MultiLineString)):
                    geoms.append(g)
        except Exception as e:
            logger.warning(f"读取 {p.name} 失败: {e}")
    return geoms


def is_in_bbox(lon_deg: float, lat_deg: float, bbox: Dict[str, float]) -> bool:
    return (
        float(bbox["west"]) <= float(lon_deg) <= float(bbox["east"]) and
        float(bbox["south"]) <= float(lat_deg) <= float(bbox["north"]) 
    )


def densify_line(line: LineString, min_points: int = 200) -> List[Point]:
    """按均匀比例在 [0,1] 线性插值采样点，数量与线长成比例并有下限。"""
    try:
        # 粗略以度长近似，适度提高采样密度
        n = max(int(max(line.length, 1.0) * 20), min_points)
    except Exception:
        n = min_points
    ts = np.linspace(0.0, 1.0, n)
    return [line.interpolate(float(t), normalized=True) for t in ts]


def clean_routes(in_path_list: Sequence[Path], out_path: Path, land_mask_fn: Callable[[float, float], bool], bbox: Dict[str, float]) -> Tuple[gpd.GeoDataFrame, Dict[str, float]]:
    geoms_in = _load_routes_geoms(in_path_list)
    logger.info(f"[PRIOR_CLEAN] 输入路线文件 {len(in_path_list)} 个，几何条数 {len(geoms_in)}")

    cleaned_geoms: List[LineString | MultiLineString] = []
    sampled_flags: List[str] = []

    for geom in geoms_in:
        lines: List[LineString] = []
        if isinstance(geom, LineString):
            lines = [geom]
        elif isinstance(geom, MultiLineString):
            lines = list(geom.geoms)
        if not lines:
            continue

        kept_segments: List[LineString] = []
        for line in lines:
            pts = densify_line(line)
            flags: List[str] = []
            for p in pts:
                lon_deg, lat_deg = float(p.x), float(p.y)
                if not is_in_bbox(lon_deg, lat_deg, bbox):
                    flags.append("oob")
                    continue
                ocean = land_mask_fn(lon_deg, lat_deg)
                flags.append("ocean" if ocean else "land")
            if not flags:
                continue
            land_ratio = flags.count("land") / len(flags)
            oob_ratio = flags.count("oob") / len(flags)
            # 过滤掉绝大多数落陆或域外的线段
            if land_ratio > 0.2 or oob_ratio > 0.2:
                continue
            kept_segments.append(line)
            # 收集抽样标记用于总体统计
            sampled_flags.extend(flags)

        if not kept_segments:
            continue
        new_geom: LineString | MultiLineString
        new_geom = kept_segments[0] if len(kept_segments) == 1 else MultiLineString(kept_segments)
        cleaned_geoms.append(new_geom)

    # 输出 GeoJSON
    if gpd is None:
        raise RuntimeError("需要 geopandas 以写出 GeoJSON")

    if not cleaned_geoms:
        logger.warning("[PRIOR_CLEAN] WARNING: no segments survived cleaning, writing empty GeoJSON")
        gdf_out = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    else:
        gdf_out = gpd.GeoDataFrame(geometry=cleaned_geoms, crs="EPSG:4326")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_out.to_file(out_path, driver="GeoJSON")
    logger.info(f"[PRIOR_CLEAN] wrote {len(gdf_out)} routes to {out_path}")

    # 统计：从 sampled_flags（已包含清洗保留段的采样）中抽 2000 点做比例
    if sampled_flags:
        # 若数量不足 2000，则直接使用全部；否则随机子样本
        rng = np.random.default_rng(20251126)
        if len(sampled_flags) > 2000:
            idx = rng.choice(len(sampled_flags), size=2000, replace=False)
            subs = [sampled_flags[int(i)] for i in idx]
        else:
            subs = sampled_flags
        n = len(subs)
        land = subs.count("land")
        ocean = subs.count("ocean")
        oob = subs.count("oob")
        logger.info(f"[PRIOR_CLEAN] sample={n}, on_land={land/n*100:.2f}(%), ocean={ocean/n*100:.2f}(%), oob={oob/n*100:.2f}(%)")
    else:
        logger.info("[PRIOR_CLEAN] sample=0, 无有效抽样（可能所有段被过滤）")

    stats = {
        "num_input_files": int(len(in_path_list)),
        "num_input_geoms": int(len(geoms_in)),
        "num_cleaned_geoms": int(len(gdf_out)),
    }
    return gdf_out, stats


# -----------------------------
# corridor 构建
# -----------------------------

def pick_env_ref() -> Tuple[Path, xr.Dataset, str, str, Dict[str, float]]:
    if xr is None:
        raise RuntimeError("xarray 不可用")
    env_candidates = [NEWENV_DIR / "ice_copernicus_sic.nc", ENV_DIR / "env_clean.nc"]
    for p in env_candidates:
        if p.exists():
            ds = xr.open_dataset(p)
            latn, lonn = _detect_lat_lon_names(ds)
            lat, lon = _as_1d_coords(ds, latn, lonn)
            bbox = {
                "north": float(np.nanmax(lat)),
                "south": float(np.nanmin(lat)),
                "west": float(np.nanmin(lon)),
                "east": float(np.nanmax(lon)),
            }
            # 若 bbox 异常，回退到固定域
            if not np.isfinite(list(bbox.values())).all if isinstance(bbox, dict) else False:
                bbox = {"north": 80.0, "south": 65.0, "west": 0.0, "east": 160.0}
            return p, ds, latn, lonn, bbox
    # 最终回退：固定 bbox
    return env_candidates[-1], xr.Dataset(), "latitude", "longitude", {"north": 80.0, "south": 65.0, "west": 0.0, "east": 160.0}


def sample_points_on_line(line: LineString, min_points: int = 80) -> List[Point]:
    return densify_line(line, min_points=min_points)


def build_corridor_from_routes(routes_path: Path, env_ref_path: Path, out_path: Path, use_land_mask_zero: bool = True, smooth: str = "none") -> Tuple[xr.DataArray, Dict[str, float]]:
    if xr is None or gpd is None:
        raise RuntimeError("需要 xarray 和 geopandas")

    # 读取路线
    gdf = gpd.read_file(routes_path)
    try:
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
    except Exception:
        pass
    lines: List[LineString] = []
    for g in gdf.geometry:
        if g is None or g.is_empty:
            continue
        if isinstance(g, LineString):
            lines.append(g)
        elif isinstance(g, MultiLineString):
            lines.extend(list(g.geoms))

    # 参考网格
    ds = xr.open_dataset(env_ref_path)
    latn, lonn = _detect_lat_lon_names(ds)
    lat, lon = _as_1d_coords(ds, latn, lonn)
    H, W = len(lat), len(lon)
    counts = np.zeros((H, W), dtype=np.int32)

    bbox = {"north": float(np.nanmax(lat)), "south": float(np.nanmin(lat)), "west": float(np.nanmin(lon)), "east": float(np.nanmax(lon))}

    for ln in lines:
        pts = sample_points_on_line(ln, min_points=120)
        for p in pts:
            lo, la = float(p.x), float(p.y)
            if not is_in_bbox(lo, la, bbox):
                continue
            i = _nearest_index(lat, la)
            j = _nearest_index(lon, lo)
            counts[i, j] += 1

    hits_da = xr.DataArray(counts, coords={latn: lat, lonn: lon}, dims=(latn, lonn))

    # 可选：对陆地方格清零
    if use_land_mask_zero:
        try:
            lm_ds = xr.open_dataset(NEWENV_DIR / "land_mask_gebco.nc")
            lm_var = None
            for k in list(lm_ds.data_vars):
                nm = k.lower()
                if "land" in nm or "mask" in nm:
                    lm_var = lm_ds[k]
                    break
            if lm_var is None and lm_ds.data_vars:
                lm_var = next(iter(lm_ds.data_vars.values()))
                lm_var = (lm_var > 0).astype("int8")
            # 识别 land_mask 的坐标名，并对齐到 env 的 (latn,lonn)
            lm_latn, lm_lonn = _detect_lat_lon_names(lm_var)
            lm_on_grid = lm_var.interp({lm_latn: lat, lm_lonn: lon}, method="nearest")
            # 重命名到 env 的维度名，避免 xarray 对齐导致 4D 笛卡尔积
            ren = {}
            if lm_latn != latn:
                ren[lm_latn] = latn
            if lm_lonn != lonn:
                ren[lm_lonn] = lonn
            if ren:
                lm_on_grid = lm_on_grid.rename(ren)
            lm_on_grid = lm_on_grid.transpose(latn, lonn)
            hits_da = hits_da.where(lm_on_grid == 0, 0)
        except Exception as e:
            logger.warning(f"[PRIOR_CORRIDOR] land_mask 清零失败，跳过：{e}")

    total_hits = float(hits_da.sum().item())
    if total_hits > 0:
        prob = (hits_da / total_hits).astype("float32")
    else:
        logger.warning("[PRIOR_CORRIDOR] WARNING: total_hits=0, writing all-zero corridor_prob")
        prob = (hits_da * 0).astype("float32")

    prob = prob.fillna(0.0).clip(0.0, None)

    # 可选平滑（不引入硬依赖）：uniform3/gauss3
    try:
        from scipy.ndimage import uniform_filter, gaussian_filter  # type: ignore
    except Exception:
        uniform_filter = None  # type: ignore
        gaussian_filter = None  # type: ignore
    sm = (smooth or "none").lower()
    if sm != "none":
        arr = np.asarray(prob.values, dtype=float)
        try:
            if sm == "uniform3" and uniform_filter is not None:
                arr = uniform_filter(arr, size=3, mode="nearest")
            elif sm == "gauss3" and gaussian_filter is not None:
                arr = gaussian_filter(arr, sigma=1.0, mode="nearest")
            else:
                logger.warning(f"[PRIOR_CORRIDOR] smooth={sm} 未找到对应实现或 scipy 未安装，跳过平滑")
            # 重新归一化到 [0,1]
            vmin = float(np.nanmin(arr)) if np.isfinite(arr).any() else 0.0
            vmax = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0
            if np.isfinite(vmax) and (vmax > vmin):
                arr = (arr - vmin) / (vmax - vmin)
            prob = xr.DataArray(arr.astype("float32"), coords=prob.coords, dims=prob.dims, name="corridor_prob")
        except Exception as e:
            logger.warning(f"[PRIOR_CORRIDOR] 平滑失败，保持原始分布: {e}")

    # 统计
    vals = np.asarray(prob.values, dtype=float)
    finite = np.isfinite(vals)
    st = {
        "shape": (H, W),
        "min": float(np.nanmin(vals[finite])) if finite.any() else float("nan"),
        "max": float(np.nanmax(vals[finite])) if finite.any() else float("nan"),
        "mean": float(np.nanmean(vals[finite])) if finite.any() else float("nan"),
        "spread": float((np.nanpercentile(vals[finite], 95) - np.nanpercentile(vals[finite], 5))) if finite.any() else float("nan"),
        "total_hits": total_hits,
    }

    # 写出
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out = prob.to_dataset(name="corridor_prob")
    ds_out["corridor_prob"].attrs.update({
        "description": "Historical route corridor probability" if (smooth or "none") == "none" else "Historical route corridor probability (smoothed)",
        "smooth": str(smooth or "none"),
        "total_hits": total_hits,
        "min": st["min"],
        "max": st["max"],
        "spread": st["spread"],
    })
    ds_out.to_netcdf(out_path, encoding={"corridor_prob": {"zlib": True, "complevel": 4}})

    logger.info(f"[PRIOR_CORRIDOR] wrote corridor_prob to {out_path}, total_hits={total_hits:.0f}, shape={H}x{W}, min={st['min']:.6f}, max={st['max']:.6f}, spread={st['spread']:.6f}, smooth={smooth}")

    return prob, st


def main() -> int:
    parser = argparse.ArgumentParser(description="强化历史主航线清洗与 corridor 构建")
    parser.add_argument("--out-prior", default=str(DEFAULT_OUT_PRIOR))
    parser.add_argument("--out-corridor", default=str(DEFAULT_OUT_CORRIDOR))
    parser.add_argument("--env-ref", default=None, help="参考网格 nc（默认优先使用 newenv/ice_copernicus_sic.nc，其次 env/env_clean.nc）")
    parser.add_argument("--bbox", default=None, help='可选自定义 bbox，如 "{\"north\":80,\"south\":65,\"west\":0,\"east\":160}"')
    parser.add_argument("--smooth", type=str, default="none", choices=["none", "uniform3", "gauss3"], help="平滑 corridor 概率场方式：none / uniform3 / gauss3，默认 none")
    args = parser.parse_args()

    if xr is None or gpd is None:
        logger.error("需要 xarray 与 geopandas")
        return 1

    # is_ocean
    is_ocean, lm_ds, lm_latn, lm_lonn = load_land_mask_is_ocean()

    # env 参考与 bbox
    if args.env_ref:
        env_ref_path = Path(args.env_ref)
        if not env_ref_path.exists():
            logger.error(f"env_ref 不存在: {env_ref_path}")
            return 1
        ds_env = xr.open_dataset(env_ref_path)
        latn, lonn = _detect_lat_lon_names(ds_env)
        lat, lon = _as_1d_coords(ds_env, latn, lonn)
        bbox_env = {"north": float(np.nanmax(lat)), "south": float(np.nanmin(lat)), "west": float(np.nanmin(lon)), "east": float(np.nanmax(lon))}
    else:
        env_ref_path, ds_env, latn, lonn, bbox_env = pick_env_ref()

    if args.bbox:
        try:
            bbox = json.loads(args.bbox)
        except Exception as e:
            logger.warning(f"解析 bbox 失败，使用 env bbox：{e}")
            bbox = bbox_env
    else:
        # 默认使用 env 提供的 bbox（若异常则回退固定域）
        bbox = bbox_env or {"north": 80.0, "south": 65.0, "west": 0.0, "east": 160.0}

    # 输入/输出路径
    out_prior = Path(args.out_prior)
    out_corridor = Path(args.out_corridor)

    in_paths = _find_prior_inputs()
    if not in_paths:
        logger.error("未找到任何 prior 路线输入文件")
        return 1

    # 清洗
    gdf_out, _ = clean_routes(in_paths, out_prior, land_mask_fn=is_ocean, bbox=bbox)

    # corridor 构建
    try:
        build_corridor_from_routes(out_prior, env_ref_path, out_corridor, use_land_mask_zero=True, smooth=str(args.smooth or "none"))
    except Exception as e:
        logger.error(f"构建 corridor 失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
