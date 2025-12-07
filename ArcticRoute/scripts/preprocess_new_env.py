"""
预处理 newDATA_RAW 下的海冰 / 风 / 地形数据，裁剪到东北航道区域并标准化输出。
@role: pipeline

用法示例：
python ArcticRoute/scripts/preprocess_new_env.py --north 80 --south 65 --west 0 --east 160
"""
from __future__ import annotations

from pathlib import Path
import argparse
import warnings
from typing import Iterable, Optional, Tuple, List, Dict

import numpy as np
import xarray as xr

try:
    import pandas as pd  # 可选，仅在 NSIDC CSV 场景使用
except Exception:  # pragma: no cover
    pd = None


# ----------------------- 路径 -----------------------

def get_project_root() -> Path:
    """本脚本位于 minimum/ArcticRoute/scripts/ 下，项目根为 scripts 的上上级目录。"""
    return Path(__file__).resolve().parents[2]


# ----------------------- 通用工具函数 -----------------------

_LAT_CANDIDATES = [
    "lat", "latitude", "nav_lat", "LAT", "Latitude", "y"
]
_LON_CANDIDATES = [
    "lon", "longitude", "nav_lon", "LON", "Longitude", "x"
]


def _find_coord_by_attrs(ds: xr.Dataset, axis: str) -> Optional[str]:
    """从坐标 attrs 中根据 units/standard_name 猜测纬度或经度坐标名。"""
    assert axis in ("lat", "lon")
    target_units = {"lat": ["degrees_north"], "lon": ["degrees_east"]}[axis]
    for name, coord in ds.coords.items():
        units = str(coord.attrs.get("units", ""))
        stdn = str(coord.attrs.get("standard_name", ""))
        if any(u in units for u in target_units):
            return name
        if axis == "lat" and stdn.lower() in ("lat", "latitude"):  # 容错
            return name
        if axis == "lon" and stdn.lower() in ("lon", "longitude"):
            return name
    return None


def detect_lat_lon_names(ds: xr.Dataset) -> Tuple[str, str]:
    """
    在 ds 中自动识别纬度/经度坐标名，支持 latitude/lat/Latitude 与 longitude/lon/Longitude 等。
    找不到时抛 ValueError。
    """
    # 1) 先从 attrs 猜
    lat = _find_coord_by_attrs(ds, "lat")
    lon = _find_coord_by_attrs(ds, "lon")

    # 2) 再从候选名匹配坐标/维度
    if lat is None:
        for cand in _LAT_CANDIDATES:
            if cand in ds.coords:
                lat = cand
                break
            if cand in ds.dims:  # 有些数据 lat 不在 coords，只是一个维度
                lat = cand
                break
    if lon is None:
        for cand in _LON_CANDIDATES:
            if cand in ds.coords:
                lon = cand
                break
            if cand in ds.dims:
                lon = cand
                break

    # 3) 失败时在 data_vars 的 coords 上找
    if (lat is None or lon is None) and len(ds.data_vars) > 0:
        any_da = next(iter(ds.data_vars))
        dv = ds[any_da]
        for cand in _LAT_CANDIDATES:
            if cand in dv.coords:
                lat = lat or cand
                break
        for cand in _LON_CANDIDATES:
            if cand in dv.coords:
                lon = lon or cand
                break

    if lat is None or lon is None:
        raise ValueError("无法自动识别经纬度坐标名，请检查数据集 coords/dims")
    return lat, lon


def normalize_lon(lon_vals: np.ndarray) -> np.ndarray:
    """
    将经度统一到 [0, 360) 范围。
    """
    lon = np.asarray(lon_vals)
    out = np.mod(lon, 360.0)
    out[out < 0] += 360.0
    return out


def _ensure_lon_0_360(ds: xr.Dataset, lon_name: str) -> xr.Dataset:
    """将 Dataset 的经度坐标统一到 [0,360)，并按经度排序。"""
    lon = ds[lon_name]
    lon_new = xr.DataArray(normalize_lon(lon.values), dims=lon.dims, coords=lon.coords, attrs=lon.attrs)
    ds2 = ds.assign_coords({lon_name: lon_new})
    try:
        ds2 = ds2.sortby(lon_name)
    except Exception:
        pass
    return ds2


def _slice_lat(ds: xr.Dataset, lat_name: str, south: float, north: float) -> xr.Dataset:
    lat_vals = ds[lat_name].values
    lat_min, lat_max = float(min(south, north)), float(max(south, north))
    ascending = np.all(np.diff(lat_vals) > 0) if lat_vals.ndim == 1 else True
    if lat_vals.ndim != 1:
        # 网格化二维坐标：退化到 where 过滤
        return ds.where((ds[lat_name] >= lat_min) & (ds[lat_name] <= lat_max), drop=True)
    if ascending:
        return ds.sel({lat_name: slice(lat_min, lat_max)})
    else:
        return ds.sel({lat_name: slice(lat_max, lat_min)})


def _slice_lon(ds: xr.Dataset, lon_name: str, west: float, east: float) -> xr.Dataset:
    # 假设已经是 0..360，经度跨越情况也处理
    west0 = (west % 360 + 360) % 360
    east0 = (east % 360 + 360) % 360
    lon_vals = ds[lon_name].values
    if lon_vals.ndim != 1:
        cond = None
        if west0 <= east0:
            cond = (ds[lon_name] >= west0) & (ds[lon_name] <= east0)
        else:  # wrap
            cond = (ds[lon_name] >= west0) | (ds[lon_name] <= east0)
        return ds.where(cond, drop=True)

    if west0 <= east0:
        return ds.sel({lon_name: slice(west0, east0)})
    else:
        # wrap-around: 取两段拼接
        left = ds.sel({lon_name: slice(west0, 360.0)})
        right = ds.sel({lon_name: slice(0.0, east0)})
        try:
            return xr.concat([left, right], dim=lon_name)
        except Exception:
            # 最差退化为 where
            cond = (ds[lon_name] >= west0) | (ds[lon_name] <= east0)
            return ds.where(cond, drop=True)


def clip_to_bbox(ds: xr.Dataset, lat_name: str, lon_name: str,
                 north: float, south: float, west: float, east: float) -> xr.Dataset:
    """
    根据给定 bbox 裁剪 Dataset。注意纬度可能降序，经度统一到 0..360 后处理。
    """
    ds0 = _ensure_lon_0_360(ds, lon_name)
    ds1 = _slice_lat(ds0, lat_name, south, north)
    ds2 = _slice_lon(ds1, lon_name, west, east)
    return ds2


_DEPTH_CANDS = ["depth", "lev", "level", "z", "sigma", "height"]
_TIME_CANDS = ["time", "TIME", "t"]


def _get_time_name(da: xr.DataArray) -> Optional[str]:
    for c in _TIME_CANDS:
        if c in da.dims:
            return c
    # 有时 time 在 coords 不在 dims
    for c in _TIME_CANDS:
        if c in da.coords:
            return c
    return None


def drop_to_2d(ds: xr.Dataset, var_name: str) -> xr.DataArray:
    """
    将 var_name 对 time 等额外维度做简化：
    - 若有 time 维：打印 time[0]/time[-1]，对 time 求 mean；
    - 若有 depth/lev 等：优先取表层（索引 0）；
    - squeeze 后返回只带 (lat, lon) 的 DataArray。
    """
    if var_name not in ds:
        raise KeyError(f"变量 {var_name} 不在数据集中")
    da = ds[var_name]

    # 时间处理
    tname = _get_time_name(da)
    if tname is not None:
        try:
            tcoord = xr.decode_cf(da.to_dataset(name=var_name))[var_name].coords.get(tname, None)
            tcoord = da.coords.get(tname, tcoord)
        except Exception:
            tcoord = da.coords.get(tname, None)
        if tcoord is not None and tcoord.size > 0:
            try:
                t0 = np.asarray(tcoord.values)[0]
                t1 = np.asarray(tcoord.values)[-1]
                print(f"      time_range={np.datetime_as_string(np.array(t0, dtype='datetime64[ns]'), timezone='UTC')}.." \
                      f"{np.datetime_as_string(np.array(t1, dtype='datetime64[ns]'), timezone='UTC')}")
            except Exception:
                print(f"      time_range={str(tcoord.values[0])}..{str(tcoord.values[-1])}")
        da = da.mean(dim=tname, skipna=True)

    # 深度/层处理：优先取第 0 层
    for zname in _DEPTH_CANDS:
        if zname in da.dims:
            da = da.isel({zname: 0})
            break

    # 压缩掉长度为 1 的多余维
    da = da.squeeze(drop=True)

    # 只保留 (lat, lon)
    lat_name, lon_name = detect_lat_lon_names(ds)
    # 若还有其他维度，尝试尽量降到 2D
    other_dims = [d for d in da.dims if d not in (lat_name, lon_name)]
    if other_dims:
        # 若还有 y/x 等别名，尝试重命名
        rename_map = {}
        for d in other_dims:
            if d in _LAT_CANDIDATES:
                rename_map[d] = lat_name
            if d in _LON_CANDIDATES:
                rename_map[d] = lon_name
        if rename_map:
            da = da.rename(rename_map)
        # 再 squeeze 一次
        da = da.squeeze(drop=True)
        other_dims = [d for d in da.dims if d not in (lat_name, lon_name)]
        if other_dims:
            # 如果仍不是 2D，则尝试对剩余维求 mean
            da = da.mean(dim=other_dims, skipna=True)
            da = da.squeeze(drop=True)

    # 确保经度为 0..360
    ds_tmp = da.to_dataset(name=var_name)
    ds_tmp = _ensure_lon_0_360(ds_tmp, detect_lat_lon_names(ds_tmp)[1])
    da = ds_tmp[var_name]

    return da


def _value_stats(da: xr.DataArray) -> Dict[str, float | bool]:
    vals = da.values
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "p5": np.nan, "p95": np.nan, "std": np.nan, "constant_like": True}
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    vmean = float(np.nanmean(vals))
    p5 = float(np.nanpercentile(vals, 5))
    p95 = float(np.nanpercentile(vals, 95))
    std = float(np.nanstd(vals))
    spread = p95 - p5
    constant_like = bool((std < 1e-8) or (spread < 1e-6))
    return {"min": vmin, "max": vmax, "mean": vmean, "p5": p5, "p95": p95, "std": std, "constant_like": constant_like}


def save_da(da: xr.DataArray, out_path: Path, var_name: str, dtype: Optional[str] = None):
    """
    将 DataArray 用合理压缩参数写出 NetCDF。dims 应为 (lat, lon) 或 (y, x) + coords。
    """
    if dtype is not None:
        try:
            da = da.astype(dtype)
        except Exception:
            pass
    encoding = {var_name: {"zlib": True, "complevel": 4}}
    ds_out = da.to_dataset(name=var_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_path, encoding=encoding)
    print(f"      -> saved: {out_path}")


def _print_geo_summary(da: xr.DataArray, tag: str = "SUMMARY"):
    lat_name, lon_name = detect_lat_lon_names(da.to_dataset(name="tmp"))
    latv = da[lat_name].values
    lonv = da[lon_name].values
    try:
        lat_min, lat_max = float(np.nanmin(latv)), float(np.nanmax(latv))
        lon_min, lon_max = float(np.nanmin(lonv)), float(np.nanmax(lonv))
    except Exception:
        lat_min = lat_max = lon_min = lon_max = np.nan
    stats = _value_stats(da)
    print(f"      lat_range={lat_min:.3f}..{lat_max:.3f} lon_range={lon_min:.3f}..{lon_max:.3f}")
    print(f"      value_range={stats['min']:.2f}..{stats['max']:.2f} mean={stats['mean']:.3f} constant_like={stats['constant_like']}")


# ----------------------- 各类数据处理入口 -----------------------

# 扩展候选名（ICE + WIND + GEBCO）
ICE_SIC_CANDIDATES = [
    "sic", "SIC", "siconc", "siconc_mean",
    "seaice_conc", "seaice_conc_cdr", "sea_ice_concentration",
    "ice_conc", "ice_concentration"
]
ICE_SITHICK_CANDIDATES = [
    "sithick", "sithic", "ice_thickness", "sea_ice_thickness", "sithk"
]
ICE_UICE_CANDIDATES = ["uice", "u_ice", "ice_u", "uoice", "u"]
ICE_VICE_CANDIDATES = ["vice", "v_ice", "ice_v", "voice", "v"]

_WIND_U_CANDS = ["u10", "u10n", "uwnd", "u_wind", "u", "eastward_wind"]
_WIND_V_CANDS = ["v10", "v10n", "vwnd", "v_wind", "v", "northward_wind"]
WIND_SPEED_CANDIDATES = ["si10", "wind_speed", "wspd", "wsp", "ff", "FF"]
WAVE_SWH_CANDIDATES = ["VHM0", "VHM0_WW", "swh", "SWH", "hs", "Hm0"]

_GEBCO_CANDS = ["elevation", "z", "depth", "bathymetry", "bathy"]


def _first_present(ds: xr.Dataset, cands: Iterable[str]) -> Optional[str]:
    for c in cands:
        if c in ds.data_vars:
            return c
    # 有些在 coords 略过
    return None


def _clip_and_reduce(ds: xr.Dataset, var_name: str, bbox: dict) -> xr.DataArray:
    lat_name, lon_name = detect_lat_lon_names(ds)
    dsc = clip_to_bbox(ds, lat_name, lon_name, bbox["north"], bbox["south"], bbox["west"], bbox["east"])
    da2d = drop_to_2d(dsc, var_name)
    return da2d


def _concat_mean_safe(das: List[xr.DataArray]) -> xr.DataArray:
    if not das:
        raise ValueError("empty dataarray list")
    if len(das) == 1:
        return das[0]
    # 检查是否可以直接 concat（坐标一致）
    same_shape = all((da.shape == das[0].shape) for da in das[1:])
    same_lat = all(np.array_equal(das[0][detect_lat_lon_names(das[0].to_dataset(name='x'))[0]].values,
                                  da[detect_lat_lon_names(da.to_dataset(name='x'))[0]].values) for da in das[1:])
    same_lon = all(np.array_equal(das[0][detect_lat_lon_names(das[0].to_dataset(name='x'))[1]].values,
                                  da[detect_lat_lon_names(da.to_dataset(name='x'))[1]].values) for da in das[1:])
    if same_shape and same_lat and same_lon:
        try:
            cat = xr.concat(das, dim="stack")
            return cat.mean(dim="stack", skipna=True)
        except Exception:
            pass
    warnings.warn("多文件经纬度网格不一致，跳过平均，使用首个可用切片")
    return das[0]


def pick_var_by_candidates_or_dims(ds: xr.Dataset, candidates: List[str],
                                   lat_name: str, lon_name: str) -> Optional[str]:
    # 先按候选名精确匹配
    for name in candidates:
        if name in ds.data_vars:
            return name
    # fallback：找第一个同时含 lat/lon 且为 float 的变量
    float_like = []
    for name, da in ds.data_vars.items():
        try:
            if lat_name in da.dims and lon_name in da.dims and getattr(da.dtype, 'kind', 'f') in "f":
                float_like.append(name)
        except Exception:
            continue
    if len(float_like) == 1:
        print(f"[PICK] 候选名未命中，自动选用变量: {float_like[0]}")
        return float_like[0]
    elif len(float_like) > 1:
        print(f"[PICK] 候选名未命中，有多个可能变量: {float_like}，暂用第一个。")
        return float_like[0]
    return None


def process_copernicus_ice(raw_dir: Path, out_dir: Path, bbox: dict):
    print("[ICE] 扫描 Copernicus 冰产品: ", raw_dir)
    if not raw_dir.exists():
        print("[ICE] 目录不存在，跳过")
        return

    sic_list: List[xr.DataArray] = []
    sithick_list: List[xr.DataArray] = []
    u_list: List[xr.DataArray] = []
    v_list: List[xr.DataArray] = []

    nc_files = sorted(raw_dir.glob("*.nc"))
    if not nc_files:
        print(f"[ICE] 未在 {raw_dir} 下找到 .nc 文件，跳过 Copernicus 冰产品。")
        return

    for nc_path in nc_files:
        print(f"[ICE] 打开文件: {nc_path}")
        try:
            ds = xr.open_dataset(nc_path)
        except Exception as e:
            print(f"[ICE] 打开失败: {nc_path} err={e}")
            continue
        print("[ICE] data_vars:", list(ds.data_vars))
        lat_name, lon_name = detect_lat_lon_names(ds)

        # 变量识别（宽松 + fallback）
        sic_var = pick_var_by_candidates_or_dims(ds, ICE_SIC_CANDIDATES, lat_name, lon_name)
        sithick_var = pick_var_by_candidates_or_dims(ds, ICE_SITHICK_CANDIDATES, lat_name, lon_name)
        u_var = pick_var_by_candidates_or_dims(ds, ICE_UICE_CANDIDATES, lat_name, lon_name)
        v_var = pick_var_by_candidates_or_dims(ds, ICE_VICE_CANDIDATES, lat_name, lon_name)
        print(f"[ICE] 识别变量: sic={sic_var}, sithick={sithick_var}, uice={u_var}, vice={v_var}")

        try:
            if sic_var is not None:
                shape0 = tuple(ds[sic_var].shape)
                # 按建议：drop_to_2d -> clip
                da2d = drop_to_2d(ds, sic_var)
                dac = clip_to_bbox(da2d.to_dataset(name=sic_var), lat_name, lon_name,
                                   bbox["north"], bbox["south"], bbox["west"], bbox["east"])[sic_var]
                shape1 = tuple(dac.shape)
                print(f"[ICE] file={nc_path.name} var={sic_var} shape={shape0} -> {shape1}")
                _print_geo_summary(dac)
                sic_list.append(dac)
        except Exception as e:
            print(f"[ICE] 处理 sic 失败: {nc_path} err={e}")

        try:
            if sithick_var is not None:
                shape0 = tuple(ds[sithick_var].shape)
                da2d = drop_to_2d(ds, sithick_var)
                dac = clip_to_bbox(da2d.to_dataset(name=sithick_var), lat_name, lon_name,
                                   bbox["north"], bbox["south"], bbox["west"], bbox["east"])[sithick_var]
                shape1 = tuple(dac.shape)
                print(f"[ICE] file={nc_path.name} var={sithick_var} shape={shape0} -> {shape1}")
                _print_geo_summary(dac)
                sithick_list.append(dac)
        except Exception as e:
            print(f"[ICE] 处理 sithick 失败: {nc_path} err={e}")

        try:
            if (u_var is not None) and (v_var is not None):
                shape0u = tuple(ds[u_var].shape)
                shape0v = tuple(ds[v_var].shape)
                dau2d = drop_to_2d(ds, u_var)
                dav2d = drop_to_2d(ds, v_var)
                dau = clip_to_bbox(dau2d.to_dataset(name=u_var), lat_name, lon_name,
                                   bbox["north"], bbox["south"], bbox["west"], bbox["east"])[u_var]
                dav = clip_to_bbox(dav2d.to_dataset(name=v_var), lat_name, lon_name,
                                   bbox["north"], bbox["south"], bbox["west"], bbox["east"])[v_var]
                print(f"[ICE] file={nc_path.name} var=({u_var},{v_var}) shape={shape0u}/{shape0v} -> {tuple(dau.shape)}")
                _print_geo_summary(dau)
                u_list.append(dau)
                v_list.append(dav)
        except Exception as e:
            print(f"[ICE] 处理 uice/vice 失败: {nc_path} err={e}")

        try:
            ds.close()
        except Exception:
            pass

    # 聚合并写出
    if sic_list:
        da_sic = _concat_mean_safe(sic_list)
        save_da(da_sic.astype("float32"), out_dir / "ice_copernicus_sic.nc", var_name="sic", dtype="float32")
    if sithick_list:
        da_sithick = _concat_mean_safe(sithick_list)
        save_da(da_sithick.astype("float32"), out_dir / "ice_copernicus_sithick.nc", var_name="sithick", dtype="float32")
    if u_list and v_list:
        dau = _concat_mean_safe(u_list)
        dav = _concat_mean_safe(v_list)
        # 对齐
        try:
            dau, dav = xr.align(dau, dav, join="exact")
        except Exception:
            pass
        ds_uv = xr.Dataset({"uice": dau.astype("float32"), "vice": dav.astype("float32")})
        enc = {k: {"zlib": True, "complevel": 4} for k in ds_uv.data_vars}
        out_path = out_dir / "ice_copernicus_ice_drift.nc"
        ds_uv.to_netcdf(out_path, encoding=enc)
        print(f"      -> saved: {out_path}")


def _read_nsidc_csv_to_grid(csv_path: Path) -> xr.DataArray:
    if pd is None:
        raise RuntimeError("需要 pandas 以读取 CSV 格式的 NSIDC SIC")
    df = pd.read_csv(csv_path)
    # 列名猜测
    def find_col(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in df.columns:
                return c
            if c.lower() in [x.lower() for x in df.columns]:
                # 大小写不敏感匹配
                for col in df.columns:
                    if col.lower() == c.lower():
                        return col
        return None

    lat_col = find_col(["lat", "latitude", "Lat", "Latitude"]) or "lat"
    lon_col = find_col(["lon", "longitude", "Lon", "Longitude"]) or "lon"
    sic_col = find_col(["sic", "concentration", "ice_concentration", "siconc"]) or "sic"
    # 舍入到固定网格（防止不规则点）
    df = df.dropna(subset=[lat_col, lon_col, sic_col])
    df["lat_r"] = df[lat_col].astype(float).round(3)
    # 统一经度到 0..360
    lon = np.mod(df[lon_col].astype(float).values, 360.0)
    lon[lon < 0] += 360.0
    df["lon_r"] = np.round(lon, 3)
    pvt = df.pivot_table(index="lat_r", columns="lon_r", values=sic_col, aggfunc="mean")
    lat_vals = pvt.index.values
    lon_vals = pvt.columns.values
    da = xr.DataArray(pvt.values, dims=("lat", "lon"), coords={"lat": lat_vals, "lon": lon_vals})
    da.name = "sic_nsidc"
    return da


NSIDC_SIC_CANDIDATES = [
    "sic", "SIC",
    "seaice_conc", "seaice_conc_cdr",
    "goddard_merged_seaice_conc",
    "cdr_seaice_conc",
    "sea_ice_concentration",
    "concentration", "C"
]

def _has_lat_lon_grid(ds: xr.Dataset) -> bool:
    """粗略判断是否为真正的经纬网格（而不是极区投影 x/y 网格）"""
    lat_names = [n for n in ds.coords if "lat" in n.lower()]
    lon_names = [n for n in ds.coords if "lon" in n.lower()]
    if lat_names and lon_names:
        return True
    for da in ds.data_vars.values():
        dims_lower = [d.lower() for d in da.dims]
        if any("lat" in d for d in dims_lower) and any("lon" in d for d in dims_lower):
            return True
    return False


def process_nsidc_sic(raw_dir: Path, out_dir: Path, bbox: dict):
    print("[NSIDC] 扫描 NSIDC SIC: ", raw_dir)
    if not raw_dir.exists():
        print("[NSIDC] 目录不存在，跳过")
        return

    # 仅扫描本目录下 nc/csv
    nc_files = sorted(raw_dir.glob("*.nc"))
    csv_files = sorted(raw_dir.glob("*.csv"))

    if not nc_files and not csv_files:
        print("[NSIDC] 未找到 nc/csv 文件，跳过。")
        return

    all_arrays: List[xr.DataArray] = []

    # 先处理 nc
    for fp in nc_files:
        try:
            ds = xr.open_dataset(fp)
        except Exception as e:
            print(f"[NSIDC] 打开失败: {fp} err={e}")
            continue
        print(f"[NSIDC] 文件: {fp}")
        print("        data_vars:", list(ds.data_vars))

        # 新增：非经纬网格（极区投影等）直接跳过
        if not _has_lat_lon_grid(ds):
            print(f"[NSIDC] 文件 {fp.name} 不是经纬网格（可能是极区投影），暂不支持，跳过。")
            try:
                ds.close()
            except Exception:
                pass
            continue

        try:
            lat_name, lon_name = detect_lat_lon_names(ds)
        except Exception as e:
            print(f"[NSIDC] 无法识别经纬度: {e}")
            continue

        # 先按候选名
        var_name = None
        for name in NSIDC_SIC_CANDIDATES:
            if name in ds.data_vars:
                var_name = name
                break

        # fallback：找含 lat/lon 的 float 变量，排除明显非场变量
        if var_name is None:
            candidates = []
            for name, da in ds.data_vars.items():
                if lat_name in da.dims and lon_name in da.dims and getattr(da.dtype, 'kind', 'f') in 'f':
                    if name.lower() not in ("mask", "stdev", "standard_error", "qc_flag", "crs"):
                        candidates.append(name)
            if len(candidates) == 1:
                var_name = candidates[0]
                print(f"[NSIDC] 自动选用 sic 变量: {var_name}")
            elif len(candidates) > 1:
                var_name = candidates[0]
                print(f"[NSIDC] 自动候选过多 {candidates}，暂用第一个: {var_name}")

        if var_name is None:
            print(f"[NSIDC] 仍未找到 sic 变量，跳过文件: {fp.name}")
            try:
                ds.close()
            except Exception:
                pass
            continue

        try:
            da = drop_to_2d(ds, var_name)
            da_clip = clip_to_bbox(da.to_dataset(name=var_name), lat_name, lon_name, **bbox)[var_name]
            print(
                f"[NSIDC] {fp.name} -> {var_name} "
                f"shape={da_clip.shape} "
                f"lat_range={float(da_clip[lat_name].min())}..{float(da_clip[lat_name].max())} "
                f"lon_range={float(da_clip[lon_name].min())}..{float(da_clip[lon_name].max())} "
                f"value_range={float(da_clip.min())}..{float(da_clip.max())}"
            )
            all_arrays.append(da_clip)
        except Exception as e:
            print(f"[NSIDC] 处理失败: {fp.name} err={e}")
        finally:
            try:
                ds.close()
            except Exception:
                pass

    # CSV 生成网格
    for fp in csv_files:
        try:
            da = _read_nsidc_csv_to_grid(fp)
            ds_tmp = da.to_dataset(name="sic_nsidc")
            lat_name, lon_name = detect_lat_lon_names(ds_tmp)
            ds_tmp = clip_to_bbox(ds_tmp, lat_name, lon_name, **bbox)
            da_clip = ds_tmp["sic_nsidc"]
            print(f"[NSIDC] {fp.name} -> sic_nsidc shape={tuple(da_clip.shape)}")
            all_arrays.append(da_clip)
        except Exception as e:
            print(f"[NSIDC] CSV 读取失败: {fp} err={e}")

    if not all_arrays:
        print("[NSIDC] 没有任何 sic 数组被识别出来或均为不支持网格，结束。")
        return

    # 多文件合并后取 mean
    sic_stack = xr.concat(all_arrays, dim="time_stack")
    sic_mean = sic_stack.mean(dim="time_stack", keep_attrs=True)
    sic_mean.name = "sic_nsidc"

    out_path = out_dir / "nsidc_sic.nc"
    save_da(sic_mean, out_path, "sic_nsidc")
    print("[NSIDC] 汇总输出:", out_path)


def process_gebco_bathymetry(raw_path: Path, out_dir: Path, bbox: dict):
    print("[GEBCO] 处理 GEBCO: ", raw_path)
    if not raw_path.exists():
        print("[GEBCO] 文件不存在，跳过")
        return
    try:
        ds = xr.open_dataset(raw_path)
    except Exception as e:
        print(f"[GEBCO] 打开失败: {e}")
        return

    var = _first_present(ds, _GEBCO_CANDS)
    if var is None:
        # 尝试唯一变量
        if len(ds.data_vars) == 1:
            var = list(ds.data_vars)[0]
        else:
            print("[GEBCO] 未识别到深度变量，跳过")
            return

    lat_name, lon_name = detect_lat_lon_names(ds)
    dsc = clip_to_bbox(ds, lat_name, lon_name, bbox["north"], bbox["south"], bbox["west"], bbox["east"])
    da = drop_to_2d(dsc, var)
    da = da.astype("float32")

    stats = _value_stats(da)
    print(f"[GEBCO] var={var} depth_range={stats['min']:.1f}..{stats['max']:.1f}")
    _print_geo_summary(da)

    # 写出裁剪后的 bathy
    save_da(da, out_dir / "gebco_bathy_clip.nc", var_name=var, dtype="float32")

    # 生成陆地掩膜：GEBCO elevation: 陆地>=0，海洋<0
    land_mask = (da >= 0).astype("uint8").rename("land_mask_gebco")
    save_da(land_mask, out_dir / "land_mask_gebco.nc", var_name="land_mask_gebco")

    try:
        ds.close()
    except Exception:
        pass


def process_wind(raw_path: Path, out_dir: Path, bbox: dict):
    """处理风场/波浪场：
    - 若存在 u/v → 输出 wind_u.nc / wind_v.nc / wind_speed.nc
    - 否则若存在显著波高变量 → 输出 wave_swh.nc
    - 两者都不存在则跳过
    """
    if not raw_path.exists():
        print("[WIND] 文件不存在，跳过。", raw_path)
        return

    print("[WIND] 处理风/浪场: ", raw_path)
    try:
        ds = xr.open_dataset(raw_path)
    except Exception as e:
        print(f"[WIND] 打开失败: {e}")
        return

    print("[WIND] data_vars:", list(ds.data_vars))

    try:
        lat_name, lon_name = detect_lat_lon_names(ds)
    except Exception as exc:
        print(f"[WIND] 无法识别经纬度坐标，跳过。err={exc}")
        try:
            ds.close()
        except Exception:
            pass
        return

    # 1) 先尝试找 u/v 风
    u_name = next((n for n in _WIND_U_CANDS if n in ds.data_vars), None)
    v_name = next((n for n in _WIND_V_CANDS if n in ds.data_vars), None)

    if u_name and v_name:
        print(f"[WIND] 使用 u/v 变量: u={u_name}, v={v_name}")
        u_da = drop_to_2d(ds, u_name)
        v_da = drop_to_2d(ds, v_name)

        u_clip = clip_to_bbox(u_da.to_dataset(name="u10"), lat_name, lon_name, **bbox)["u10"]
        v_clip = clip_to_bbox(v_da.to_dataset(name="v10"), lat_name, lon_name, **bbox)["v10"]

        speed = np.sqrt(u_clip ** 2 + v_clip ** 2)
        speed.name = "wind_speed"

        save_da(u_clip, out_dir / "wind_u.nc", "u10")
        save_da(v_clip, out_dir / "wind_v.nc", "v10")
        save_da(speed, out_dir / "wind_speed.nc", "wind_speed")
        print("[WIND] 已输出 u/v/speed 三个风场文件。")
        try:
            ds.close()
        except Exception:
            pass
        return

    # 2) 没有 u/v，则尝试显著波高（VHM0 等）
    swh_name = next((n for n in WAVE_SWH_CANDIDATES if n in ds.data_vars), None)
    if swh_name is None:
        print("[WIND] 未找到 u/v 风，也未找到显著波高变量，跳过。")
        try:
            ds.close()
        except Exception:
            pass
        return

    print(f"[WAVE] 使用显著波高变量: {swh_name}")
    swh_da = drop_to_2d(ds, swh_name)
    swh_clip = clip_to_bbox(swh_da.to_dataset(name="wave_swh"), lat_name, lon_name, **bbox)["wave_swh"]

    save_da(swh_clip, out_dir / "wave_swh.nc", "wave_swh")
    print(
        "[WAVE] wave_swh.nc 输出完成，"
        f"范围 lat={float(swh_clip[lat_name].min())}..{float(swh_clip[lat_name].max())}, "
        f"lon={float(swh_clip[lon_name].min())}..{float(swh_clip[lon_name].max())}, "
        f"value_range={float(swh_clip.min())}..{float(swh_clip.max())}"
    )

    try:
        ds.close()
    except Exception:
        pass


# ----------------------- main -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--north", type=float, default=80.0)
    parser.add_argument("--south", type=float, default=65.0)
    parser.add_argument("--west", type=float, default=0.0)
    parser.add_argument("--east", type=float, default=160.0)
    args = parser.parse_args()

    root = get_project_root()
    arctic_dir = root / "ArcticRoute"
    raw_root = arctic_dir / "newDATA_RAW"
    out_dir = arctic_dir / "data_processed" / "newenv"
    out_dir.mkdir(parents=True, exist_ok=True)

    bbox = dict(north=args.north, south=args.south, west=args.west, east=args.east)

    print("[NEW_ENV] project_root:", root)
    print("[NEW_ENV] raw_root:", raw_root)
    print("[NEW_ENV] out_dir:", out_dir)
    print("[NEW_ENV] bbox:", bbox)

    try:
        process_copernicus_ice(raw_root / "sic_sithick_uice_vice", out_dir, bbox)
    except Exception as e:
        print("[NEW_ENV] Copernicus 冰处理异常:", e)

    try:
        process_nsidc_sic(raw_root / "sic", out_dir, bbox)
    except Exception as e:
        print("[NEW_ENV] NSIDC 处理异常:", e)

    gebco_path = raw_root / "GEBCO_2025.nc"
    if gebco_path.exists():
        try:
            process_gebco_bathymetry(gebco_path, out_dir, bbox)
        except Exception as e:
            print("[NEW_ENV] GEBCO 处理异常:", e)

    wind_path = raw_root / "wind.nc"
    if wind_path.exists():
        try:
            process_wind(wind_path, out_dir, bbox)
        except Exception as e:
            print("[NEW_ENV] 风场处理异常:", e)


if __name__ == "__main__":
    main()

