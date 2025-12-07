import os
import sys
import json
import argparse
import warnings
from typing import List, Tuple, Dict

import numpy as np
import xarray as xr
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# （开发期可选）静音 xarray 相关 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module=r"xarray.*")


def compute_chunks(ds: xr.Dataset) -> dict:
    """统一分块策略：time=12，其它维=256，且不超过实际长度。"""
    chunks: Dict[str, int] = {}
    for dim, size in ds.sizes.items():
        if dim.lower() == "time":
            chunks[dim] = int(min(12, max(1, size)))
        else:
            chunks[dim] = int(min(256, max(1, size)))
    return chunks


def open_ds(path: str) -> xr.Dataset:
    """
    Open dataset with xarray and dask chunks. Supports .nc or zarr-like via engine inference.
    使用 compute_chunks 统一分块策略。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    # 先无分块打开一次以获取尺寸
    tmp = xr.open_dataset(path, chunks={}, decode_times=True)
    chunks = compute_chunks(tmp)
    tmp.close()
    # 使用分块重新打开
    ds = xr.open_dataset(path, chunks=chunks, decode_times=True)
    return ds


def _get_lon_name(ds: xr.Dataset) -> str | None:
    for name in ["lon", "longitude", "x"]:
        if name in ds.coords or name in ds:
            return name
    # 2D lon
    for name in ds.data_vars:
        if name.lower() in ("lon", "longitude"):
            return name
    return None


def _get_lat_name(ds: xr.Dataset) -> str | None:
    for name in ["lat", "latitude", "y"]:
        if name in ds.coords or name in ds:
            return name
    for name in ds.data_vars:
        if name.lower() in ("lat", "latitude"):
            return name
    return None


def normalize_lon(ds: xr.Dataset) -> xr.Dataset:
    """If lon is in [0,360], convert to [-180,180). Return ds with adjusted lon coord only when 1D lon exists."""
    lon_name = _get_lon_name(ds)
    if lon_name is None:
        return ds
    lon = ds[lon_name]
    # Only handle 1D lon easily and safely
    if lon.ndim == 1:
        lon_vals = lon
        try:
            max_lon = float(lon_vals.max().compute() if hasattr(lon_vals, "compute") else lon_vals.max())
        except Exception:
            max_lon = float(lon_vals.max())
        if max_lon > 180:
            new_lon = ((lon_vals + 180) % 360) - 180
            ds = ds.assign_coords({lon_name: new_lon})
            # sort by lon to keep monotonic increasing if possible
            try:
                ds = ds.sortby(lon_name)
            except Exception:
                pass
    else:
        # 2D lon: just record in findings; actual normalization skipped here
        pass
    return ds


def check_monthly(time: xr.DataArray) -> dict:
    """
    Determine if time steps are approximately monthly. Allow ±3 days tolerance around median.
    Returns {'is_monthly': bool, 'median_delta_days': float}
    """
    if time.size < 2:
        return {"is_monthly": False, "median_delta_days": float("nan")}
    tvalues = time.values
    # Handle cftime or numpy datetime64 uniformly via numpy timedelta64 days
    deltas_days = []
    for i in range(1, len(tvalues)):
        dt = tvalues[i] - tvalues[i - 1]
        # convert to days
        try:
            days = np.abs(np.asarray(dt, dtype="timedelta64[D]").astype("timedelta64[D]") / np.timedelta64(1, "D"))
            days = float(days)
        except Exception:
            # fallback using numpy
            days = float(np.abs(dt) / np.timedelta64(1, "D"))
        deltas_days.append(days)
    median_days = float(np.nanmedian(deltas_days))
    is_monthly = 27.0 <= median_days <= 34.0
    return {"is_monthly": bool(is_monthly), "median_delta_days": round(median_days, 2)}


def _nan_ratio(da: xr.DataArray) -> float:
    total = da.size
    if total == 0:
        return float("nan")
    n_nan = int(da.isnull().sum().compute())
    return float(n_nan / total)


def var_stats(da: xr.DataArray, sample_idx: List[int]) -> dict:
    """Compute basic stats: min, max, units, and NaN ratios at selected time indices (if time dim exists)."""
    out: Dict[str, object] = {}
    # Units
    out["units"] = da.attrs.get("units", None)
    # Global min/max (ignoring NaN)
    try:
        vmin = float(da.min(skipna=True).compute())
        vmax = float(da.max(skipna=True).compute())
    except Exception:
        vmin = float(da.min(skipna=True))
        vmax = float(da.max(skipna=True))
    out["min"] = vmin
    out["max"] = vmax

    # NaN ratios at sample indices along time, if present
    ratios = {}
    if ("time" in da.sizes) and da.sizes.get("time", 0) > 0:
        T = da.sizes["time"]
        for idx in sample_idx:
            ii = int(np.clip(idx, 0, max(T - 1, 0)))
            try:
                ratios[str(ii)] = _nan_ratio(da.isel(time=ii))
            except Exception:
                ratios[str(ii)] = float("nan")
    else:
        ratios["all"] = _nan_ratio(da)
    out["nan_ratio"] = ratios
    return out


def slice_bbox(ds: xr.Dataset, bbox: Tuple[float, float, float, float]) -> xr.Dataset:
    """
    Slice dataset to bbox (lat_min, lat_max, lon_min, lon_max).
    Supports 1D lat/lon or 2D lat/lon fields. If only x/y without lat/lon, returns original ds.
    """
    lat_name = _get_lat_name(ds)
    lon_name = _get_lon_name(ds)
    if lat_name is None or lon_name is None:
        return ds

    lat = ds[lat_name]
    lon = ds[lon_name]

    if lon.ndim == 1 and lat.ndim == 1:
        lat_min, lat_max, lon_min, lon_max = bbox
        # Handle potential descending coordinates
        lat_slice = slice(lat_min, lat_max) if lat[0] < lat[-1] else slice(lat_max, lat_min)
        # Ensure lon normalized to [-180, 180) prior to slicing if bbox expects that range
        ds_sub = ds.sel({lat_name: lat_slice})
        # Handle lon wrap-around: if lon_min < lon_max simple slice, else concatenate two slices
        if lon_min <= lon_max:
            # choose ascending or descending accordingly
            lon_slice = slice(lon_min, lon_max) if ds_sub[lon_name][0] < ds_sub[lon_name][-1] else slice(lon_max, lon_min)
            ds_sub = ds_sub.sel({lon_name: lon_slice})
        else:
            # crossing dateline, e.g., 170 to -170
            lon1 = slice(lon_min, 180) if ds_sub[lon_name][0] < ds_sub[lon_name][-1] else slice(180, lon_min)
            lon2 = slice(-180, lon_max) if ds_sub[lon_name][0] < ds_sub[lon_name][-1] else slice(lon_max, -180)
            ds_sub = xr.concat([ds_sub.sel({lon_name: lon1}), ds_sub.sel({lon_name: lon2})], dim=lon_name)
        return ds_sub
    else:
        # 2D curvilinear grid: mask and drop
        lat_min, lat_max, lon_min, lon_max = bbox
        # handle wrap-around by building mask in two parts if needed
        latmask = (lat >= lat_min) & (lat <= lat_max)
        if lon_min <= lon_max:
            lonmask = (lon >= lon_min) & (lon <= lon_max)
            mask = latmask & lonmask
            return ds.where(mask, drop=True)
        else:
            lonmask1 = (lon >= lon_min) & (lon <= 180)
            lonmask2 = (lon >= -180) & (lon <= lon_max)
            mask = latmask & (lonmask1 | lonmask2)
            return ds.where(mask, drop=True)


def _ensure_vars(ds: xr.Dataset) -> Tuple[xr.Dataset, Dict[str, str]]:
    """Ensure presence of required variables and return mapping to canonical names: sic, sit, snow."""
    mapping = {}
    # siconc -> sic
    if "siconc" in ds.data_vars:
        mapping["sic"] = "siconc"
    # sithick_corr fallback to sithick -> sit
    if "sithick_corr" in ds.data_vars:
        mapping["sit"] = "sithick_corr"
    elif "sithick" in ds.data_vars:
        mapping["sit"] = "sithick"
    # sisnthick -> snow
    if "sisnthick" in ds.data_vars:
        mapping["snow"] = "sisnthick"

    # attach views with renamed variables (do not copy data)
    new_vars = {}
    for canon, orig in mapping.items():
        new_vars[canon] = ds[orig]
    ds2 = ds.assign(new_vars)
    return ds2, mapping


def quick_plots(ds_box: xr.Dataset, outdir: str, times: List[int]) -> List[str]:
    """Generate quick snapshot plots for sic and sit at given time indices. Returns list of file paths."""
    out_paths: List[str] = []
    vars_to_plot = [v for v in ["sic", "sit"] if v in ds_box]
    if not vars_to_plot:
        return out_paths

    lat_name = _get_lat_name(ds_box)
    lon_name = _get_lon_name(ds_box)

    for ti in times:
        for v in vars_to_plot:
            try:
                da = ds_box[v]
                if ("time" in da.sizes) and da.sizes.get("time", 0) > 0:
                    ii = int(np.clip(ti, 0, da.sizes["time"] - 1))
                    frame = da.isel(time=ii)
                    tlabel = str(ds_box["time"].isel(time=ii).values) if "time" in ds_box.coords else f"t{ii}"
                else:
                    frame = da
                    tlabel = "static"

                plt.figure(figsize=(6, 4))
                if lat_name is not None and lon_name is not None and frame.dims and lat_name in ds_box and lon_name in ds_box:
                    if lat_name in frame.dims and lon_name in frame.dims and frame.ndim == 2:
                        im = plt.pcolormesh(ds_box[lon_name], ds_box[lat_name], frame, shading="auto")
                    else:
                        # fallback to imshow
                        im = plt.imshow(frame.values, origin="lower")
                else:
                    im = plt.imshow(frame.values, origin="lower")
                plt.colorbar(im, label=v)
                plt.title(f"{v} @ {tlabel}")
                fp = os.path.join(outdir, f"fig_{v}_{ti}.png")
                plt.tight_layout()
                plt.savefig(fp, dpi=150)
                plt.close()
                out_paths.append(fp)
            except Exception as e:
                print(f"Plot failed for {v} at time {ti}: {e}")
    return out_paths


def build_report(ds: xr.Dataset, ds_box: xr.Dataset, findings: dict, outdir: str) -> None:
    """Write JSON and Markdown reports."""
    json_path = os.path.join(outdir, "inspect_ice_nc.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(findings, f, ensure_ascii=False, indent=2)

    # Markdown summary
    md_path = os.path.join(outdir, "inspect_ice_nc.md")
    lines = []
    lines.append(f"# 冰参数 NetCDF 体检报告\n")
    lines.append(f"- 源文件: {findings.get('source')}\n")
    lines.append(f"- 维度: {findings.get('dims')}\n")
    lines.append(f"- 变量映射: {findings.get('var_mapping')}\n")
    # coords
    coords = findings.get("coords", {})
    lines.append("## 坐标/投影\n")
    lines.append(f"- time 存在: {coords.get('has_time')}\n")
    lines.append(f"- lat/lon 名称: {coords.get('lat_name')} / {coords.get('lon_name')}\n")
    lines.append(f"- lon 最大值: {coords.get('lon_max')}，{coords.get('lon_advice', '')}\n")
    # time
    tinfo = findings.get("time", {})
    lines.append("## 时间频率\n")
    lines.append(f"- 时间范围: {tinfo.get('start')} ~ {tinfo.get('end')} (共 {tinfo.get('length')} 个步)\n")
    lines.append(f"- 月频判断: {tinfo.get('is_monthly')}，中位间隔 {tinfo.get('median_delta_days')} 天\n")
    # variables
    vstats = findings.get("variables", {})
    lines.append("## 变量检查\n")
    for v, st in vstats.items():
        lines.append(f"- {v}: min={st.get('min')}, max={st.get('max')}, units={st.get('units')}, 备注={st.get('note', '')}\n")
        nr = st.get("nan_ratio", {})
        if isinstance(nr, dict):
            ratios_str = ", ".join([f"t{idx}:{val:.3f}" for idx, val in nr.items() if isinstance(val, (int, float))])
            lines.append(f"  - 抽样缺测比例: {ratios_str}\n")
    # bbox
    bbox = findings.get("bbox", {})
    lines.append("## BBOX 抽查\n")
    lines.append(f"- ARCTIC_BBOX: {bbox.get('values')}\n")
    lines.append(f"- 子区尺寸 dims: {bbox.get('dims')}\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _sample_indices(T: int) -> List[int]:
    if T <= 0:
        return []
    return [0, T // 2, max(T - 1, 0)]


def _infer_percent_need(sic_stats: dict) -> bool:
    vmax = sic_stats.get("max")
    if vmax is None or not np.isfinite(vmax):
        return False
    return vmax > 1.0 + 1e-6


def _check_thickness_units(stats: dict) -> dict:
    units = (stats.get("units") or "").strip().lower()
    vmax = stats.get("max")
    note = ""
    convert = None
    if units in ("m", "meter", "metre", "meters", "metres"):
        # typical sea ice thickness rarely > 10 m monthly mean
        if vmax is not None and np.isfinite(vmax) and vmax > 50:
            note = "数值偏大，疑似单位(cm)或尺度问题"
    elif units in ("cm", "centimeter", "centimetre", "centimeters", "centimetres"):
        note = "单位为cm，建议/需要换算为米 (/100)"
        convert = "cm_to_m"
    elif units == "":
        note = "缺少units属性，需人工核对 (m/cm?)"
    else:
        note = f"非常规单位: {units}"
    stats["note"] = note
    if convert:
        stats["convert_suggest"] = convert
    return stats


if __name__ == "__main__":
    # 1) 读取 .env
    load_dotenv()
    bbox_env = os.getenv("ARCTIC_BBOX", "65,80,25,-170")
    try:
        bbox: Tuple[float, float, float, float] = tuple(map(float, bbox_env.split(",")))  # type: ignore
    except Exception:
        bbox = (65.0, 80.0, 25.0, -170.0)

    # 解析参数：--src, --outdir
    parser = argparse.ArgumentParser(description="Inspect ice NetCDF file (siconc/sithick/snow) and output report/figures.")
    parser.add_argument("--src", type=str, default=None, help="单个 .nc 文件路径；若未提供则在默认目录中选择最新的 .nc")
    parser.add_argument("--outdir", type=str, default="reports", help="输出目录，默认 reports")
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # 默认目录（Windows 绝对路径）
    default_dir = r"C:\\Users\\sgddsf\\Desktop\\minimum\\ArcticRoute\\data\\raw\\cmems_arc"

    def _latest_nc_in_dir(d: str) -> str | None:
        if not os.path.isdir(d):
            return None
        try:
            files = [f for f in os.listdir(d) if f.lower().endswith(".nc")]
        except Exception:
            return None
        if not files:
            return None
        files_full = [os.path.join(d, f) for f in files]
        files_full.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return files_full[0]

    # 决定 src
    src: str | None = args.src
    if src is None:
        src = _latest_nc_in_dir(default_dir)
        if src is None:
            print(f"未提供 --src，且在默认目录中未找到任何 .nc 文件：{default_dir}")
            sys.exit(1)
    
    # 打印最终使用的文件路径
    print(f"使用文件: {src}")

    if not os.path.exists(src):
        print(f"错误：找不到文件：{src}")
        sys.exit(1)

    # 2) 打开与基本信息
    ds = open_ds(src)

    # 3) 变量存在性与重命名（添加别名 sic/sit/snow）
    ds, mapping = _ensure_vars(ds)

    findings: Dict[str, object] = {"source": src}
    findings["dims"] = dict(ds.sizes)
    findings["var_mapping"] = mapping

    # 维度与坐标
    lat_name = _get_lat_name(ds)
    lon_name = _get_lon_name(ds)
    coords_info = {
        "has_time": ("time" in ds.sizes) or ("time" in ds.coords),
        "lat_name": lat_name,
        "lon_name": lon_name,
        "lon_max": None,
        "lon_advice": "",
    }

    # 4) 时间频率、范围
    time_info = {"is_monthly": False, "median_delta_days": None, "start": None, "end": None, "length": 0}
    if "time" in ds.coords:
        t = ds["time"]
        time_info["length"] = int(t.size)
        time_info["start"] = str(t.values[0]) if t.size > 0 else None
        time_info["end"] = str(t.values[-1]) if t.size > 0 else None
        time_info.update(check_monthly(t))
    findings["time"] = time_info

    # 经度规范化建议 & 实施
    if lon_name is not None:
        lon = ds[lon_name]
        try:
            lon_max = float(lon.max().compute()) if hasattr(lon, "compute") else float(lon.max())
        except Exception:
            lon_max = float(lon.max())
        coords_info["lon_max"] = lon_max
        if np.isfinite(lon_max) and lon_max > 180:
            coords_info["lon_advice"] = "建议转到 [-180,180)"
        ds = normalize_lon(ds)
    findings["coords"] = coords_info

    # 缺测比例抽样索引
    sample_idx = _sample_indices(time_info.get("length", 0) or 0)

    # 数值范围与单位检查
    variables = {}
    if "sic" in ds:
        sic_stats = var_stats(ds["sic"], sample_idx)
        if _infer_percent_need(sic_stats):
            sic_stats["note"] = (sic_stats.get("note") or "") + " 需 /100 标准化"
        variables["siconc(sic)"] = sic_stats
    else:
        variables["siconc(sic)"] = {"note": "缺失"}

    if "sit" in ds:
        sit_stats = var_stats(ds["sit"], sample_idx)
        sit_stats = _check_thickness_units(sit_stats)
        variables["sithick_corr/sithick(sit)"] = sit_stats
    else:
        variables["sithick_corr/sithick(sit)"] = {"note": "缺失"}

    if "snow" in ds:
        snow_stats = var_stats(ds["snow"], sample_idx)
        snow_stats = _check_thickness_units(snow_stats)
        variables["sisnthick(snow)"] = snow_stats
    else:
        variables["sisnthick(snow)"] = {"note": "缺失"}

    findings["variables"] = variables

    # 5) BBOX 裁剪与快照图
    ds_box = slice_bbox(ds, bbox)
    findings["bbox"] = {
        "values": bbox,
        "dims": dict(ds_box.sizes),
    }

    # snapshots: 首/中/末月
    fig_paths = quick_plots(ds_box, outdir, sample_idx if sample_idx else [0])
    findings["figures"] = fig_paths

    # 6) 生成 JSON + MD 报告
    build_report(ds, ds_box, findings, outdir)

    print("检查完成。输出: ")
    print("- ", os.path.join(outdir, "inspect_ice_nc.json"))
    print("- ", os.path.join(outdir, "inspect_ice_nc.md"))
    for p in fig_paths:
        print("- ", p)

