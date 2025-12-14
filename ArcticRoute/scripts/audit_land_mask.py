"""
ArcticRoute/scripts/audit_land_mask.py

Phase LM-0: 对 land_mask 与 GEBCO bathymetry 进行盘点与一致性审计（只读，不写数据）。

运行：
  python -m ArcticRoute.scripts.audit_land_mask

输出：
  - 控制台打印统计信息
  - 报告写入 ArcticRoute/reports/land_mask_audit_report.md（覆盖）
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import xarray as xr


# ----------------------- 路径与工具 -----------------------

def get_project_root() -> Path:
    # 本文件位于 minimum/ArcticRoute/ArcticRoute/scripts/ 下
    return Path(__file__).resolve().parents[2]


def detect_lat_lon_names(ds: xr.Dataset) -> Tuple[str, str]:
    lat_cands = [
        "lat", "latitude", "nav_lat", "LAT", "Latitude", "y"
    ]
    lon_cands = [
        "lon", "longitude", "nav_lon", "LON", "Longitude", "x"
    ]
    # 先看 coords
    for n in ds.coords:
        ln = n.lower()
        if any(c == n for c in lat_cands) or ln == "lat":
            lat = n
            break
    else:
        lat = None
    for n in ds.coords:
        ln = n.lower()
        if any(c == n for c in lon_cands) or ln == "lon":
            lon = n
            break
    else:
        lon = None
    # 再看 dims/data_vars
    if (lat is None or lon is None) and len(ds.data_vars) > 0:
        any_da = next(iter(ds.data_vars))
        dv = ds[any_da]
        for cand in lat_cands:
            if cand in dv.coords or cand in dv.dims:
                lat = lat or cand
                break
        for cand in lon_cands:
            if cand in dv.coords or cand in dv.dims:
                lon = lon or cand
                break
    if lat is None or lon is None:
        # 再扫描 data_vars
        for name, da in ds.data_vars.items():
            dims_lower = [d.lower() for d in da.dims]
            if lat is None and any("lat" in d for d in dims_lower):
                lat = next(d for d in da.dims if "lat" in d.lower())
            if lon is None and any("lon" in d for d in dims_lower):
                lon = next(d for d in da.dims if "lon" in d.lower())
    if lat is None or lon is None:
        raise ValueError(f"无法识别经纬度坐标名：coords={list(ds.coords)} dims={list(ds.dims)}")
    return lat, lon


def _choose_var(ds: xr.Dataset, prefer: Optional[List[str]] = None) -> str:
    if prefer:
        for v in prefer:
            if v in ds.data_vars:
                return v
    # 选第一个二维 float 变量
    for name, da in ds.data_vars.items():
        try:
            if da.ndim >= 2 and getattr(da.dtype, 'kind', 'f') in 'fi':
                return name
        except Exception:
            continue
    # 退化：任选第一个
    return next(iter(ds.data_vars))


def _value_counts01(da: xr.DataArray) -> dict:
    arr = np.asarray(da.values)
    nan = int(np.count_nonzero(~np.isfinite(arr)))
    zeros = int(np.count_nonzero(arr == 0))
    ones = int(np.count_nonzero(arr == 1))
    others = int(np.size(arr) - zeros - ones - nan)
    return {"zeros": zeros, "ones": ones, "others": others, "nans": nan, "size": int(np.size(arr))}


def _range_stats(da: xr.DataArray) -> dict:
    vals = np.asarray(da.values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"min": np.nan, "max": np.nan}
    return {"min": float(np.min(vals)), "max": float(np.max(vals))}


def _latlon_range(da: xr.DataArray) -> tuple:
    ds = da.to_dataset(name="tmp")
    latn, lonn = detect_lat_lon_names(ds)
    lat = np.asarray(da[latn].values)
    lon = np.asarray(da[lonn].values)
    latmin, latmax = float(np.nanmin(lat)), float(np.nanmax(lat))
    lonmin, lonmax = float(np.nanmin(lon)), float(np.nanmax(lon))
    return latn, lonn, latmin, latmax, lonmin, lonmax


def _interp_like(src: xr.DataArray, tgt: xr.DataArray) -> xr.DataArray:
    # 按坐标插值/对齐，保持二维
    src_ds = src.to_dataset(name="v")
    tgt_ds = tgt.to_dataset(name="tgt")
    slat, slon = detect_lat_lon_names(src_ds)
    tlat, tlon = detect_lat_lon_names(tgt_ds)
    try:
        interp = src.interp({slat: tgt[tlat], slon: tgt[tlon]})
        return interp
    except Exception:
        # 兜底：最近邻重采样（非常粗糙）
        arr = np.asarray(src.values)
        H, W = int(tgt.shape[-2]), int(tgt.shape[-1])
        si, sj = arr.shape[-2], arr.shape[-1]
        yi = (np.linspace(0, si - 1, H)).astype(int)
        xj = (np.linspace(0, sj - 1, W)).astype(int)
        return xr.DataArray(arr[yi[:, None], xj[None, :]], dims=("y", "x"))


# ----------------------- 审计逻辑 -----------------------

def main():
    root = get_project_root()
    base = root / "ArcticRoute"
    newenv_dir = base / "data_processed" / "newenv"
    report_path = base / "reports" / "land_mask_audit_report.md"

    print("===== LM-0 | 数据盘点 =====")
    print("newenv_dir:", newenv_dir)

    land_path = newenv_dir / "land_mask_gebco.nc"
    bathy_path = newenv_dir / "gebco_bathy_clip.nc"

    land_da = None
    bathy_da = None

    # 读取 land_mask_gebco
    land_stats = {}
    if land_path.exists():
        ds_land = xr.open_dataset(land_path)
        try:
            # 变量优先匹配名称含 land/mask
            var = None
            for name in ds_land.data_vars:
                lname = name.lower()
                if ("land" in lname or "mask" in lname) and ds_land[name].ndim >= 2:
                    var = name
                    break
            if var is None:
                var = _choose_var(ds_land)
            land_da = ds_land[var]
            land_da.load()
            latn, lonn, latmin, latmax, lonmin, lonmax = _latlon_range(land_da)
            land_stats = _value_counts01(land_da)
            print(f"[LAND] file={land_path.name} var={var} shape={tuple(land_da.shape)} dims={land_da.dims}")
            print(f"       grid: {latn}={latmin:.3f}..{latmax:.3f} {lonn}={lonmin:.3f}..{lonmax:.3f}")
            print(f"       values: zeros={land_stats['zeros']} ones={land_stats['ones']} others={land_stats['others']} nans={land_stats['nans']} size={land_stats['size']}")
        finally:
            ds_land.close()
    else:
        print("[LAND] 缺失:", land_path)

    # 读取 bathy
    bathy_stats = {}
    if bathy_path.exists():
        ds_b = xr.open_dataset(bathy_path)
        try:
            var = None
            for cand in ["elevation", "z", "depth", "bathymetry", "bathy"]:
                if cand in ds_b.data_vars:
                    var = cand
                    break
            if var is None:
                var = _choose_var(ds_b)
            bathy_da = ds_b[var]
            bathy_da.load()
            rng = _range_stats(bathy_da)
            latn, lonn, latmin, latmax, lonmin, lonmax = _latlon_range(bathy_da)
            bathy_stats = {"min": rng["min"], "max": rng["max"],
                           "latmin": latmin, "latmax": latmax, "lonmin": lonmin, "lonmax": lonmax}
            print(f"[BATHY] file={bathy_path.name} var={var} shape={tuple(bathy_da.shape)} dims={bathy_da.dims}")
            print(f"        grid: {latn}={latmin:.3f}..{latmax:.3f} {lonn}={lonmin:.3f}..{lonmax:.3f}")
            print(f"        elevation range: {rng['min']:.1f} .. {rng['max']:.1f}  (GEBCO: 海洋为负、陆地为正)")
        finally:
            ds_b.close()
    else:
        print("[BATHY] 缺失:", bathy_path)

    # env_clean.nc（两种可能路径）
    env_clean_paths = [base / "data_processed" / "env_clean.nc", base / "data_processed" / "env" / "env_clean.nc"]
    env_clean_found = None
    env_lm_stats = None
    for p in env_clean_paths:
        if p.exists():
            env_clean_found = p
            break
    if env_clean_found is not None:
        ds_env = xr.open_dataset(env_clean_found)
        try:
            # 寻找 land 掩膜变量
            var = None
            for name in ds_env.data_vars:
                lname = name.lower()
                if ("land" in lname or "mask" in lname) and ds_env[name].ndim >= 2:
                    var = name
                    break
            if var is not None:
                da = ds_env[var].load()
                latn, lonn, latmin, latmax, lonmin, lonmax = _latlon_range(da)
                env_lm_stats = _value_counts01(da)
                print(f"[ENV_CLEAN] file={env_clean_found.name} var={var} shape={tuple(da.shape)} dims={da.dims}")
                print(f"            grid: {latn}={latmin:.3f}..{latmax:.3f} {lonn}={lonmin:.3f}..{lonmax:.3f}")
                print(f"            values: zeros={env_lm_stats['zeros']} ones={env_lm_stats['ones']} others={env_lm_stats['others']} nans={env_lm_stats['nans']} size={env_lm_stats['size']}")
            else:
                print(f"[ENV_CLEAN] {env_clean_found.name} 内未发现 land/mask 变量")
        finally:
            ds_env.close()
    else:
        print("[ENV_CLEAN] 未找到 env_clean.nc（可选项）")

    # 一致性：对比 land_mask_gebco 与 (bathy>=0)
    mismatch_ratio = np.nan
    mismatch_cnt = -1
    total_cnt = -1
    if (land_da is not None) and (bathy_da is not None):
        # 保证对齐
        try:
            bathy_on_land = _interp_like(bathy_da, land_da)
        except Exception:
            bathy_on_land = bathy_da
        mask_from_bathy = (bathy_on_land >= 0).astype(np.uint8)
        # 有些 land_da 不是 0/1，统一到二值
        land01 = xr.where(land_da > 0.5, 1, xr.where(land_da < 0.5, 0, np.nan)).astype("float32")
        both_valid = np.isfinite(land01.values) & np.isfinite(mask_from_bathy.values)
        mism = (land01.values != mask_from_bathy.values) & both_valid
        mismatch_cnt = int(np.count_nonzero(mism))
        total_cnt = int(np.count_nonzero(both_valid))
        mismatch_ratio = float(mismatch_cnt / total_cnt) if total_cnt > 0 else np.nan
        print(f"[COMPARE] land_mask_gebco vs (bathy>=0): mismatch={mismatch_cnt} / {total_cnt} = {mismatch_ratio:.6f}")

    # 写报告
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Land Mask 审计报告（初始版）\n\n")
        f.write("本报告由 scripts/audit_land_mask.py 自动生成（只读统计）。\n\n")
        f.write("## 数据位置\n\n")
        f.write(f"- newenv: {newenv_dir.as_posix()}\n")
        f.write(f"- land_mask_gebco.nc: {'存在' if land_da is not None else '缺失'}\n")
        f.write(f"- gebco_bathy_clip.nc: {'存在' if bathy_da is not None else '缺失'}\n")
        f.write(f"- env_clean.nc: {env_clean_found.name if env_clean_found else '未找到'}\n\n")
        if land_da is not None:
            latn, lonn, latmin, latmax, lonmin, lonmax = _latlon_range(land_da)
            st = land_stats
            f.write("## land_mask_gebco 基本统计\n\n")
            f.write(f"- shape: {tuple(land_da.shape)} dims: {land_da.dims}\n")
            f.write(f"- grid: {latn}={latmin:.3f}..{latmax:.3f} {lonn}={lonmin:.3f}..{lonmax:.3f}\n")
            f.write(f"- 值统计: zeros={st['zeros']} ones={st['ones']} others={st['others']} nans={st['nans']} size={st['size']}\n\n")
        if bathy_da is not None:
            rng = _range_stats(bathy_da)
            latn, lonn, latmin, latmax, lonmin, lonmax = _latlon_range(bathy_da)
            f.write("## GEBCO bathymetry 基本统计\n\n")
            f.write(f"- shape: {tuple(bathy_da.shape)} dims: {bathy_da.dims}\n")
            f.write(f"- grid: {latn}={latmin:.3f}..{latmax:.3f} {lonn}={lonmin:.3f}..{lonmax:.3f}\n")
            f.write(f"- elevation range: {rng['min']:.1f} .. {rng['max']:.1f}（GEBCO: 海洋为负、陆地为正）\n\n")
        if env_lm_stats is not None:
            f.write("## env_clean 内 land_mask（如存在）\n\n")
            f.write(f"- 值统计: zeros={env_lm_stats['zeros']} ones={env_lm_stats['ones']} others={env_lm_stats['others']} nans={env_lm_stats['nans']} size={env_lm_stats['size']}\n\n")
        if np.isfinite(mismatch_ratio):
            f.write("## 一致性（land_mask_gebco vs bathy>=0）\n\n")
            f.write(f"- mismatch 像元: {mismatch_cnt} / {total_cnt} = {mismatch_ratio:.6f}\n")
        f.write("\n（注：后续 Phase 将在修复后覆盖更新本报告。）\n")

    print("===== LM-0 | 审计完成，报告已写入:", report_path)


if __name__ == "__main__":
    main()










