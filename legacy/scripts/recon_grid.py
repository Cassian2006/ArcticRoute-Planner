import os
import json
import math
import argparse
from typing import Dict, Optional, Tuple, Any

import numpy as np
import xarray as xr
import pandas as pd


def compute_chunks(ds: xr.Dataset) -> dict:
    chunks: Dict[str, int] = {}
    for dim, size in ds.sizes.items():
        if dim.lower() == "time":
            chunks[dim] = int(min(12, max(1, size)))
        else:
            chunks[dim] = int(min(256, max(1, size)))
    return chunks


def open_ds(path: str) -> xr.Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    tmp = xr.open_dataset(path, chunks={}, decode_times=True)
    chunks = compute_chunks(tmp)
    tmp.close()
    ds = xr.open_dataset(path, chunks=chunks, decode_times=True)
    return ds


def _get_name(ds: xr.Dataset, cands: Tuple[str, ...]) -> Optional[str]:
    for n in cands:
        if n in ds.coords or n in ds.variables:
            return n
    return None


def _round_sig(x: float, nd: int = 6) -> float:
    if not np.isfinite(x):
        return float("nan")
    if x == 0:
        return 0.0
    return float(round(x, nd))


def _nearest_fraction(x: float, base_fracs=(1, 1/2, 1/3, 1/4, 1/5, 1/6, 1/8, 1/10, 1/12, 1/16, 1/20, 1/24, 1/32)) -> Tuple[str, float]:
    if not np.isfinite(x):
        return ("nan", x)
    best = None
    best_err = float("inf")
    for f in base_fracs:
        err = abs(x - f)
        if err < best_err:
            best = f
            best_err = err
    if best is None:
        return (f"{x:.4f}°", x)
    # if within 5% relative error or absolute 0.01 deg, snap
    if (best_err <= 0.01) or (best_err / max(x, 1e-9) <= 0.05):
        # format 1/4° style if < 1
        if best < 1 and best > 0:
            num = 1
            den = round(1 / best)
            return (f"{num}/{den}°", best)
        else:
            return (f"{best:.2f}°", best)
    return (f"{x:.4f}°", x)


def infer_grid_resolution(ds: xr.Dataset, lat_name: Optional[str], lon_name: Optional[str]) -> Dict[str, Any]:
    res: Dict[str, Any] = {
        "lat_name": lat_name,
        "lon_name": lon_name,
        "lat_ndim": None,
        "lon_ndim": None,
        "lat_res_deg": None,
        "lon_res_deg": None,
        "lat_res_pretty": None,
        "lon_res_pretty": None,
        "grid_type": None,
    }
    if lat_name is None or lon_name is None:
        return res
    lat = ds[lat_name]
    lon = ds[lon_name]
    res["lat_ndim"] = int(lat.ndim)
    res["lon_ndim"] = int(lon.ndim)

    if lat.ndim == 1 and lon.ndim == 1:
        # Rectilinear
        res["grid_type"] = "rectilinear_1d"
        try:
            dlat = np.diff(np.asarray(lat.values, dtype=float))
            dlon = np.diff(np.asarray(lon.values, dtype=float))
            # use median of absolute diffs, ignore zeros
            dlat_med = float(np.median(np.abs(dlat[np.nonzero(dlat)]))) if dlat.size > 0 else float("nan")
            dlon_med = float(np.median(np.abs(dlon[np.nonzero(dlon)]))) if dlon.size > 0 else float("nan")
        except Exception:
            dlat_med = float("nan")
            dlon_med = float("nan")
        res["lat_res_deg"] = _round_sig(dlat_med, 6)
        res["lon_res_deg"] = _round_sig(dlon_med, 6)
        res["lat_res_pretty"], _ = _nearest_fraction(res["lat_res_deg"]) if np.isfinite(res["lat_res_deg"]) else ("nan", res["lat_res_deg"]) 
        res["lon_res_pretty"], _ = _nearest_fraction(res["lon_res_deg"]) if np.isfinite(res["lon_res_deg"]) else ("nan", res["lon_res_deg"]) 
    else:
        # Curvilinear 2D grid: estimate local spacing near center by great-circle in degrees approx
        res["grid_type"] = "curvilinear_2d"
        try:
            latv = np.asarray(lat.values, dtype=float)
            lonv = np.asarray(lon.values, dtype=float)
            # choose center indices
            iy = latv.shape[-2] // 2 if lat.ndim >= 2 else 0
            ix = latv.shape[-1] // 2 if lat.ndim >= 2 else 0
            # neighbor deltas along y and x
            def safe_get(a, i, j):
                i = max(0, min(a.shape[-2]-1, i))
                j = max(0, min(a.shape[-1]-1, j))
                return a[i, j]
            lat_c = safe_get(latv, iy, ix)
            lon_c = safe_get(lonv, iy, ix)
            lat_y = safe_get(latv, iy+1, ix)
            lon_y = safe_get(lonv, iy+1, ix)
            lat_x = safe_get(latv, iy, ix+1)
            lon_x = safe_get(lonv, iy, ix+1)
            dlat_y = abs(lat_y - lat_c)
            dlon_x = abs(lon_x - lon_c)
            # crude: report as degrees
            res["lat_res_deg"] = _round_sig(float(dlat_y), 6)
            res["lon_res_deg"] = _round_sig(float(dlon_x), 6)
            res["lat_res_pretty"], _ = _nearest_fraction(res["lat_res_deg"]) if np.isfinite(res["lat_res_deg"]) else ("nan", res["lat_res_deg"]) 
            res["lon_res_pretty"], _ = _nearest_fraction(res["lon_res_deg"]) if np.isfinite(res["lon_res_deg"]) else ("nan", res["lon_res_deg"]) 
        except Exception:
            pass
    return res


def infer_time_frequency(ds: xr.Dataset) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "has_time": False,
        "length": 0,
        "start": None,
        "end": None,
        "freq": None,
        "delta_hours": None,
    }
    if "time" not in ds.coords:
        return out
    t = ds["time"].to_index() if hasattr(ds["time"], "to_index") else pd.Index(ds["time"].values)
    out["has_time"] = True
    out["length"] = int(len(t))
    if len(t) > 0:
        out["start"] = str(t[0])
        out["end"] = str(t[-1])
    if len(t) >= 3:
        try:
            freq = pd.infer_freq(t)
        except Exception:
            freq = None
        if freq is None:
            # fallback via median delta
            deltas = pd.Series(t[1:]).values - pd.Series(t[:-1]).values
            try:
                med = np.median(deltas.astype("timedelta64[h]") / np.timedelta64(1, "h"))
                out["delta_hours"] = float(med)
                if abs(med - 6) < 0.1:
                    out["freq"] = "6H"
                elif abs(med - 3) < 0.1:
                    out["freq"] = "3H"
                elif abs(med - 1) < 0.1:
                    out["freq"] = "H"
                elif abs(med - 24) < 0.1:
                    out["freq"] = "D"
                else:
                    out["freq"] = f"~{med:.1f}H"
            except Exception:
                out["freq"] = None
        else:
            out["freq"] = freq
            # derive hours if possible
            try:
                # convert first diff to hours
                if len(t) >= 2:
                    dh = (t[1] - t[0]) / np.timedelta64(1, "h")
                    out["delta_hours"] = float(dh)
            except Exception:
                pass
    return out


def ds_summary(ds: xr.Dataset) -> Dict[str, Any]:
    lat_name = _get_name(ds, ("lat", "latitude", "y"))
    lon_name = _get_name(ds, ("lon", "longitude", "x"))
    grid = infer_grid_resolution(ds, lat_name, lon_name)
    time_info = infer_time_frequency(ds)
    return {
        "dims": dict(ds.sizes),
        "coords": {
            "lat": lat_name,
            "lon": lon_name,
            "time": "time" if "time" in ds.coords else None,
        },
        "attrs": dict(ds.attrs),
        "grid": grid,
        "time": time_info,
    }


def find_latest_merged(dir_path: str) -> Tuple[Optional[str], Optional[str]]:
    if not os.path.isdir(dir_path):
        return None, None
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.nc')]
    sic = [f for f in files if os.path.basename(f).startswith('sic_fcst_')]
    cost = [f for f in files if os.path.basename(f).startswith('ice_cost_')]
    def latest(lst):
        if not lst:
            return None
        lst.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return lst[0]
    return latest(sic), latest(cost)


def main():
    parser = argparse.ArgumentParser(description="Recon grid/time spec from merged P1 outputs")
    parser.add_argument("--merged_dir", type=str, default=os.path.join("ArcticRoute", "data_processed", "ice_forecast", "merged"), help="目录，包含 sic_fcst_*.nc 与 ice_cost_*.nc")
    parser.add_argument("--out", type=str, default=os.path.join("reports", "recon", "grid_spec.json"), help="输出 JSON 路径")
    parser.add_argument("--sic", type=str, default=None, help="显式指定 sic_fcst_*.nc")
    parser.add_argument("--cost", type=str, default=None, help="显式指定 ice_cost_*.nc")
    args = parser.parse_args()

    sic_path = args.sic
    cost_path = args.cost

    if sic_path is None or cost_path is None:
        s, c = find_latest_merged(args.merged_dir)
        sic_path = sic_path or s
        cost_path = cost_path or c

    if sic_path is None or not os.path.exists(sic_path):
        raise FileNotFoundError(f"未找到 sic_fcst_*.nc 于 {args.merged_dir}")
    if cost_path is None or not os.path.exists(cost_path):
        raise FileNotFoundError(f"未找到 ice_cost_*.nc 于 {args.merged_dir}")

    ds_sic = open_ds(sic_path)
    ds_cost = open_ds(cost_path)

    sum_sic = ds_summary(ds_sic)
    sum_cost = ds_summary(ds_cost)

    # 对齐检查
    aligned = (sum_sic["dims"] == sum_cost["dims"]) and (
        sum_sic["grid"]["lat_res_deg"] == sum_cost["grid"]["lat_res_deg"] and
        sum_sic["grid"]["lon_res_deg"] == sum_cost["grid"]["lon_res_deg"]
    ) and (sum_sic["time"]["freq"] == sum_cost["time"]["freq"]) 

    # 汇总 freq 与分辨率（优先 sic）
    freq = sum_sic["time"].get("freq") or sum_cost["time"].get("freq")
    delta_hours = sum_sic["time"].get("delta_hours") or sum_cost["time"].get("delta_hours")
    lat_res_pretty = sum_sic["grid"].get("lat_res_pretty") or sum_cost["grid"].get("lat_res_pretty")
    lon_res_pretty = sum_sic["grid"].get("lon_res_pretty") or sum_cost["grid"].get("lon_res_pretty")

    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)

    spec = {
        "sources": {"sic_fcst": sic_path, "ice_cost": cost_path},
        "sic_summary": sum_sic,
        "cost_summary": sum_cost,
        "contract": {
            "aligned": bool(aligned),
            "freq": freq,
            "delta_hours": delta_hours,
            "grid_resolution": {
                "lat": lat_res_pretty,
                "lon": lon_res_pretty,
                "lat_deg": sum_sic["grid"].get("lat_res_deg"),
                "lon_deg": sum_sic["grid"].get("lon_res_deg"),
                "type": sum_sic["grid"].get("grid_type"),
            },
        },
    }

    # ---- JSON 安全化转换 ----
    def json_safe(o):
        import numpy as _np
        import pandas as _pd
        if isinstance(o, (str, int, float, bool)) or o is None:
            return o
        if isinstance(o, (np.generic,)):
            return o.item()
        if isinstance(o, (np.ndarray,)):
            return [json_safe(x) for x in o.tolist()]
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        if isinstance(o, (dict,)):
            return {str(k): json_safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [json_safe(x) for x in o]
        # Fallback: string repr
        return str(o)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(json_safe(spec), f, ensure_ascii=False, indent=2)

    print(f"已生成: {args.out}")


if __name__ == "__main__":
    main()

