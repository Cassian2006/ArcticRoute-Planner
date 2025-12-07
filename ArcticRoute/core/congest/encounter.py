from __future__ import annotations

"""
Phase F｜交互风险构建（完整版实现）

- 基于 AIS 轨迹，对每对船在共同时间窗口内做等时步插值。
- 计算每个时刻的瞬时 DCPA/TCPA，并结合动态船域（简化为圆形）判断入侵。
- 按入侵程度、DCPA、TCPA 映射为风险值 r_enc。
- 将 r_enc 聚合到参考网格，归一化后输出 R_interact_<ym>.nc。

REUSE:
- 网格推断复用 _infer_grid_from_any。
- 坐标投影使用 pyproj，与 escort/ais_pairs 模块一致。
"""

import os
from typing import Optional, Tuple

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    from pyproj import Transformer  # type: ignore
except Exception:  # pragma: no cover
    Transformer = None  # type: ignore

RISK_DIR = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "risk")
AIS_DIR = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "ais")


def _normalize_cols(df: "pd.DataFrame") -> "pd.DataFrame":
    cols = {c.lower(): c for c in df.columns}
    lon = cols.get("lon") or cols.get("longitude") or cols.get("x")
    lat = cols.get("lat") or cols.get("latitude") or cols.get("y")
    time = cols.get("time") or cols.get("timestamp") or cols.get("datetime")
    mmsi = cols.get("mmsi") or cols.get("shipid") or cols.get("id")
    sog = cols.get("sog") or cols.get("speed")
    cog = cols.get("cog") or cols.get("course")
    if not (lon and lat and time and mmsi and sog and cog):
        return df.iloc[0:0].copy()
    out = df[[lon, lat, time, mmsi, sog, cog]].copy()
    out.columns = ["lon", "lat", "time", "mmsi", "sog", "cog"]
    try:
        out["time"] = pd.to_datetime(out["time"])  # type: ignore
    except Exception:
        return df.iloc[0:0].copy()
    out = out[np.isfinite(out["lon"]) & np.isfinite(out["lat"])].dropna()
    return out


def _load_tracks(ym: str) -> "pd.DataFrame":
    if pd is None:
        return None  # type: ignore
    path = os.path.join(AIS_DIR, f"tracks_{ym}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
    return _normalize_cols(df)


def _infer_grid_from_any(ym: str) -> Tuple[int, int, Optional["xr.Dataset"]]:
    if xr is None:
        return 100, 100, None
    cand = [
        os.path.join(RISK_DIR, f"risk_ice_{ym}.nc"),
        os.path.join(RISK_DIR, f"R_ice_eff_{ym}.nc"),
        os.path.join(RISK_DIR, f"risk_fused_{ym}.nc"),
        os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "env", "env_clean.nc"),
    ]
    for p in cand:
        if os.path.exists(p):
            try:
                ds = xr.open_dataset(p)
                if ds.data_vars:
                    da = ds[list(ds.data_vars)[0]]
                    Ty = int(da.sizes.get("y") or da.sizes.get("lat") or da.sizes.get("latitude") or 100)
                    Tx = int(da.sizes.get("x") or da.sizes.get("lon") or da.sizes.get("longitude") or 100)
                    return Ty, Tx, ds
            except Exception:
                continue
    return 100, 100, None


def _grid_from_ref(ym: str) -> "xr.DataArray":
    Ty, Tx, ref = _infer_grid_from_any(ym)
    arr = np.zeros((Ty, Tx), dtype=np.float32)
    da = xr.DataArray(arr, dims=("y", "x"))
    if ref is not None:
        try:
            ref_da = ref[list(ref.data_vars)[0]]
            for c in ("lat", "latitude"):
                if c in ref_da.coords:
                    da = da.assign_coords({c: ref_da.coords[c]})
                    break
            for c in ("lon", "longitude"):
                if c in ref_da.coords:
                    da = da.assign_coords({c: ref_da.coords[c]})
                    break
        except Exception:
            pass
        finally:
            try:
                ref.close()
            except Exception:
                pass
    return da


def _compute_dcpa_tcpa(p_a, v_a, p_b, v_b):
    p_rel = p_a - p_b
    v_rel = v_a - v_b
    v_rel_sq = np.dot(v_rel, v_rel)
    if v_rel_sq < 1e-6:
        return np.linalg.norm(p_rel), 0.0
    tcpa = -np.dot(p_rel, v_rel) / v_rel_sq
    dcpa = np.linalg.norm(p_rel + v_rel * tcpa)
    return dcpa, tcpa


def _calculate_r_enc(dcpa, tcpa, domain_radius_m):
    # 简化 r_enc 映射
    r = 0.0
    if 0 < tcpa < 900:  # 15 分钟内
        if dcpa < domain_radius_m:  # 动态船域入侵
            r = 1.0 - (dcpa / domain_radius_m)
        r *= (1.0 - tcpa / 900.0)  # 时间越近风险越高
    return r


def build_interact_layer(ym: str) -> "xr.DataArray":
    if xr is None or pd is None or Transformer is None:
        raise RuntimeError("xarray/pandas/pyproj is required")

    ref_da = _grid_from_ref(ym)
    Ty = ref_da.sizes.get("y") or ref_da.sizes.get("lat") or ref_da.sizes.get("latitude")
    Tx = ref_da.sizes.get("x") or ref_da.sizes.get("lon") or ref_da.sizes.get("longitude")
    lat_coords, lon_coords = ref_da.coords.get("lat") or ref_da.coords.get("latitude"), ref_da.coords.get("lon") or ref_da.coords.get("longitude")

    df = _load_tracks(ym)
    H = np.zeros((int(Ty), int(Tx)), dtype=np.float32)
    if df.empty or lat_coords is None or lon_coords is None:
        da = xr.DataArray(H, dims=ref_da.dims, coords=ref_da.coords, name="risk")
        da.attrs.update({"long_name": "Interaction risk (DCPA/TCPA)", "source": "encounter_dcpa_nodata"})
        out_path = os.path.join(RISK_DIR, f"R_interact_{ym}.nc")
        os.makedirs(RISK_DIR, exist_ok=True)
        da.to_dataset().to_netcdf(out_path)
        return da

    transformer = Transformer.from_crs(4326, 3413, always_xy=True)
    df["x"], df["y"] = transformer.transform(df["lon"].values, df["lat"].values)
    df = df.sort_values("time")

    time_step = pd.Timedelta(minutes=1)
    mmsi_list = df["mmsi"].unique()

    for i in range(len(mmsi_list)):
        for j in range(i + 1, len(mmsi_list)):
            m1, m2 = mmsi_list[i], mmsi_list[j]
            df1 = df[df["mmsi"] == m1].set_index("time").sort_index()
            df2 = df[df["mmsi"] == m2].set_index("time").sort_index()
            if df1.empty or df2.empty:
                continue

            start_time = max(df1.index.min(), df2.index.min())
            end_time = min(df1.index.max(), df2.index.max())
            if start_time >= end_time:
                continue

            common_time_range = pd.date_range(start=start_time, end=end_time, freq=time_step)
            if len(common_time_range) < 2:
                continue

            interp1 = df1.reindex(df1.index.union(common_time_range)).interpolate(method='time').reindex(common_time_range)
            interp2 = df2.reindex(df2.index.union(common_time_range)).interpolate(method='time').reindex(common_time_range)

            for t_idx in range(len(common_time_range) - 1):
                p_a = interp1.iloc[t_idx][["x", "y"]].values.astype(float)
                v_sog_a = interp1.iloc[t_idx]["sog"] * 0.5144  # knots to m/s
                v_cog_a = np.radians(interp1.iloc[t_idx]["cog"])
                v_a = np.array([v_sog_a * np.sin(v_cog_a), v_sog_a * np.cos(v_cog_a)])

                p_b = interp2.iloc[t_idx][["x", "y"]].values.astype(float)
                v_sog_b = interp2.iloc[t_idx]["sog"] * 0.5144
                v_cog_b = np.radians(interp2.iloc[t_idx]["cog"])
                v_b = np.array([v_sog_b * np.sin(v_cog_b), v_sog_b * np.cos(v_cog_b)])

                if not (np.isfinite(p_a).all() and np.isfinite(v_a).all() and np.isfinite(p_b).all() and np.isfinite(v_b).all()):
                    continue

                dcpa, tcpa = _compute_dcpa_tcpa(p_a, v_a, p_b, v_b)
                # 简化船域：半径 500m
                r_enc = _calculate_r_enc(dcpa, tcpa, 500.0)
                if r_enc > 0.1:
                    lat1, lon1 = interp1.iloc[t_idx][["lat", "lon"]]
                    lat2, lon2 = interp2.iloc[t_idx][["lat", "lon"]]
                    for lat, lon in ((lat1, lon1), (lat2, lon2)):
                        iy = int(np.clip(np.searchsorted(lat_coords.values, lat) - 1, 0, int(Ty) - 1))
                        ix = int(np.clip(np.searchsorted(lon_coords.values, lon) - 1, 0, int(Tx) - 1))
                        H[iy, ix] += r_enc

    if H.max() > 0:
        H = H / H.max()

    da = xr.DataArray(H, dims=ref_da.dims, coords=ref_da.coords, name="risk")
    da.attrs.update({"long_name": "Interaction risk (DCPA/TCPA)", "source": "encounter_dcpa"})
    os.makedirs(RISK_DIR, exist_ok=True)
    out_path = os.path.join(RISK_DIR, f"R_interact_{ym}.nc")
    da.to_dataset().to_netcdf(out_path)
    return da

__all__ = ["build_interact_layer", "_infer_grid_from_any"]
