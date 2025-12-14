from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple
import math
import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

try:
    from shapely.geometry import shape, Point, LineString, Polygon  # type: ignore
except Exception:  # pragma: no cover
    shape = None  # type: ignore
    Point = None  # type: ignore
    LineString = None  # type: ignore
    Polygon = None  # type: ignore


def _deg_per_km(lat_deg: float) -> Tuple[float, float]:
    # 粗略换算：1度纬度≈111km；经度≈111km*cos(lat)
    dlat = 1.0 / 111.0
    dlon = 1.0 / max(1e-6, 111.0 * math.cos(math.radians(max(-89.9, min(89.9, lat_deg)))))
    return dlat, dlon


def _rasterize_polygon(poly_geojson: Dict[str, Any], tmpl: "xr.DataArray") -> "xr.DataArray":
    if xr is None:
        raise RuntimeError("xarray required")
    if shape is None:
        # 无 shapely 时返回全零
        return xr.zeros_like(tmpl.isel(time=0) if "time" in tmpl.dims else tmpl)
    poly = shape(poly_geojson)
    base = tmpl.isel(time=0) if "time" in tmpl.dims else tmpl
    # 取坐标
    latn = "lat" if "lat" in base.coords else ("latitude" if "latitude" in base.coords else None)
    lonn = "lon" if "lon" in base.coords else ("longitude" if "longitude" in base.coords else None)
    if not (latn and lonn):
        raise RuntimeError("grid missing lat/lon")
    latc = np.asarray(base.coords[latn].values)
    lonc = np.asarray(base.coords[lonn].values)
    # 若为2D网格，取行/列第一条作为1D轴
    if latc.ndim == 2:
        lat1 = latc[:, 0]
        lon1 = lonc[0, :]
    else:
        lat1 = latc
        lon1 = lonc
    H, W = int(len(lat1)), int(len(lon1))
    out = np.zeros((H, W), dtype="float32")
    for i in range(H):
        for j in range(W):
            p = Point(float(lon1[j]), float(lat1[i]))
            if poly.contains(p):
                out[i, j] = 1.0
    return base.copy(data=out)


def _rasterize_line_band(line_geojson: Dict[str, Any], band_km: float, tmpl: "xr.DataArray") -> "xr.DataArray":
    if xr is None:
        raise RuntimeError("xarray required")
    if shape is None:
        return xr.zeros_like(tmpl.isel(time=0) if "time" in tmpl.dims else tmpl)
    line = shape(line_geojson)
    base = tmpl.isel(time=0) if "time" in tmpl.dims else tmpl
    latn = "lat" if "lat" in base.coords else ("latitude" if "latitude" in base.coords else None)
    lonn = "lon" if "lon" in base.coords else ("longitude" if "longitude" in base.coords else None)
    if not (latn and lonn):
        raise RuntimeError("grid missing lat/lon")
    latc = np.asarray(base.coords[latn].values)
    lonc = np.asarray(base.coords[lonn].values)
    if latc.ndim == 2:
        lat1 = latc[:, 0]
        lon1 = lonc[0, :]
                else:
        lat1 = latc
        lon1 = lonc
    # 用纬度中位数估计经纬换算
    lat_mid = float(np.median(lat1))
    dlat, dlon = _deg_per_km(lat_mid)
    # 以带宽 band_km 形成缓冲
    buf_deg = max(1e-6, band_km * max(dlat, dlon))
    band = line.buffer(buf_deg)
    H, W = int(len(lat1)), int(len(lon1))
    out = np.zeros((H, W), dtype="float32")
                for i in range(H):
                    for j in range(W):
            p = Point(float(lon1[j]), float(lat1[i]))
            if band.contains(p):
                out[i, j] = 1.0
    return base.copy(data=out)


def build_constraints(
    *,
    ym: str,
    feedback_items: Iterable[Dict[str, Any]],
    grid_like: "xr.DataArray",
    defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, "xr.DataArray"]:
    """根据反馈构建约束：
    - constraints_mask: 硬约束（1 表示禁行）
    - constraints_soft_cost: 软惩罚（[0,1]），如锁定走廊带宽区域
    """
    if xr is None:
        raise RuntimeError("xarray required")
    defaults = defaults or {"band_km_default": 5.0}
    base2d = grid_like.isel(time=0) if "time" in grid_like.dims else grid_like
    H = int(base2d.sizes.get("y", base2d.shape[-2]))
    W = int(base2d.sizes.get("x", base2d.shape[-1]))
    mask = np.zeros((H, W), dtype="float32")
    soft = np.zeros((H, W), dtype="float32")

    for it in feedback_items:
        tag = str(it.get("tag"))
        geom = it.get("geometry")
        sev = str(it.get("severity") or "med")
        band_km = float(it.get("value") or defaults.get("band_km_default", 5.0))
        if tag == "no_go_polygon" and geom:
            da = _rasterize_polygon(geom, base2d)
            mask = np.maximum(mask, np.asarray(da.values, dtype="float32"))
        elif tag in ("lock_corridor", "prefer_prior") and geom:
            da = _rasterize_line_band(geom, band_km, base2d)
            # 软惩罚权重可按严重度微调
            w = 0.5 if sev == "low" else (0.8 if sev == "med" else 1.0)
            soft = np.maximum(soft, np.asarray(da.values, dtype="float32") * float(w))
        # 其他标签可在后续扩展

    mask_da = base2d.copy(data=mask).rename("constraints_mask")
    soft_da = base2d.copy(data=np.clip(soft, 0.0, 1.0)).rename("constraints_soft_cost")
    # 广播回 time 维（如有）
    if "time" in grid_like.dims:
        tcoord = grid_like.coords["time"]
        mask_da = mask_da.expand_dims({"time": tcoord})
        soft_da = soft_da.expand_dims({"time": tcoord})
    return {"mask": mask_da, "soft": soft_da}


__all__ = ["build_constraints"]
