from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

from ArcticRoute.core.interfaces import PredictorOutput
from ArcticRoute.core.planners.astar_grid_time import AStarGridTimePlanner
from ArcticRoute.core.cost.env_risk_cost import EnvRiskCostProvider
from ArcticRoute.core.route.weights import iter_weight_grid
from ArcticRoute.core.route.metrics import summarize_route
from ArcticRoute.core.route.pareto import nondominated, pick_representatives
from ArcticRoute.core.cost.aggregators import aggregate_risk  # REUSE: Phase I 聚合器
from ArcticRoute.core.eco.fuel import fuel_per_nm_map, eco_cost_norm  # REUSE
from ArcticRoute.core.eco.route_eval import eval_route_eco  # REUSE
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RISK_DIR = os.path.join(REPO_ROOT, "ArcticRoute", "data_processed", "risk")
PRIOR_DIR = os.path.join(REPO_ROOT, "ArcticRoute", "data_processed", "prior")
ROUTES_DIR = os.path.join(REPO_ROOT, "ArcticRoute", "data_processed", "routes")
REPORT_DIR = os.path.join(REPO_ROOT, "ArcticRoute", "reports", "d_stage", "phaseG")
ECO_DIR = os.path.join(REPO_ROOT, "ArcticRoute", "data_processed", "eco")


def _open_da(path: str, var_hint: Optional[str] = None, return_ds: bool = False):
    if xr is None or not path or not os.path.exists(path):
        return (None, None) if return_ds else None
    try:
        ds = xr.open_dataset(path)
        if var_hint and var_hint in ds:
            da = ds[var_hint]
        else:
            # try common names
            for k in ("Risk", "risk", "prior_penalty", "P_prior", "R_ice", "P"):
                if k in ds:
                    da = ds[k]
                    break
            else:
                da = ds[list(ds.data_vars)[0]] if ds.data_vars else None
        return (da, ds) if return_ds else da
    except Exception:
        return (None, None) if return_ds else None


def _load_layers(ym: str, risk_source: str, risk_agg: str = "mean", alpha: float = 0.95) -> Dict[str, Optional["xr.DataArray"]]:
    # risk
    if risk_source == "fused":
        p_risk = os.path.join(RISK_DIR, f"risk_fused_{ym}.nc")
        risk_da, risk_ds = _open_da(p_risk, "Risk", return_ds=True)
        risk = risk_da
        if risk is not None and risk_ds is not None:
            # Manually attach lat/lon coordinates to the DataArray before closing the dataset
            coords_to_assign = {}
            lat_name = next((n for n in ('lat', 'latitude') if n in risk_ds), None)
            lon_name = next((n for n in ('lon', 'longitude') if n in risk_ds), None)
            if lat_name:
                coords_to_assign[lat_name] = risk_ds[lat_name]
            if lon_name:
                coords_to_assign[lon_name] = risk_ds[lon_name]
            if coords_to_assign:
                risk = risk.assign_coords(coords_to_assign)
        # Phase I: 若存在方差，则按聚合模式进行二次聚合（mean/q/cvar）
        if risk is not None and risk_ds is not None:
            var_da = None
            for cand in ("RiskVar", "risk_var", "risk_variance", "Var", "variance"):
                if cand in risk_ds:
                    var_da = risk_ds[cand]
                    break
            try:
                if var_da is not None:
                    rm = np.asarray(risk.values, dtype=float)
                    rv = np.asarray(var_da.values, dtype=float)
                    agg = aggregate_risk(rm, rv, mode=risk_agg, alpha=float(alpha))
                    risk = risk.copy(data=np.asarray(agg, dtype="float32"))
                    risk.name = "Risk"
                    risk.attrs["aggregated"] = risk_agg
                    risk.attrs["alpha"] = float(alpha)
            except Exception:
                # 无法聚合则按原样使用
                pass
            try:
                risk_ds.close()
            except Exception:
                pass
    else:
        # ice
        cand = [
            os.path.join(RISK_DIR, f"R_ice_eff_{ym}.nc"),
            os.path.join(RISK_DIR, f"risk_ice_{ym}.nc"),
        ]
        risk = None
        for p in cand:
            risk = _open_da(p, None)
            if risk is not None:
                break
    # prior penalty
    prior = None
    p_sel = os.path.join(PRIOR_DIR, f"prior_corridor_selected_{ym}.nc")
    if os.path.exists(p_sel):
        prior = _open_da(p_sel, "prior_penalty")
    else:
        p_tr = os.path.join(PRIOR_DIR, f"prior_transformer_{ym}.nc")
        da = _open_da(p_tr, None)
        if da is not None:
            arr = da
            if "time" in arr.dims and arr.sizes.get("time", 0) > 0:
                arr = arr.isel(time=0)
            v = np.asarray(arr.values, dtype=float)
            pen = 1.0 - np.clip(v, 0.0, 1.0)
            prior = arr.copy(data=pen)
            prior.name = "prior_penalty"
    # interact
    p_inter = os.path.join(RISK_DIR, f"R_interact_{ym}.nc")
    interact = _open_da(p_inter, "risk")
    return {"risk": risk, "prior": prior, "interact": interact}


def _predictor_from_layers(risk: "xr.DataArray", prior: Optional["xr.DataArray"], interact: Optional["xr.DataArray"]) -> PredictorOutput:
    if "time" not in risk.dims:
        risk = risk.expand_dims({"time": [0]})
    latn = "lat" if "lat" in risk.coords else ("latitude" if "latitude" in risk.coords else None)
    lonn = "lon" if "lon" in risk.coords else ("longitude" if "longitude" in risk.coords else None)
    if not (latn and lonn):
        # Fallback 1: try to reload the original dataset to get coordinates
        try:
            ym = risk.attrs.get("ym") or (risk.encoding.get("source", "").split("_")[-1].replace(".nc","") if hasattr(risk, 'encoding') else None) or ""
            p_risk = os.path.join(RISK_DIR, f"risk_fused_{ym}.nc") if ym else None
            if p_risk and os.path.exists(p_risk):
                with xr.open_dataset(p_risk) as ds_orig:
                    lat_name_orig = next((n for n in ('lat', 'latitude') if (n in ds_orig.coords or n in ds_orig.variables)), None)
                    lon_name_orig = next((n for n in ('lon', 'longitude') if (n in ds_orig.coords or n in ds_orig.variables)), None)
                    if lat_name_orig and lon_name_orig:
                        lat_da = ds_orig[lat_name_orig]
                        lon_da = ds_orig[lon_name_orig]
                        risk = risk.assign_coords({lat_name_orig: lat_da, lon_name_orig: lon_da})
                        latn = lat_name_orig
                        lonn = lon_name_orig
        except Exception:
            pass
    if not (latn and lonn):
        # Fallback 2: construct from canonical grid (1D lat/lon) and meshgrid
        try:
            from ArcticRoute.io.grid_index import _load_rect_grid  # REUSE
            lat1d, lon1d = _load_rect_grid(None)
            Ty = int(risk.sizes.get('y') or risk.shape[-2])
            Tx = int(risk.sizes.get('x') or risk.shape[-1])
            # 安全截断/重复到尺寸匹配
            lat1 = lat1d.astype('float32')
            lon1 = lon1d.astype('float32')
            if lat1.shape[0] != Ty:
                import numpy as _np
                idx_y = _np.linspace(0, max(1, lat1.shape[0]-1), Ty).round().astype(int)
                lat1 = lat1[idx_y]
            if lon1.shape[0] != Tx:
                import numpy as _np
                idx_x = _np.linspace(0, max(1, lon1.shape[0]-1), Tx).round().astype(int)
                lon1 = lon1[idx_x]
            Lon, Lat = np.meshgrid(lon1, lat1)
            risk = risk.assign_coords({"lat": (('y','x'), Lat), "lon": (('y','x'), Lon)})
            latn, lonn = 'lat', 'lon'
        except Exception:
            pass
    if not (latn and lonn):
        raise RuntimeError("risk layer missing lat/lon coordinates")
    latc = np.asarray(risk.coords[latn].values)
    lonc = np.asarray(risk.coords[lonn].values)
    # 规划器期望 lat 为长度 Ty 的一维数组（按 y 维），lon 为长度 Tx 的一维数组（按 x 维）
    Ty = int(risk.sizes.get('y') or risk.shape[-2])
    Tx = int(risk.sizes.get('x') or risk.shape[-1])
    if latc.ndim == 2:
        lat1 = latc[:, 0].astype("float32")
    elif latc.ndim == 1:
        lat1 = latc.astype("float32")
    else:
        lat1 = latc.astype("float32").reshape(-1)
    if lonc.ndim == 2:
        lon1 = lonc[0, :].astype("float32")
    elif lonc.ndim == 1:
        lon1 = lonc.astype("float32")
    else:
        lon1 = lonc.astype("float32").reshape(-1)
    # 对齐长度到 Ty/Tx（必要时近邻重采样）
    def _fit_1d(arr, target_len):
        if arr.shape[0] == target_len:
            return arr
        idx = np.linspace(0, max(1, arr.shape[0]-1), target_len).round().astype(int)
        return arr[idx]
    lat = _fit_1d(lat1, Ty)
    lon = _fit_1d(lon1, Tx)
    # 对齐 prior/interact 到 risk 的 time 维
    if prior is not None and "time" not in prior.dims and "time" in risk.dims:
        prior = prior.expand_dims({"time": risk.coords["time"]})
    if interact is not None and "time" not in interact.dims and "time" in risk.dims:
        interact = interact.expand_dims({"time": risk.coords["time"]})
    return PredictorOutput(risk=risk, corridor=prior, lat=lat, lon=lon, base_time_index=0, accident=interact)


def _save_geojson(path: str, lonlat: Sequence[Tuple[float, float]], props: Dict[str, Any]) -> None:
    feat = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[float(lon), float(lat)] for lon, lat in lonlat]},
        "properties": props,
    }
    data = {"type": "FeatureCollection", "features": [feat]}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    # meta
    meta_path = path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"logical_id": os.path.basename(path), "inputs": list(props.keys())}, f, ensure_ascii=False, indent=2)


def run_scan(*, scenario: Dict[str, Any], ym: str, risk_source: str, risk_agg: str = "mean", alpha: float = 0.95, export_top: int = 3, out_dir: Optional[str] = None, eco: str = "off", w_e: float = 0.0, vessel_class: str = "cargo_iceclass") -> Dict[str, Any]:
    coarse_factor = 4  # Memory optimization
    neighbor8 = False    # Performance optimization

    layers = _load_layers(ym, risk_source, risk_agg=risk_agg, alpha=alpha)
    risk = layers["risk"]
    if risk is None:
        raise FileNotFoundError("risk layer missing for scan")
    prior = layers["prior"]
    interact = layers["interact"]

    # Coarsen layers for performance
    risk_coarse = risk.coarsen(y=coarse_factor, x=coarse_factor, boundary="trim").mean().astype(np.float32)
    prior_coarse = prior.coarsen(y=coarse_factor, x=coarse_factor, boundary="trim").mean().astype(np.float32) if prior is not None else None
    interact_coarse = interact.coarsen(y=coarse_factor, x=coarse_factor, boundary="trim").mean().astype(np.float32) if interact is not None else None

    predictor = _predictor_from_layers(risk_coarse, prior_coarse, interact_coarse)
    planner = AStarGridTimePlanner()

    # Phase M: ECO 开关，构建归一化代价并注入规划器/成本
    eco_norm_da = None
    eco_fuel_da = None
    if str(eco).lower() == "on":
        try:
            eco_fuel_da, _meta = fuel_per_nm_map(ym, vessel_class=vessel_class)
            eco_norm_da = eco_cost_norm(eco_fuel_da)
        except Exception:
            eco_norm_da = None
            eco_fuel_da = None
    if eco_norm_da is not None:
        # 对齐 eco_norm 到 risk 的维度与尺寸（time,y,x）
        try:
            tmpl = risk
            # time 维对齐
            if "time" in tmpl.dims and "time" not in eco_norm_da.dims:
                eco_norm_da = eco_norm_da.expand_dims({"time": tmpl.coords["time"]})
            # 重排维度
            try:
                eco_norm_da = eco_norm_da.transpose(*tmpl.dims, missing_dims="ignore")
            except Exception:
                pass
            # 广播/尺寸对齐
            try:
                eco_norm_da = eco_norm_da.broadcast_like(tmpl)
            except Exception:
                # 兜底：按最近邻索引重采样到 (time,y,x) 尺寸
                Ay = int(eco_norm_da.sizes.get("y", eco_norm_da.shape[-2]))
                Ax = int(eco_norm_da.sizes.get("x", eco_norm_da.shape[-1]))
                Ty = int(tmpl.sizes.get("y", tmpl.shape[-2]))
                Tx = int(tmpl.sizes.get("x", tmpl.shape[-1]))
                import numpy as _np
                iy = _np.linspace(0, max(1, Ay - 1), Ty).round().astype(int)
                ix = _np.linspace(0, max(1, Ax - 1), Tx).round().astype(int)
                if "time" in eco_norm_da.dims:
                    outs = []
                    for t in range(int(eco_norm_da.sizes.get("time", 1))):
                        a = _np.asarray(eco_norm_da.isel(time=t).values)
                        outs.append(a[iy][:, ix])
                    arr = _np.stack(outs, axis=0)
                    eco_norm_da = xr.DataArray(arr, dims=("time","y","x"), coords={"time": tmpl.coords.get("time", _np.arange(arr.shape[0]))})
                else:
                    a = _np.asarray(eco_norm_da.values)
                    arr = a[iy][:, ix]
                    eco_norm_da = xr.DataArray(arr, dims=("y","x"))
        except Exception:
            pass
        try:
            # 注入到 predictor_output（非破坏）
            setattr(predictor, "eco_norm", eco_norm_da)
        except Exception:
            pass
        # 写盘缓存归一化 ECO 栅格
        try:
            os.makedirs(ECO_DIR, exist_ok=True)
            eco_nc = os.path.join(ECO_DIR, f"eco_cost_{ym}.nc")
            ds = eco_norm_da.rename("eco_cost_norm").to_dataset()
            ds.attrs.update({"ym": str(ym), "vessel_class": str(vessel_class), "kind": "eco_cost_norm"})
            ds.to_netcdf(eco_nc)
            # 写 meta
            with open(eco_nc + ".meta.json", "w", encoding="utf-8") as f:
                json.dump({
                    "logical_id": os.path.basename(eco_nc),
                    "inputs": [f"R_ice_eff_{ym}.nc"],
                    "metrics": {"min": float(np.nanmin(eco_norm_da.values)), "max": float(np.nanmax(eco_norm_da.values))},
                }, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    start = tuple(scenario.get("start", [0.0, 0.0]))
    goal = tuple(scenario.get("goal", [0.0, 0.0]))

    weights_spec = (scenario.get("weights") or {})
    combos = list(iter_weight_grid(weights_spec))

    points: List[Dict[str, Any]] = []
    route_refs: List[str] = []

    for idx, w in enumerate(combos):
        # 将权重应用到代价提供者：
        # - w_r → beta（风险强度）
        # - w_c → interact_weight（交互惩罚）
        # - w_p → prior_penalty_weight（先验惩罚）
        beta = float(w.get("w_r", 1.0))
        w_c = float(w.get("w_c", 0.0))
        w_p = float(w.get("w_p", 0.0))
        cost = EnvRiskCostProvider(beta=beta, p_exp=1.0, gamma=0.0, interact_weight=w_c, prior_penalty_weight=w_p)
        # Phase G: 距离权重
        try:
            cost.distance_weight = float(w.get("w_d", 1.0))
        except Exception:
            cost.distance_weight = 1.0
        # Phase M: ECO 权重
        try:
            if eco_norm_da is not None and str(eco).lower() == "on":
                cost.eco_weight = float(w_e)
        except Exception:
            pass
        res = planner.plan(predictor_output=predictor, cost_provider=cost, start_latlon=start, goal_latlon=goal)
        path = [(float(lon), float(lat)) for lon, lat in zip(res.lon_path.tolist(), res.lat_path.tolist())]
        metrics = summarize_route(path, risk=risk, prior_penalty=prior, interact=interact)
        # Phase M: 若启用 ECO，则计算该路径 CO2 总量
        if eco_fuel_da is not None and str(eco).lower() == "on":
            # 读取排放因子
            ef = 3.114
            try:
                eco_cfg_path = os.path.join(REPO_ROOT, "ArcticRoute", "config", "eco.yaml")
                if os.path.exists(eco_cfg_path) and yaml is not None:
                    cfg = yaml.safe_load(open(eco_cfg_path, "r", encoding="utf-8").read()) or {}
                    ef = float(((cfg.get("eco") or {}).get("ef_co2_t_per_t_fuel", 3.114)))
            except Exception:
                pass
            eco_tot = eval_route_eco(path, eco_fuel_da, ef)
            metrics["co2_total_t"] = float(eco_tot.get("co2_total_t", 0.0))
        point = {"idx": idx, "risk_agg": risk_agg, "alpha": alpha, **w, **metrics}
        points.append(point)
        geo_path = os.path.join(ROUTES_DIR, f"route_{ym}_{scenario.get('id','scn')}_cand{idx:02d}.geojson")
        route_refs.append(geo_path)
        props = {"ym": ym, "scenario": scenario.get("id"), "risk_agg": risk_agg, "alpha": float(alpha), **w, **metrics}
        _save_geojson(geo_path, path, props)

    # Pareto & representatives
    # Phase M: Pareto 维度扩展（开启 eco 时加入 CO2）
    keys = ["risk_integral", "distance_km", "congest_integral"]
    if eco_fuel_da is not None and str(eco).lower() == "on":
        keys.append("co2_total_t")
    nd_idx = nondominated(points, keys=tuple(keys))
    reps = pick_representatives([points[i] for i in nd_idx])
    # Map back to global indices
    rep_global = {k: (nd_idx[v] if v >= 0 and v < len(nd_idx) else -1) for k, v in reps.items()}

    # Export representatives copies
    labels = {"safe": rep_global.get("safe"), "balanced": rep_global.get("balanced"), "efficient": rep_global.get("efficient")}
    for lab, gi in labels.items():
        if gi is None or gi < 0:
            continue
        src = route_refs[gi]
        dst = os.path.join(ROUTES_DIR, f"route_{ym}_{scenario.get('id','scn')}_{lab}.geojson")
        try:
            with open(src, "r", encoding="utf-8") as f:
                data = json.load(f)
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            with open(dst + ".meta.json", "w", encoding="utf-8") as f:
                json.dump({"logical_id": os.path.basename(dst), "src": os.path.basename(src)}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # Write front & summary
    os.makedirs(REPORT_DIR, exist_ok=True)
    front_path = os.path.join(REPORT_DIR, f"pareto_front_{ym}_{scenario.get('id','scn')}.json")
    summary_path = os.path.join(REPORT_DIR, f"summary_{ym}_{scenario.get('id','scn')}.json")

    payload = {
        "ym": ym,
        "scenario": scenario.get("id"),
        "points": points,
        "pareto": {"indices": nd_idx},
        "representatives": rep_global,
        "routes": {"candidates": route_refs},
    }
    with open(front_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(front_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"logical_id": os.path.basename(front_path), "inputs": [ym, scenario.get("id")]}, f, ensure_ascii=False, indent=2)

    summary = {"ym": ym, "scenario": scenario.get("id"), "representatives": labels, "pareto_size": len(nd_idx)}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(summary_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"logical_id": os.path.basename(summary_path)}, f, ensure_ascii=False, indent=2)

    # Simple HTML
    html_path = os.path.join(REPORT_DIR, f"pareto_{ym}_{scenario.get('id','scn')}.html")
    try:
        xs = [float(p.get("distance_km", 0.0)) for p in points]
        ys = [float(p.get("risk_integral", 0.0)) for p in points]
        cols = [float(p.get("congest_integral", 0.0)) for p in points]
        co2s = [float(p.get("co2_total_t", np.nan)) for p in points]
        # 颜色编码：按 CO2 百分位归一到 [0,1] → 从绿到红
        try:
            arr = np.array([c for c in co2s if np.isfinite(c)], dtype=float)
            if arr.size > 0:
                lo = float(np.quantile(arr, 0.05))
                hi = float(np.quantile(arr, 0.95))
            else:
                lo, hi = 0.0, 1.0
        except Exception:
            lo, hi = 0.0, 1.0
        def color_for(c: float) -> str:
            if not np.isfinite(c):
                return "#888888"
            t = 0.0 if hi <= lo else max(0.0, min(1.0, (c - lo) / (hi - lo)))
            # t=0 绿色，t=1 红色（简单线性渐变）
            r = int(255 * t)
            g = int(200 * (1.0 - t))
            b = 80
            return f"#{r:02x}{g:02x}{b:02x}"
        rows = []
        for i, p in enumerate(points):
            co2 = co2s[i]
            color = color_for(co2)
            rows.append({"distance_km": xs[i], "risk": ys[i], "congest": cols[i], "co2_t": co2, "color": color})
        html = [
            "<html><head><meta charset='utf-8'><title>Pareto</title></head><body>",
            f"<h1>Pareto Front {ym} / {scenario.get('id')}</h1>",
            f"<p>Total candidates: {len(points)}; Pareto size: {len(nd_idx)}</p>",
            "<h2>Points (distance vs risk)</h2>",
            "<style> .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;} table{border-collapse:collapse;} td,th{border:1px solid #ccc;padding:4px;} </style>",
            "<table><tr><th>#</th><th>Dist(km)</th><th>Risk</th><th>Congest</th><th>CO₂ (t)</th><th>Color</th></tr>",
        ]
        for i, r in enumerate(rows[:200]):
            co2_str = "" if not np.isfinite(r['co2_t']) else f"{r['co2_t']:.2f}"
            html.append(f"<tr><td>{i}</td><td>{r['distance_km']:.1f}</td><td>{r['risk']:.2f}</td><td>{r['congest']:.2f}</td><td>{co2_str}</td><td><span class='dot' style='background:{r['color']}'></span></td></tr>")
        html.extend([
            "</table>",
            "<p>Legend: color encodes CO₂ (green=low, red=high) based on 5–95% quantiles.</p>",
            "</body></html>",
        ])
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
        with open(html_path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump({"logical_id": os.path.basename(html_path), "has_co2": any(np.isfinite(c) for c in co2s)}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return {"front": front_path, "summary": summary_path, "html": html_path, "points": len(points), "pareto": len(nd_idx)}


__all__ = ["run_scan"]

