from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

from ArcticRoute.core.planners.astar_grid_time import AStarGridTimePlanner
from ArcticRoute.core.interfaces import PredictorOutput
from ArcticRoute.core.cost.env_risk_cost import EnvRiskCostProvider
from ArcticRoute.core.cost.aggregators import beta_from_mean_var  # REUSE
from ArcticRoute.core.route.metrics import summarize_route
from ArcticRoute.core.route.scan import _open_da  # REUSE helper (文件内函数)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RISK_DIR = os.path.join(REPO_ROOT, "ArcticRoute", "data_processed", "risk")
PRIOR_DIR = os.path.join(REPO_ROOT, "ArcticRoute", "data_processed", "prior")
ROUTES_DIR = os.path.join(REPO_ROOT, "ArcticRoute", "data_processed", "routes")
REPORT_DIR = os.path.join(REPO_ROOT, "ArcticRoute", "reports", "d_stage", "phaseI")


def _load_layers_with_var(ym: str, risk_source: str) -> Dict[str, Optional["xr.DataArray"]]:
    # 与 scan._load_layers 相同，但对于 fused 返回 Risk 与 RiskVar（若存在）
    if xr is None:
        raise RuntimeError("xarray required")
    if risk_source == "fused":
        p_risk = os.path.join(RISK_DIR, f"risk_fused_{ym}.nc")
        risk, ds = _open_da(p_risk, "Risk", return_ds=True)
        var = None
        if ds is not None:
            for cand in ("RiskVar", "risk_var", "risk_variance", "Var", "variance"):
                if cand in ds:
                    var = ds[cand]
                    break
            try:
                ds.close()
            except Exception:
                pass
    else:
        # ice-only 回退，无方差
        risk = None
        for p in [os.path.join(RISK_DIR, f"R_ice_eff_{ym}.nc"), os.path.join(RISK_DIR, f"risk_ice_{ym}.nc")]:
            r = _open_da(p, None)
            if r is not None:
                risk = r
                break
        var = None
    # prior penalty 与 interact
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
    inter = _open_da(os.path.join(RISK_DIR, f"R_interact_{ym}.nc"), "risk")
    out: Dict[str, Optional[xr.DataArray]] = {"risk": risk, "risk_var": var, "prior": prior, "interact": inter}
    return out


def _predictor_for_risk(risk: "xr.DataArray", prior: Optional["xr.DataArray"], inter: Optional["xr.DataArray"]) -> PredictorOutput:
    if "time" not in risk.dims:
        risk = risk.expand_dims({"time": [0]})
    if prior is not None and "time" not in prior.dims and "time" in risk.dims:
        prior = prior.expand_dims({"time": risk.coords["time"]})
    if inter is not None and "time" not in inter.dims and "time" in risk.dims:
        inter = inter.expand_dims({"time": risk.coords["time"]})
    latn = "lat" if "lat" in risk.coords else ("latitude" if "latitude" in risk.coords else None)
    lonn = "lon" if "lon" in risk.coords else ("longitude" if "longitude" in risk.coords else None)
    if not (latn and lonn):
        raise RuntimeError("risk layer missing lat/lon")
    lat = np.asarray(risk.coords[latn].values, dtype="float32")
    lon = np.asarray(risk.coords[lonn].values, dtype="float32")
    return PredictorOutput(risk=risk, corridor=prior, accident=inter, lat=lat, lon=lon, base_time_index=0)


def _sample_beta_surfaces(risk: "xr.DataArray", risk_var: Optional["xr.DataArray"], k: int, rng: Optional[np.random.Generator] = None) -> List["xr.DataArray"]:
    rng = rng or np.random.default_rng(20231101)
    r = np.asarray(risk.values, dtype=float)
    if risk_var is None:
        # 无方差：返回 k 份拷贝
        return [risk.copy(deep=False) for _ in range(k)]
    v = np.asarray(risk_var.values, dtype=float)
    a, b = beta_from_mean_var(r, v)
    # 采样 shape: (k, ...)
    if np.ndim(a) == 0:
        samples = rng.beta(float(a), float(b), size=(k,))
    else:
        samples = rng.beta(a[None, ...], b[None, ...], size=(k,) + a.shape)
    out: List[xr.DataArray] = []
    for i in range(k):
        da = risk.copy(data=samples[i].astype("float32"))
        da.name = "Risk"
        out.append(da)
    return out


def _expected_shortfall(values: np.ndarray, alpha: float) -> float:
    # ES_alpha = mean of worst (1-alpha) tail
    if values.size == 0:
        return float("nan")
    a = float(np.clip(alpha, 0.0, 1.0))
    if a >= 1.0:
        return float(np.mean(values))
    q = np.quantile(values, a)
    tail = values[values >= q]
    if tail.size == 0:
        tail = np.array([q], dtype=float)
    return float(np.mean(tail))


def run_robust(*, scenario: Dict[str, Any], ym: str, risk_source: str, samples: int = 16, alpha: float = 0.9, out_dir: Optional[str] = None) -> Dict[str, Any]:
    layers = _load_layers_with_var(ym, risk_source)
    risk = layers["risk"]
    if risk is None:
        raise FileNotFoundError("risk layer missing for robust routing")
    risk_var = layers.get("risk_var")
    prior = layers.get("prior")
    inter = layers.get("interact")

    # 生成 K 个风险面
    K = int(max(1, samples))
    surfaces = _sample_beta_surfaces(risk, risk_var, K)

    planner = AStarGridTimePlanner()
    start = tuple(scenario.get("start", [0.0, 0.0]))
    goal = tuple(scenario.get("goal", [0.0, 0.0]))

    # 成本：使用基础设置（beta=1,w_d=1），其余按场景权重也可扩展
    cost = EnvRiskCostProvider(beta=1.0, p_exp=1.0, gamma=0.0, interact_weight=0.0, prior_penalty_weight=0.0)

    candidates: List[Dict[str, Any]] = []
    routes_geo: List[str] = []

    # 为每个样本规划一条路线
    for k, r_da in enumerate(surfaces):
        pred = _predictor_for_risk(r_da, prior, inter)
        res = planner.plan(predictor_output=pred, cost_provider=cost, start_latlon=start, goal_latlon=goal)
        path = [(float(lon), float(lat)) for lon, lat in zip(res.lon_path.tolist(), res.lat_path.tolist())]
        # 在所有样本上评估该路径的风险积分，计算 ES@alpha
        vals = []
        for s_da in surfaces:
            vals.append(summarize_route(path, risk=s_da, prior_penalty=None, interact=None).get("risk_integral", 0.0))
        vals_arr = np.asarray(vals, dtype=float)
        es = _expected_shortfall(vals_arr, alpha=float(alpha))
        metrics = summarize_route(path, risk=r_da, prior_penalty=prior, interact=inter)
        metrics["es_alpha"] = float(alpha)
        metrics["es_value"] = float(es)
        metrics["samples"] = K
        # 导出临时候选（可选）
        geo_path = os.path.join(ROUTES_DIR, f"route_{ym}_{scenario.get('id','scn')}_robust_cand{k:02d}.geojson")
        props = {"ym": ym, "scenario": scenario.get("id"), "k": k, **metrics}
        _save_geojson(geo_path, path, props)
        routes_geo.append(geo_path)
        candidates.append({"idx": k, "path": path, "es": es, "metrics": metrics, "geo": geo_path})

    # 选择 ES 最小的候选作为鲁棒路线
    best = min(candidates, key=lambda d: d["es"]) if candidates else None
    if best is None:
        raise RuntimeError("robust routing failed: no candidates")

    # 计算路径上节点风险的跨样本方差统计
    # 将路径映射到栅格索引并抽取每样本对应值（近邻取样）
    try:
        latn = "lat" if "lat" in risk.coords else "latitude"
        lonn = "lon" if "lon" in risk.coords else "longitude"
        lat = np.asarray(risk.coords[latn].values)
        lon = np.asarray(risk.coords[lonn].values)
        idx_pairs: List[Tuple[int, int]] = []
        for (xlon, xlat) in best["path"]:
            iy = int(np.abs(lat - xlat).argmin())
            ix = int(np.abs(lon - xlon).argmin())
            idx_pairs.append((iy, ix))
        vals_mat = []
        for s_da in surfaces:
            arr = np.asarray(s_da.values, dtype=float)
            if arr.ndim > 2:
                arr = np.squeeze(arr)
                if arr.ndim > 2:
                    axes = tuple(range(0, arr.ndim - 2))
                    arr = arr.mean(axis=axes)
            seq = [float(np.nan_to_num(arr[iy, ix], nan=0.0)) for (iy, ix) in idx_pairs]
            vals_mat.append(seq)
        vals_mat = np.asarray(vals_mat, dtype=float)  # [K, L]
        per_node_std = np.nanstd(vals_mat, axis=0)
        disp = {"std_mean": float(np.nanmean(per_node_std)), "std_max": float(np.nanmax(per_node_std)), "L": int(per_node_std.size)}
    except Exception:
        disp = {"std_mean": float("nan"), "std_max": float("nan"), "L": 0}

    # 保存最终鲁棒路线
    out_dir2 = out_dir or (os.path.join(REPO_ROOT, "ArcticRoute", "reports", "d_stage", "phaseI"))
    os.makedirs(out_dir2, exist_ok=True)
    robust_path = os.path.join(out_dir2, f"route_{ym}_{scenario.get('id','scn')}_robust.geojson")
    props = {"ym": ym, "scenario": scenario.get("id"), "risk_agg": "robust-es", "alpha": float(alpha), "samples": K, **best["metrics"], **{f"disp_{k}": v for k, v in disp.items()}}
    _save_geojson(robust_path, best["path"], props)

    return {"ym": ym, "scenario": scenario.get("id"), "alpha": float(alpha), "samples": K, "route": robust_path, "candidates": [c["geo"] for c in candidates], "dispersion": disp}


def _save_geojson(path: str, lonlat: Sequence[Tuple[float, float]], props: Dict[str, Any]) -> None:
    feat = {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[float(lon), float(lat)] for lon, lat in lonlat]}, "properties": props}
    data = {"type": "FeatureCollection", "features": [feat]}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    with open(path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"logical_id": os.path.basename(path), "inputs": [props.get("ym"), props.get("scenario")]}, f, ensure_ascii=False, indent=2)


__all__ = ["run_robust"]



