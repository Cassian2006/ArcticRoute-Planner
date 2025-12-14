"""D-02: 风险融合（分位数归一 + 稳健融合）

实现 fuse_risk(ym, weights/norm/missing)：
- Risk = alpha·R_ice + beta·R_wave + gamma·R_acc
- 对每层按 q01–q99 分位数做归一（逐时间帧）
- 缺层：按剩余层对权重重归一；全缺：报错（不写盘）
- 输出尺寸/坐标与基准层一致（以首个可用层为基准；不做重投影，尺寸不一致的层将被跳过并告警）
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

from ArcticRoute.core.risk.fuse_prep import find_layer_paths  # 复用发现策略（含 ice_cost 回退）

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

RISK_DIR = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "risk")
MERGED_DIR = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "ice_forecast", "merged")


def _pick_var(ds: "xr.Dataset", kind: str) -> Tuple[Optional[str], List[str], Optional[str]]:
    """返回 (var_name, issues, source_hint)。
    - kind in {ice,wave,acc}
    - 对 ice: 若无 R_ice 但有 ice_cost → 使用 ice_cost，source_hint="ice_cost"
    - 其他：优先 R_wave/R_acc，其次 Risk/risk/等模糊匹配
    """
    issues: List[str] = []
    src: Optional[str] = None
    if kind == "ice":
        if "R_ice" in ds:
            return "R_ice", issues, None
        if "ice_cost" in ds:
            issues.append("回退映射 ice_cost→R_ice (fuse)")
            src = "ice_cost"
            return "ice_cost", issues, src
    # 通用优先序
    cand = {
        "ice": ["R_ice", "ice", "risk_ice", "Risk"],
        "wave": ["R_wave", "wave", "risk_wave", "Risk"],
        "acc": ["R_acc", "acc", "accident", "risk_acc", "Risk"],
    }[kind]
    for name in cand:
        if name in ds:
            return name, issues, src
    # 模糊
    names = list(ds.data_vars.keys())
    lower = {n.lower(): n for n in names}
    keys = {
        "ice": ["r_ice", "ice"],
        "wave": ["r_wave", "wave"],
        "acc": ["r_acc", "acc", "accident"],
    }[kind]
    for key in keys:
        for k, v in lower.items():
            if key in k:
                return v, issues, src
    issues.append("未识别到合适变量")
    return None, issues, src


def _quantile_norm(da: "xr.DataArray") -> "xr.DataArray":
    """按 q01–q99 将 da 线性映射到 [0,1]，逐 time 帧处理；若无 time 维则整体处理。"""
    if "time" in da.dims and int(da.sizes.get("time", 0)) > 0:
        out = []
        for t in range(int(da.sizes["time"])):
            a = da.isel(time=t)
            arr = np.asarray(a.values, dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                q01 = float(np.nanpercentile(finite, 1))
                q99 = float(np.nanpercentile(finite, 99))
            else:
                q01 = 0.0
                q99 = 0.0
            if not np.isfinite(q01) or not np.isfinite(q99) or q99 <= q01:
                normed = np.zeros_like(arr, dtype=float)
            else:
                normed = (arr - q01) / (q99 - q01)
                normed = np.clip(normed, 0.0, 1.0)
                normed[~np.isfinite(arr)] = np.nan
            out.append(xr.DataArray(normed, dims=a.dims, coords=a.coords, attrs=a.attrs))
        return xr.concat(out, dim="time")
    else:
        arr = np.asarray(da.values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            q01 = float(np.nanpercentile(finite, 1))
            q99 = float(np.nanpercentile(finite, 99))
        else:
            q01 = 0.0
            q99 = 0.0
        if not np.isfinite(q01) or not np.isfinite(q99) or q99 <= q01:
            normed = np.zeros_like(arr, dtype=float)
        else:
            normed = (arr - q01) / (q99 - q01)
            normed = np.clip(normed, 0.0, 1.0)
            normed[~np.isfinite(arr)] = np.nan
        return xr.DataArray(normed, dims=da.dims, coords=da.coords, attrs=da.attrs)


def fuse_risk(
    ym: str,
    *,
    weights: Optional[Dict[str, float]] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    norm: str = "quantile",
    missing: str = "skip_and_warn",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """融合风险。

    返回 payload，若 dry_run=False 则写盘 risk_fused_<ym>.nc。
    - weights 可由 alpha/beta/gamma 或 weights 字典提供（alpha 优先级与 weights 合并，alpha/beta/gamma 覆盖同名）。
    - norm 目前仅支持 "quantile"。
    - missing: "skip_and_warn"（仅支持该策略）。
    """
    if xr is None:
        raise RuntimeError("xarray is required")

    # 组合权重
    w = {"alpha": 0.6, "beta": 0.2, "gamma": 0.2}
    if isinstance(weights, dict):
        for k in ("alpha", "beta", "gamma"):
            if k in weights and isinstance(weights[k], (int, float)):
                w[k] = float(weights[k])
    if alpha is not None:
        w["alpha"] = float(alpha)
    if beta is not None:
        w["beta"] = float(beta)
    if gamma is not None:
        w["gamma"] = float(gamma)

    # 找层
    paths = find_layer_paths(ym)
    # Phase F: 优先使用 R_ice_eff_<ym>.nc；可选接入 R_interact_<ym>.nc
    risk_dir = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "risk")
    p_ice_eff = os.path.join(risk_dir, f"R_ice_eff_{ym}.nc")
    if os.path.exists(p_ice_eff):
        paths["ice"] = p_ice_eff
    p_interact = os.path.join(risk_dir, f"R_interact_{ym}.nc")

    info: Dict[str, Any] = {"ym": ym, "paths": paths, "used": {}, "skipped": []}

    # 打开并选择变量
    datasets: Dict[str, xr.Dataset] = {}
    vars_map: Dict[str, str] = {}
    source_hint: Dict[str, Optional[str]] = {}
    issues: List[str] = []

    for kind in ("ice", "wave", "acc"):
        p = paths.get(kind)
        if not (isinstance(p, str) and os.path.exists(p)):
            continue
        try:
            ds = xr.open_dataset(p)
        except Exception as e:
            issues.append(f"无法打开 {kind}: {p} · {e}")
            continue
        # REUSE: 若为护航折减产物 R_ice_eff_<ym>.nc，变量名为 'risk'
        base = os.path.basename(p or "") if isinstance(p, str) else ""
        if kind == "ice" and isinstance(p, str) and base.startswith("R_ice_eff_"):
            if "risk" in ds:
                var = "risk"
                iss = []
                src = "escort"
            else:
                var, iss, src = _pick_var(ds, kind)
        else:
            var, iss, src = _pick_var(ds, kind)
        issues.extend([f"{kind}:{m}" for m in iss])
        if var is None or var not in ds:
            issues.append(f"{kind}: 未找到有效变量")
            try:
                ds.close()
            except Exception:
                pass
            continue
        datasets[kind] = ds
        vars_map[kind] = var
        source_hint[kind] = src
        info["used"][kind] = {"path": p, "var": ("R_ice(ice_cost)" if (kind == "ice" and src == "ice_cost") else var)}

    # 选择基准层（ice>wave>acc 顺序首个可用）
    ref_kind = next((k for k in ("ice", "wave", "acc") if k in datasets), None)
    if ref_kind is None:
        raise RuntimeError("无可用层：无法进行融合")
    ref_ds = datasets[ref_kind]
    # 基准 DataArray（用于坐标/维度）
    ref_da = ref_ds[vars_map[ref_kind]]

    # 空间网格一致性：若有经纬度坐标，则需数值一致；若任一层与参考层空间尺寸(y/x)不一致 → FAIL
    def _latlon(ds: xr.Dataset) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        latn = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else None)
        lonn = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
        if latn and lonn:
            return np.asarray(ds[latn].values), np.asarray(ds[lonn].values)
        return None, None

    ref_lat, ref_lon = _latlon(ref_ds)

    Ty = ref_da.sizes.get("y") or ref_da.sizes.get("lat") or ref_da.sizes.get("latitude")
    Tx = ref_da.sizes.get("x") or ref_da.sizes.get("lon") or ref_da.sizes.get("longitude")

    for k2, ds2 in datasets.items():
        da2 = ds2[vars_map[k2]]
        y2 = da2.sizes.get("y") or da2.sizes.get("lat") or da2.sizes.get("latitude")
        x2 = da2.sizes.get("x") or da2.sizes.get("lon") or da2.sizes.get("longitude")
        if (Ty is not None and int(y2 or -1) != int(Ty)) or (Tx is not None and int(x2 or -1) != int(Tx)):
            for ds in datasets.values():
                try:
                    ds.close()
                except Exception:
                    pass
            raise RuntimeError(f"空间网格尺寸不一致：ref={Ty}x{Tx} vs {k2}={y2}x{x2}")
        lat2, lon2 = _latlon(ds2)
        if ref_lat is not None and lat2 is not None:
            try:
                if not (np.allclose(ref_lat, lat2, equal_nan=True) and np.allclose(ref_lon, lon2, equal_nan=True)):
                    for ds in datasets.values():
                        try:
                            ds.close()
                        except Exception:
                            pass
                    raise RuntimeError(f"空间坐标(lat/lon)不一致：ref vs {k2}")
            except Exception:
                for ds in datasets.values():
                    try:
                        ds.close()
                    except Exception:
                        pass
                raise RuntimeError(f"空间坐标比较失败：ref vs {k2}")

    # 时间轴对齐：若存在 time 维的不一致，取交集并对齐；若交集为空则 FAIL
    time_axes: List[np.ndarray] = []
    for k2, ds2 in datasets.items():
        da2 = ds2[vars_map[k2]]
        if "time" in da2.dims:
            try:
                time_axes.append(np.asarray(da2["time"].values))
            except Exception:
                pass
    global_time: Optional[np.ndarray] = None
    if time_axes:
        inter = time_axes[0]
        for arr in time_axes[1:]:
            inter = np.intersect1d(inter, arr)
        if inter.size == 0:
            for ds in datasets.values():
                try:
                    ds.close()
                except Exception:
                    pass
            raise RuntimeError("时间轴无交集，无法对齐")
        global_time = inter
        if any((len(ax) != len(inter)) for ax in time_axes):
            issues.append(f"WARN: 时间轴不一致，已自动对齐至交集共 {inter.size} 帧")

    # 构建各层的归一化 DataArray（先对齐时间轴，再归一）；若参考无 time 且存在 global_time，则将静态层广播到该时间轴
    normed: Dict[str, xr.DataArray] = {}
    for kind, ds in datasets.items():
        da = ds[vars_map[kind]]
        # 时间对齐
        if global_time is not None:
            if "time" in da.dims:
                da = da.sel(time=global_time)
            else:
                # 广播静态层
                da = da.expand_dims({"time": global_time})  # type: ignore
        # 归一
        if norm == "quantile":
            nda = _quantile_norm(da)
        else:
            nda = da.astype(float)
        normed[kind] = nda

    # 重归一权重
    present = [k for k in ("ice", "wave", "acc") if k in normed]
    if not present:
        # 全部缺或被跳过
        for ds in datasets.values():
            try:
                ds.close()
            except Exception:
                pass
        raise RuntimeError("融合输入全缺或尺寸不一致导致全被跳过")
    w_map = {"ice": float(w["alpha"]), "wave": float(w["beta"]), "acc": float(w["gamma"])}
    s = sum(w_map[k] for k in present)
    if s <= 0:
        # 若提供的在场层权重和为0，则均分
        w_eff = {k: 1.0 / len(present) for k in present}
    else:
        w_eff = {k: (w_map[k] / s) for k in present}

    # 融合
    combo = None
    for k in present:
        term = normed[k] * float(w_eff[k])
        combo = term if combo is None else (combo + term)

    # Phase F: 可选接入 R_interact（不改动 alpha/beta/gamma 归一，仅作附加项，开关来自 config/runtime.yaml）
    iw = 0.0
    interact_used = False
    try:
        cfg_path = os.path.join(os.getcwd(), "ArcticRoute", "config", "runtime.yaml")
        if yaml is not None and os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            iw = float(cfg.get("behavior", {}).get("interact_weight", 0.0) or 0.0)
    except Exception:
        iw = 0.0
    if iw > 0.0 and os.path.exists(p_interact):
        try:
            ds_int = xr.open_dataset(p_interact)
            # 变量名固定 risk
            if "risk" in ds_int:
                da_i = ds_int["risk"]
                # 对齐时间
                if global_time is not None:
                    if "time" in da_i.dims:
                        da_i = da_i.sel(time=global_time)
                    else:
                        da_i = da_i.expand_dims({"time": global_time})  # type: ignore
                nda_i = _quantile_norm(da_i)
                combo = combo + (nda_i * float(iw))
                interact_used = True
        except Exception:
            pass
        finally:
            try:
                ds_int.close()
            except Exception:
                pass

    # 输出数据集，采用基准坐标
    risk_da = combo.rename("Risk")
    risk_da.attrs.update({"long_name": "Fused risk", "norm": norm})
    out_ds = risk_da.to_dataset()

    # attrs（全局）
    run_id = time.strftime("%Y%m%dT%H%M%S")
    sources = {k: info["used"][k] for k in present if k in info.get("used", {})}
    
    # 方差过低警告
    suspect_flag = False
    if risk_da.std() < 1e-6:
        suspect_flag = True
        issues.append("WARNING: Fused risk has near-zero variance.")

    out_ds.attrs.update({
        "ym": str(ym),
        "norm": str(norm),
        "missing": str(missing),
        "weights": str({"alpha": w["alpha"], "beta": w["beta"], "gamma": w["gamma"]}),
        "weights_effective": str({k: float(w_eff[k]) for k in present}),
        "sources": str(sources),
        "run_id": run_id,
        "suspect": str(suspect_flag),
    })

    # 写盘
    os.makedirs(RISK_DIR, exist_ok=True)
    out_path = os.path.join(RISK_DIR, f"risk_fused_{ym}.nc")

    payload = {
        "ym": ym,
        "out": out_path,
        "present_layers": present,
        "weights": {"alpha": w["alpha"], "beta": w["beta"], "gamma": w["gamma"]},
        "weights_effective": {k: float(w_eff[k]) for k in present},
        "sources": sources,
        "issues": issues,
        "dry_run": bool(dry_run),
    }

    if dry_run:
        for ds in datasets.values():
            try:
                ds.close()
            except Exception:
                pass
        return payload

    try:
        out_ds.to_netcdf(out_path)
    finally:
        for ds in datasets.values():
            try:
                ds.close()
            except Exception:
                pass
        try:
            out_ds.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    return payload

def debug_risk_fusion(ym: str) -> Dict[str, Any]:
    """读取某月的风险输入层和 risk_fused_*.nc，并打印详细统计信息。"""
    if xr is None:
        raise RuntimeError("xarray is required")

    # 1. 加载输入层
    paths = find_layer_paths(ym)
    p_ice_eff = os.path.join(RISK_DIR, f"R_ice_eff_{ym}.nc")
    if os.path.exists(p_ice_eff):
        paths["ice"] = p_ice_eff

    inputs = {}
    for kind, p in paths.items():
        if p and os.path.exists(p):
            try:
                ds = xr.open_dataset(p)
                base_name = os.path.basename(p)
                if kind == "ice" and base_name.startswith("R_ice_eff_") and "risk" in ds:
                    var = "risk"
                else:
                    var, _, _ = _pick_var(ds, kind)
                if var:
                    inputs[kind] = ds[var]
            except Exception as e:
                print(f"[WARN] Failed to load input {kind}: {e}")

    # 2. 加载融合层
    fused_path = os.path.join(RISK_DIR, f"risk_fused_{ym}.nc")
    fused_da = None
    if os.path.exists(fused_path):
        try:
            ds_fused = xr.open_dataset(fused_path)
            var_name = "risk" if "risk" in ds_fused else ("Risk" if "Risk" in ds_fused else (list(ds_fused.data_vars)[0] if ds_fused.data_vars else None))
            if var_name:
                fused_da = ds_fused[var_name]
            else:
                print("[WARN] Could not find a data variable in fused risk file.")
        except Exception as e:
            print(f"[WARN] Failed to load fused risk: {e}")

    # 3. 打印统计信息
    results = {"inputs": {}, "fused": {}}
    print(f"--- Risk Fusion Debug for {ym} ---")

    def get_stats(da: xr.DataArray, name: str):
        arr = da.values.astype(float)
        finite_mask = np.isfinite(arr)
        finite_arr = arr[finite_mask]
        stats = {
            "min": float(np.min(finite_arr)) if finite_arr.size > 0 else 'nan',
            "max": float(np.max(finite_arr)) if finite_arr.size > 0 else 'nan',
            "mean": float(np.mean(finite_arr)) if finite_arr.size > 0 else 'nan',
            "std": float(np.std(finite_arr)) if finite_arr.size > 0 else 'nan',
            "non_zero_ratio": float(np.count_nonzero(finite_arr) / finite_arr.size) if finite_arr.size > 0 else 0.0,
            "nan_ratio": float(np.count_nonzero(np.isnan(arr)) / arr.size) if arr.size > 0 else 0.0,
        }
        print(f"  Layer: {name}")
        for k, v in stats.items():
            print(f"    {k}: {v:.6f}")
        return stats

    for name, da in inputs.items():
        results["inputs"][name] = get_stats(da, name)

    if fused_da is not None:
        results["fused"] = get_stats(fused_da, "risk_fused")
        # 4. 计算相关系数
        print("  Correlations with fused risk:")
        correlations = {}
        # 对齐并展平
        fused_flat = fused_da.values.flatten()
        valid_fused_mask = np.isfinite(fused_flat)

        for name, da_in in inputs.items():
            try:
                # 确保输入层与融合层对齐
                da_aligned = da_in.interp_like(fused_da, method="nearest")
                in_flat = da_aligned.values.flatten()
                valid_mask = valid_fused_mask & np.isfinite(in_flat)
                
                if np.sum(valid_mask) > 1:
                    corr = np.corrcoef(fused_flat[valid_mask], in_flat[valid_mask])[0, 1]
                    correlations[name] = float(corr)
                    print(f"    {name}: {corr:.6f}")
                else:
                    correlations[name] = 'nan'
                    print(f"    {name}: Not enough valid data to compute correlation")
            except Exception as e:
                correlations[name] = 'error'
                print(f"    {name}: Failed to compute correlation - {e}")
        results["correlations"] = correlations

    return results

__all__ = ["fuse_risk", "debug_risk_fusion"]