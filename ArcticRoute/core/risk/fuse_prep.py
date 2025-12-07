"""D-Prep-01: 风险融合前置自检与配置落盘（不执行融合计算）

提供以下函数（签名固定，供 D 阶段复用）：
- find_layer_paths(ym)
- check_dims_coords_equal(paths)
- summarize_layer(path, var_hint=None)
- recommend_weights(stats)
- build_fuse_config(ym, spec, weights)
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from ArcticRoute.cache.index_util import register_artifact

# ---- helpers ----

RISK_DIR = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "risk")
MERGED_DIR = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "ice_forecast", "merged")
REPORT_DIR = os.path.join(os.getcwd(), "reports", "d_stage")
CONFIG_DIR = os.path.join(os.getcwd(), "ArcticRoute", "config")


def _looks_like(path: Optional[str]) -> bool:
    return isinstance(path, str) and bool(path) and os.path.exists(path)


def _pick_var_name(ds: "xr.Dataset", hint: Optional[str]) -> Tuple[Optional[str], List[str]]:
    issues: List[str] = []
    if hint and hint in ds:
        return hint, issues
    # 优先常规名
    for name in ("R_ice", "R_wave", "R_acc", "Risk", "ice_cost"):
        if name in ds:
            return name, issues
    # 模糊匹配：按变量名包含关键字
    names = list(ds.data_vars.keys())
    lower = {n.lower(): n for n in names}
    for key in ("r_ice", "ice", "wave", "r_wave", "acc", "accident", "risk"):
        for k, v in lower.items():
            if key in k:
                return v, issues
    issues.append("未能可靠识别变量名，使用首个变量")
    if names:
        return names[0], issues
    return None, ["数据集中无变量可用"]


# ---- public APIs ----

def find_layer_paths(ym: str) -> Dict[str, Optional[str]]:
    """返回 {'ice': path or None, 'wave': path or None, 'acc': path or None}。
    搜索优先级：
    1) ArcticRoute/data_processed/risk_*/risk_*_<ym>.nc（分目录）
    2) ArcticRoute/data_processed/risk/risk_*_<ym>.nc（汇总目录）
    3) 若 ice 仍缺失，回退 ArcticRoute/data_processed/ice_forecast/merged/ice_cost_<ym>.nc
    """
    base_dp = os.path.join(os.getcwd(), "ArcticRoute", "data_processed")
    # 候选通配
    candidates = [
        os.path.join(base_dp, f"risk*", f"risk_*_{ym}.nc"),
        os.path.join(RISK_DIR, f"risk_*_{ym}.nc"),
    ]
    found: Dict[str, Optional[str]] = {"ice": None, "wave": None, "acc": None}

    import glob as _glob  # 局部导入，避免顶层依赖
    for pat in candidates:
        for p in sorted(_glob.glob(pat)):
            name = os.path.basename(p)
            low = name.lower()
            if (found["ice"] is None) and ("ice" in low) and ("accident" not in low):
                if os.path.exists(p):
                    found["ice"] = p
                    continue
            if (found["wave"] is None) and ("wave" in low):
                if os.path.exists(p):
                    found["wave"] = p
                    continue
            if (found["acc"] is None) and ("accident" in low or low.startswith("risk_acc_")):
                if os.path.exists(p):
                    found["acc"] = p
                    continue

    # 回退：ice → merged/ice_cost_<ym>.nc
    if found["ice"] is None:
        fallback = os.path.join(MERGED_DIR, f"ice_cost_{ym}.nc")
        if os.path.exists(fallback):
            found["ice"] = fallback

    return found


def check_dims_coords_equal(paths: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """校验 (time,y,x)、lat/lon 一致性；返回 {'ok': bool, 'issues': [...], 'spec': {...}}。
    规则：
    - 所有存在的层：time/y/x 维度长度完全一致；
    - 若 time 仅不一致 → WARN（建议 align_time）；
    - 若 y/x 不一致 → FAIL；
    - lat/lon 坐标：若存在则要求数值一致（允许无经纬坐标）。
    """
    if xr is None:
        raise RuntimeError("xarray required for check")
    ds_list: List[Tuple[str, xr.Dataset]] = []
    issues: List[str] = []
    for k in ("ice", "wave", "acc"):
        p = paths.get(k)
        if not _looks_like(p):
            continue
        try:
            ds = xr.open_dataset(p)  # type: ignore
            ds_list.append((k, ds))
        except Exception as e:
            issues.append(f"无法打开 {k}: {p} · {e}")
    if len(ds_list) <= 1:
        # 单层可视为 OK（融合将跳过缺失层）
        spec = {"n_layers": len(ds_list)}
        for k, ds in ds_list:
            spec.setdefault("shapes", {})[k] = {d: int(ds.dims[d]) for d in ds.dims}
        for _, ds in ds_list:
            try:
                ds.close()
            except Exception:
                pass
        return {"ok": True, "issues": issues, "spec": spec}

    # 选首层为参考
    ref_key, ref_ds = ds_list[0]
    def _dims3(ds: xr.Dataset) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        t = ds.dims.get("time")
        y = ds.dims.get("y") or ds.dims.get("latitude") or ds.dims.get("lat")
        x = ds.dims.get("x") or ds.dims.get("longitude") or ds.dims.get("lon")
        return (int(t) if t is not None else None, int(y) if y is not None else None, int(x) if x is not None else None)

    ref_dims = _dims3(ref_ds)
    ok = True
    time_warn = False

    # lat/lon 坐标提取
    def _latlon(ds: xr.Dataset):
        latn = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else None)
        lonn = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
        if latn and lonn:
            return ds[latn].values, ds[lonn].values
        return None, None

    ref_lat, ref_lon = _latlon(ref_ds)

    for k, ds in ds_list[1:]:
        dims = _dims3(ds)
        if (dims[1] != ref_dims[1]) or (dims[2] != ref_dims[2]):
            ok = False
            issues.append(f"空间维不一致：{ref_key}={ref_dims[1:]} vs {k}={dims[1:]}")
        if dims[0] != ref_dims[0]:
            time_warn = True
            issues.append(f"时间维不一致：{ref_key}.time={ref_dims[0]} vs {k}.time={dims[0]}")
        # 坐标一致性（若两者都存在）
        lat, lon = _latlon(ds)
        if (ref_lat is not None) and (lat is not None):
            try:
                if not (np.allclose(np.asarray(ref_lat), np.asarray(lat), equal_nan=True) and np.allclose(np.asarray(ref_lon), np.asarray(lon), equal_nan=True)):
                    ok = False
                    issues.append(f"lat/lon 坐标不一致：{ref_key} vs {k}")
            except Exception:
                ok = False
                issues.append(f"lat/lon 比较失败：{ref_key} vs {k}")

    # time 频率与 grid 简要 spec
    spec: Dict[str, Any] = {"ref": ref_key, "n_layers": len(ds_list), "grid_spec": {}}
    spec["shapes"] = {k: {d: int(ds.dims[d]) for d in ds.dims} for k, ds in ds_list}
    # 尝试频率
    if pd is not None and "time" in ref_ds.dims:
        try:
            t = ref_ds["time"].to_index()
            if len(t) >= 2:
                freq = str(pd.infer_freq(t))
            else:
                freq = None
            spec["grid_spec"]["time_freq"] = freq
        except Exception:
            pass
    # 加入经纬坐标名
    latn = "lat" if "lat" in ref_ds.coords else ("latitude" if "latitude" in ref_ds.coords else None)
    lonn = "lon" if "lon" in ref_ds.coords else ("longitude" if "longitude" in ref_ds.coords else None)
    if latn and lonn:
        spec["grid_spec"]["coord_names"] = {"lat": latn, "lon": lonn}

    # 关闭
    for _, ds in ds_list:
        try:
            ds.close()
        except Exception:
            pass

    # 时间仅不一致 → WARN；空间不一致 → FAIL
    if not ok and time_warn and ("空间维不一致" not in ";".join(issues)):
        # 只有时间 warn
        return {"ok": True, "issues": issues, "spec": spec}
    return {"ok": ok, "issues": issues, "spec": spec}


def summarize_layer(path: str, var_hint: str | None = None) -> Dict[str, Any]:
    """读 xarray，给出变量名推断(R_ice/R_wave/R_acc)、非零率、分位数(1,5,50,95,99)、nan_pct 等。
    特殊规则：当 var_hint=R_ice 且文件仅含 ice_cost 时，将其视为 R_ice 的回退来源（统计用 ice_cost，显示标注来源）。"""
    if xr is None:
        raise RuntimeError("xarray required for summarize")
    try:
        ds = xr.open_dataset(path)
    except Exception as e:
        return {"path": path, "error": str(e)}

    issues: List[str]
    var_name: Optional[str]
    # 回退映射：ice_cost → R_ice（显示层面）
    if var_hint == "R_ice" and "R_ice" not in ds and "ice_cost" in ds:
        var_name = "ice_cost"
        issues = ["回退映射 ice_cost→R_ice (source=ice_cost)"]
        display_var = "R_ice(ice_cost)"
        source = "ice_cost"
    else:
        var_name, issues = _pick_var_name(ds, var_hint)
        display_var = var_name or "-"
        source = None

    if not var_name or var_name not in ds:
        try:
            ds.close()
        except Exception:
            pass
        return {"path": path, "error": "未识别到变量名"}

    da = ds[var_name]
    if "time" in da.dims:
        da = da.isel(time=0)
    arr = np.asarray(da.values, dtype=float)
    nan_pct = float(np.isnan(arr).mean() * 100.0)
    finite = arr[np.isfinite(arr)]
    nnz = float((finite > 0).mean() * 100.0) if finite.size else 0.0

    def q(p: float) -> float:
        try:
            return float(np.nanpercentile(finite, p)) if finite.size else float("nan")
        except Exception:
            return float("nan")

    out = {
        "path": path,
        "var": display_var,
        "issues": issues,
        "nan_pct": round(nan_pct, 3),
        "nonzero_pct": round(nnz, 3),
        "q01": q(1),
        "q05": q(5),
        "q50": q(50),
        "q95": q(95),
        "q99": q(99),
    }
    if source:
        out["source"] = source
    try:
        ds.close()
    except Exception:
        pass
    return out


def recommend_weights(stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """根据可用性与非零率推荐 {alpha,beta,gamma}。
    - 有三层：默认 0.6/0.2/0.2
    - 缺层：按剩余层归一
    - 非零率 <1%：该层权重×0.5 后整体归一
    返回 {'alpha':..., 'beta':..., 'gamma':..., 'notes':[...]}
    """
    w = {"alpha": 0.6, "beta": 0.2, "gamma": 0.2}
    notes: List[str] = []

    avail = {k: (v is not None and not v.get("error")) for k, v in stats.items()}
    # 缺层处理
    present = [k for k, a in avail.items() if a]
    if not present:
        notes.append("无可用分量，使用缺省占位权重 0/0/0")
        return {"alpha": 0.0, "beta": 0.0, "gamma": 0.0, "notes": notes}

    base = {"ice": 0.6, "wave": 0.2, "acc": 0.2}
    # 仅保留存在的层
    total = sum(base[k] for k in present)
    w_map = {k: (base[k] / total) for k in present}

    # 非零率低惩罚
    for k in present:
        nz = float(stats[k].get("nonzero_pct", 0.0))
        if nz < 1.0:
            w_map[k] *= 0.5
            notes.append(f"{k} 非零率低({nz:.2f}%)，降权×0.5")

    # 归一
    s = sum(w_map.values()) or 1.0
    for k in w_map:
        w_map[k] = w_map[k] / s

    # 映射到 alpha/beta/gamma
    out = {
        "alpha": float(w_map.get("ice", 0.0)),
        "beta": float(w_map.get("wave", 0.0)),
        "gamma": float(w_map.get("acc", 0.0)),
        "notes": notes,
    }
    return out


def build_fuse_config(ym: str, spec: Dict[str, Any], weights: Dict[str, Any]) -> Dict[str, Any]:
    cfg = {
        "ym": str(ym),
        "norm": "quantile",
        "missing": "skip_and_warn",
        "weights": {"alpha": float(weights.get("alpha", 0.0)), "beta": float(weights.get("beta", 0.0)), "gamma": float(weights.get("gamma", 0.0))},
        "risk_vars": {"ice": "R_ice", "wave": "R_wave", "acc": "R_acc"},
        "grid_spec": spec.get("grid_spec", {}),
    }
    return cfg


# ---- CLI utility (called from api.cli) ----

def run_precheck_and_write(ym: str, write_config: bool = False, dry_run: bool = True) -> Dict[str, Any]:
    paths = find_layer_paths(ym)
    dims = check_dims_coords_equal(paths)

    stats: Dict[str, Dict[str, Any]] = {}
    for k in ("ice", "wave", "acc"):
        p = paths.get(k)
        stats[k] = summarize_layer(p, var_hint={"ice": "R_ice", "wave": "R_wave", "acc": "R_acc"}.get(k)) if _looks_like(p) else {"path": p, "error": "missing"}

    weights = recommend_weights(stats)
    cfg = build_fuse_config(ym, dims.get("spec", {}), weights)

    # 输出位置
    os.makedirs(REPORT_DIR, exist_ok=True)
    md_path = os.path.join(REPORT_DIR, f"D_PREP_STATUS_{ym}.md")
    json_path = os.path.join(REPORT_DIR, f"D_PREP_STATUS_{ym}.json")
    cfg_path = os.path.join(CONFIG_DIR, f"risk_fuse_{ym}.yaml")

    # 生成 MD
    def _md_table_row(k: str, s: Dict[str, Any]) -> str:
        return f"| {k} | {s.get('path') or '-'} | {s.get('var') or '-'} | {s.get('nonzero_pct','-')} | {s.get('nan_pct','-')} | {s.get('q01','-')}/{s.get('q50','-')}/{s.get('q99','-')} |"

    md_lines = [
        f"# D-Prep Status {ym}",
        "", "## 层路径与变量映射", "", "| layer | path | var | nonzero%(time0) | nan%(time0) | q01/50/99 |", "|---|---|---|---:|---:|---|",
        _md_table_row("R_ice", stats.get("ice", {})),
        _md_table_row("R_wave", stats.get("wave", {})),
        _md_table_row("R_acc", stats.get("acc", {})),
        "", "## 维度/坐标一致性", "",
        f"- ok: {dims.get('ok')} ",
        f"- issues: {dims.get('issues')} ",
        f"- spec: {dims.get('spec')} ",
        "", "## 推荐权重与策略", "",
        f"- weights: {weights}",
        f"- config_preview: {cfg}",
        "", "## 建议与下一步", "",
        "- 缺层策略：skip_and_warn（D 阶段按可用层归一运行）",
        "- 若仅 time 不一致：建议 align_time()；若空间不一致：判定 FAIL（需修正）",
        "- 下一步：python -m api.cli risk.build --kind fuse --ym {ym} --alpha {a} --beta {b} --gamma {g}",
    ]
    md_lines[-1] = md_lines[-1].format(ym=ym, a=cfg["weights"]["alpha"], b=cfg["weights"]["beta"], g=cfg["weights"]["gamma"])

    payload = {
        "ym": ym,
        "paths": paths,
        "dims": dims,
        "stats": stats,
        "weights": weights,
        "config": cfg,
        "md": md_path,
        "json": json_path,
        "config_path": (cfg_path if write_config else None),
    }

    if dry_run:
        return payload

    # 写盘
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        import json as _json
        with open(json_path, "w", encoding="utf-8") as f:
            _json.dump(payload, f, ensure_ascii=False, indent=2)
        if write_config:
            os.makedirs(CONFIG_DIR, exist_ok=True)
            text = (yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True) if yaml is not None else str(cfg))
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write(text)
        # 登记 artifact
        try:
            run_id = os.environ.get("RUN_ID", "") or __import__("time").strftime("%Y%m%dT%H%M%S")
        except Exception:
            run_id = ""
        try:
            register_artifact(run_id=run_id, kind="risk_fuse_prep", path=json_path, attrs={"ym": ym, "write_config": int(write_config)})
        except Exception:
            pass
    except Exception:
        pass

    return payload


__all__ = [
    "find_layer_paths",
    "check_dims_coords_equal",
    "summarize_layer",
    "recommend_weights",
    "build_fuse_config",
    "run_precheck_and_write",
]

