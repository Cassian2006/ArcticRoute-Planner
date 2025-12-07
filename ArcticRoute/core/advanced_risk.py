# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import xarray as xr

# 统一项目根路径（minimum/）

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _risk_dir() -> Path:
    return _repo_root() / "ArcticRoute" / "data_processed" / "risk"


def _open_nc_first(path: Path, var_candidates: list[str]) -> Optional[xr.DataArray]:
    if not path.exists():
        return None
    try:
        with xr.open_dataset(path) as ds:
            # 变量优先列表
            for v in var_candidates:
                if v in ds.data_vars:
                    da = ds[v]
                    if "time" in da.dims:
                        da = da.isel(time=0)
                    return da.load()
            # 兜底：取首个变量
            if ds.data_vars:
                da = next(iter(ds.data_vars.values()))
                if "time" in da.dims:
                    da = da.isel(time=0)
                return da.load()
    except Exception:
        return None
    return None


# -------------------------
# 高级风险层加载
# -------------------------

def load_fused_risk(ym: str, fusion_mode: str, scenario: str = "default") -> Optional[xr.DataArray]:
    """
    仅从离线 .nc 文件加载融合风险；不做任何在线推理/训练。
    优先路径：
      - risk_fused_{ym}_{fusion_mode}.nc
      - risk_fused_{ym}.nc
    变量名候选："risk_fused", "risk", "prob_risk", "rv_fused"。
    找不到则返回 None 并打印降级日志。
    """
    risk_dir = _risk_dir()
    candidates = [
        risk_dir / f"risk_fused_{ym}_{fusion_mode}.nc",
        risk_dir / f"risk_fused_{ym}.nc",
    ]
    var_cands = ["risk_fused", "risk", "prob_risk", "rv_fused"]
    for p in candidates:
        da = _open_nc_first(p, var_cands)
        if da is not None:
            return da
    print(f"[FUSION] mode={fusion_mode}, ym={ym} not available, fallback to baseline")
    return None


def load_interact_risk(ym: str) -> Optional[xr.DataArray]:
    """
    拥挤/互动风险加载（优先 AIS→congestion_risk）。
    优先路径：traffic_density_{ym}.nc 中变量 "congestion_risk"（或由 traffic_density 归一化得到）
    回退路径：risk_interact_{ym}.nc, R_interact_{ym}.nc
    变量候选（回退）："risk_interact", "R_interact", "interact", "risk"。
    """
    risk_dir = _risk_dir()
    # 1) 新版 AIS 派生
    p_td = risk_dir / f"traffic_density_{ym}.nc"
    if p_td.exists():
        try:
            with xr.open_dataset(p_td) as ds:
                if "congestion_risk" in ds.data_vars:
                    da = ds["congestion_risk"]
                elif "traffic_density" in ds.data_vars:
                    td = ds["traffic_density"].astype("float32")
                    # 对数+分位归一（稳健一些）
                    arr = np.asarray(td.values, dtype=float)
                    with np.errstate(invalid="ignore"):
                        arr = np.log1p(arr)
                    finite = arr[np.isfinite(arr)]
                    if finite.size > 0:
                        lo = float(np.nanquantile(finite, 0.01))
                        hi = float(np.nanquantile(finite, 0.99))
                        if not np.isfinite(hi) or hi <= lo:
                            hi = float(np.nanmax(finite))
                            lo = float(np.nanmin(finite))
                        denom = (hi - lo) if (np.isfinite(hi) and np.isfinite(lo) and hi > lo) else 1.0
                        norm = np.clip((arr - lo) / denom, 0.0, 1.0).astype("float32")
                    else:
                        norm = np.zeros_like(arr, dtype="float32")
                    da = xr.DataArray(norm, dims=td.dims, coords=td.coords).rename("congestion_risk")
                else:
                    da = None
                if da is not None:
                    # 统一只保留空间维（其他维取第 0 或均值由外层 reduce_to_2d 处理）
                    return da.load()
        except Exception as e:
            print(f"[INTERACT] read traffic_density failed: {e}")
    # 2) 回退旧版
    candidates = [
        risk_dir / f"risk_interact_{ym}.nc",
        risk_dir / f"R_interact_{ym}.nc",
    ]
    var_cands = ["risk_interact", "R_interact", "interact", "risk"]
    for p in candidates:
        da = _open_nc_first(p, var_cands)
        if da is not None:
            return da
    print(f"[INTERACT] no interact file for ym={ym}.")
    return None


def load_escort_risk(ym: str) -> Optional[xr.DataArray]:
    """
    护航走廊风险加载（R_ice_eff）。
    路径候选：R_ice_eff_{ym}.nc, risk_ice_eff_{ym}.nc
    变量候选："R_ice_eff", "risk_ice_eff", "risk_ice"。
    """
    risk_dir = _risk_dir()
    candidates = [
        risk_dir / f"R_ice_eff_{ym}.nc",
        risk_dir / f"risk_ice_eff_{ym}.nc",
    ]
    var_cands = ["R_ice_eff", "risk_ice_eff", "risk_ice", "risk"]
    for p in candidates:
        da = _open_nc_first(p, var_cands)
        if da is not None:
            return da
    print(f"[ESCORT] no R_ice_eff for ym={ym}.")
    return None


# -------------------------
# 聚合与归一化
# -------------------------

def aggregate_risk_da(da: xr.DataArray, mode: str, alpha: float) -> xr.DataArray:
    """
    - mode == 'mean': 对样本维取均值。
    - mode == 'quantile': 按 alpha 分位。
    - mode == 'cvar': 先分位，再对尾部求均值。
    若无样本维（sample/member/ensemble），直接返回原 da，并由调用者决定 effective 模式（并打印降级日志）。
    """
    if da is None:
        return da
    dims = list(da.dims)
    sample_dim = next((d for d in ("sample", "member", "ensemble") if d in dims), None)
    if sample_dim is None:
        return da
    m = (mode or "mean").lower()
    if m == "mean":
        return da.mean(dim=sample_dim)
    if m in ("q", "quantile"):
        try:
            out = da.quantile(float(alpha), dim=sample_dim, keep_attrs=True)
            if "quantile" in out.dims:
                out = out.squeeze("quantile", drop=True)
            return out
        except Exception:
            return da.mean(dim=sample_dim)
    if m in ("cvar", "es", "expected_shortfall"):
        try:
            q = da.quantile(float(alpha), dim=sample_dim)
            if "quantile" in q.dims:
                q = q.squeeze("quantile", drop=True)
            tail = xr.where(da >= q, da, np.nan)
            out = tail.mean(dim=sample_dim, skipna=True)
            out = xr.where(np.isnan(out), q, out)
            return out
        except Exception:
            return da.mean(dim=sample_dim)
    return da.mean(dim=sample_dim)


def normalize_quantile(da: xr.DataArray, qlo: float = 0.05, qhi: float = 0.95) -> xr.DataArray:
    arr = np.asarray(da.values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return xr.zeros_like(da, dtype="float32")
    try:
        lo = float(np.nanquantile(finite, qlo))
        hi = float(np.nanquantile(finite, qhi))
    except Exception:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    norm = ((arr - lo) / (hi - lo)).clip(0.0, 1.0).astype("float32")
    return xr.DataArray(norm, dims=da.dims)


def reduce_to_2d(da: xr.DataArray) -> xr.DataArray:
    """
    将任意维度的风险场压缩为 2D，并尽量命名为 (y,x)。
    规则：
    - 优先识别 (y,x) 或 (lat,lon)/(latitude,longitude) 作为空间维；其他维做均值。
    - 若无法识别，启发式：若前两维尺寸远大于其余（如 121x1161 vs 2x2），则以前两维为空间维；否则默认最后两维为空间维。
    - 最终尝试将空间维重命名为 (y,x)。
    """
    if da is None:
        return da
    try:
        dims = list(da.dims)
        sizes = [int(da.sizes[d]) for d in dims]
        # 优先显式空间维
        y_dim = next((d for d in ("y", "lat", "latitude") if d in dims), None)
        x_dim = next((d for d in ("x", "lon", "longitude") if d in dims), None)
        if y_dim is not None and x_dim is not None:
            reduce_dims = [d for d in dims if d not in (y_dim, x_dim)]
            out = da
            if reduce_dims:
                out = out.mean(dim=reduce_dims, skipna=True)
            if list(out.dims) != [y_dim, x_dim]:
                out = out.transpose(y_dim, x_dim)
            rename_map = {}
            if y_dim != "y": rename_map[y_dim] = "y"
            if x_dim != "x": rename_map[x_dim] = "x"
            if rename_map:
                out = out.rename(rename_map)
            return out.astype("float32")
        # 启发式判断：前两维大、后两维小（如 121x1161 vs 2x2）
        if len(dims) >= 4:
            s0, s1, s2, s3 = sizes[0], sizes[1], sizes[-2], sizes[-1]
            if (s0 * s1) >= 1000 and max(s2, s3) <= 8:
                reduce_dims = dims[2:]
                out = da
                if reduce_dims:
                    out = out.mean(dim=reduce_dims, skipna=True)
                # 重命名前两维为 (y,x)
                try:
                    out = out.rename({dims[0]: "y", dims[1]: "x"})
                except Exception:
                    pass
                return out.astype("float32")
        # 默认：最后两维为空间维
        if len(dims) >= 2:
            spatial_dims = dims[-2:]
            reduce_dims = dims[:-2]
            out = da
            if reduce_dims:
                out = out.mean(dim=reduce_dims, skipna=True)
            try:
                out = out.rename({spatial_dims[0]: "y", spatial_dims[1]: "x"})
            except Exception:
                pass
            return out.astype("float32")
        # 退化
        arr = np.asarray(da.values)
        if arr.ndim == 1:
            return xr.DataArray(arr[:, None].astype("float32"), dims=("y", "x"))
        else:
            v = float(arr) if np.ndim(arr) == 0 else 0.0
            return xr.DataArray(np.full((1,1), v, dtype="float32"), dims=("y", "x"))
    except Exception:
        return da if da.ndim == 2 else xr.DataArray(np.zeros((1,1), dtype="float32"), dims=("y","x"))


def align_like(target: xr.DataArray, src: xr.DataArray) -> xr.DataArray:
    """
    将 src 对齐到 target 的 y/x 网格。
    优先使用 interp_like，失败时使用最邻近重采样。
    """
    try:
        if src.shape[-2:] == target.shape[-2:]:
            return src
        # 先尝试 xarray 对齐（如果坐标可用）
        try:
            return src.interp_like(target)
        except Exception:
            pass
        # 最邻近重采样
        Ht, Wt = target.shape[-2], target.shape[-1]
        Hs, Ws = src.shape[-2], src.shape[-1]
        yi = (np.linspace(0, Hs - 1, Ht)).astype(int)
        xj = (np.linspace(0, Ws - 1, Wt)).astype(int)
        arr = np.asarray(src.values)
        out = arr[yi[:, None], xj[None, :]]
        return xr.DataArray(out.astype("float32"), dims=("y", "x"))
    except Exception:
        return src


# -------------------------
# Evidential 鲁棒风险表面
# -------------------------

def _find_var_name(ds: xr.Dataset, prefers: Tuple[str, ...]) -> Optional[str]:
    for k in prefers:
        if k in ds.data_vars:
            return k
    # 宽松匹配（忽略大小写）
    for name in list(ds.data_vars.keys()):
        low = name.lower()
        for k in prefers:
            if k.lower() == low or k.lower() in low:
                return name
    return None


def build_evidential_robust_surface(
    ym: str,
    fusion_mode: str,
    risk_agg_mode: str,
    risk_agg_alpha: float = 0.9,
) -> xr.DataArray | None:
    """
    基于 evidential 融合输出的 Risk + RiskVar 构造鲁棒风险表面。

    - 非 evidential 模式：返回 None（调用方回退原逻辑）
    - 若仅有 Risk 无 RiskVar 或 risk_agg_mode 为 mean：返回均值 Risk（2D）
    - 若请求 cvar/robust 且存在 RiskVar：返回 R_robust = Risk + lambda * sqrt(RiskVar)
    """
    try:
        if not fusion_mode or ("evidential" not in str(fusion_mode).lower()):
            return None
        risk_dir = _risk_dir()
        # 文件名：优先带后缀，其次通用
        nc_candidates = [
            risk_dir / f"risk_fused_{ym}_{fusion_mode}.nc",
            risk_dir / f"risk_fused_{ym}.nc",
        ]
        ds = None
        nc_path = None
        for p in nc_candidates:
            if p.exists():
                nc_path = p
                break
        if nc_path is None:
            print(f"[EVIDENTIAL] risk_fused file not found for ym={ym}, mode={fusion_mode}")
            return None
        ds = xr.open_dataset(nc_path)
        try:
            # 变量名兼容：Risk/RiskVar 或 risk/riskvar
            risk_name = _find_var_name(ds, ("Risk", "risk", "risk_fused"))
            var_name = _find_var_name(ds, ("RiskVar", "riskvar", "var", "variance"))
            if risk_name is None:
                print(f"[EVIDENTIAL] dataset has no Risk variable: {list(ds.data_vars.keys())}")
                return None
            risk_da = ds[risk_name]
            # 单时间片（若包含 time 维）
            if "time" in risk_da.dims:
                risk_da = risk_da.isel(time=0)
            # 仅有均值或请求 mean → 直接返回 2D 均值
            mode_l = (risk_agg_mode or "mean").lower()
            if var_name is None or mode_l in ("mean", "avg", "average"):
                r2 = reduce_to_2d(risk_da)
                try:
                    r2.name = "risk_evidential_robust"
                    r2.attrs.update({
                        "agg": "mean",
                        "agg_alpha": float(risk_agg_alpha),
                        "source": "evidential_risk_plus_var",
                        "note": "no_var_or_mean_requested",
                    })
                except Exception:
                    pass
                return r2.astype("float32")
            # 同时存在 Risk 与 RiskVar → 构造 robust
            var_da = ds[var_name]
            if "time" in var_da.dims:
                var_da = var_da.isel(time=0)
            # 压成 2D，并对齐到 risk
            risk2 = reduce_to_2d(risk_da)
            var2 = reduce_to_2d(var_da)
            if tuple(var2.shape[-2:]) != tuple(risk2.shape[-2:]):
                var2 = align_like(risk2, var2)
            # 计算 std
            var_np = np.asarray(var2.values, dtype=float)
            var_np = np.clip(var_np, 0.0, np.nanmax(var_np) if np.isfinite(var_np).any() else 0.0)
            std_np = np.sqrt(var_np)
            mean_np = np.asarray(risk2.values, dtype=float)
            # lambda 简单映射
            a = float(risk_agg_alpha or 0.9)
            if a >= 0.95:
                lam = 2.0
            elif a >= 0.9:
                lam = 1.5
            else:
                lam = 1.0
            robust_np = mean_np + float(lam) * std_np
            robust_np = np.clip(robust_np, 0.0, None).astype("float32")
            robust = xr.DataArray(robust_np, dims=("y", "x"))
            robust.name = "risk_evidential_robust"
            try:
                robust.attrs.update({
                    "agg": ("cvar" if mode_l in ("cvar", "es", "expected_shortfall", "robust") else "mean"),
                    "agg_alpha": float(a),
                    "lambda": float(lam),
                    "source": "evidential_risk_plus_var",
                })
            except Exception:
                pass
            return robust
        finally:
            try:
                ds.close()
            except Exception:
                pass
    except Exception as e:
        try:
            print(f"[EVIDENTIAL] robust surface build failed: {e}")
        except Exception:
            pass
        return None
