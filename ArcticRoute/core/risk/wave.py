"""风浪风险层（R_wave）

基于 Hs/U10/(Tp*) 的阈值/逻辑映射，按 config/risk_wave.yaml 参数为不同船型计算风险。

- build_risk_wave(ym, vclass="all", params_yml=None, dry_run=True)
  读取：
    - P1 产物：ArcticRoute/data_processed/ice_forecast/merged/sic_fcst_<ym>.nc （用于时间/网格对齐）
    - 环境：ArcticRoute/data_processed/env_clean.nc （读取 Hs/U10/Tp 及其可能别名；若仅有风的分量则合成风速）
    - 参数：config/risk_wave.yaml 或 CLI 指定 YAML
  计算：
    R_wave = sigmoid(a*Hs + b*U10 + c*(Hs/Tp) + bias)，并裁剪到 [0,1]
  写盘（非 dry-run）：ArcticRoute/data_processed/risk/risk_wave_<ym>.nc；并在 reports/figs 导出 PNG
  注册：register_artifact(kind="risk_wave", attrs={ym,vclass,params})
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from ArcticRoute.cache.index_util import register_artifact


@dataclass
class WaveParams:
    a: float = 0.8
    b: float = 0.08
    c: float = 0.2
    bias: float = -2.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _load_params(params_yml: Optional[str], vclass: str) -> WaveParams:
    # 默认参数
    default = WaveParams()
    if params_yml and os.path.exists(params_yml) and yaml is not None:
        try:
            cfg = yaml.safe_load(open(params_yml, "r", encoding="utf-8")) or {}
            # 支持 {default:{a:..,b:..}, tanker:{...}} 或 {a:..,b:..}
            node = cfg.get(vclass) if isinstance(cfg, dict) and vclass in cfg else cfg.get("default", cfg)
            if isinstance(node, dict):
                return WaveParams(
                    a=float(node.get("a", default.a)),
                    b=float(node.get("b", default.b)),
                    c=float(node.get("c", default.c)),
                    bias=float(node.get("bias", default.bias)),
                )
        except Exception:
            pass
    return default


def _ensure_dirs() -> Tuple[str, str]:
    base = os.path.join(os.getcwd(), "ArcticRoute")
    out_dir = os.path.join(base, "data_processed", "risk")
    fig_dir = os.path.join(os.getcwd(), "reports", "figs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    return out_dir, fig_dir


def _pick_var(ds: "xr.Dataset", names: Tuple[str, ...]) -> Optional["xr.DataArray"]:
    for n in names:
        if n in ds:
            return ds[n]
    # 也尝试在 coords 里（少见）
    for n in names:
        if n in ds.coords:
            return ds.coords[n]  # type: ignore
    return None


def _load_env_components(env_path: str) -> Tuple[Optional["xr.DataArray"], Optional["xr.DataArray"], Optional["xr.DataArray"]]:
    ds = xr.open_dataset(env_path)
    # 常见别名：
    hs = _pick_var(ds, ("Hs", "hs", "swh", "VHM0"))
    u10 = _pick_var(ds, ("U10", "u10", "wind_speed_10m", "wind", "wind_speed"))
    v10 = _pick_var(ds, ("V10", "v10"))
    tp = _pick_var(ds, ("Tp", "tp", "tpeak", "peak_period", "wave_period"))
    # 合成风速
    if u10 is None and (v10 is not None):
        # 无法合成，维持 None
        pass
    if u10 is None and v10 is not None:
        u10 = v10  # 退化：仅用一个分量
    if (u10 is None) and ("u10" in ds and "v10" in ds):
        try:
            u10 = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)  # type: ignore
        except Exception:
            pass
    return hs, u10, tp


def _render_png(da: "xr.DataArray", title: str, out_png: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        arr = np.asarray(da.values)
        fig = plt.figure(figsize=(7.2, 3.2))
        ax = fig.add_subplot(111)
        im = ax.imshow(arr, origin="upper", cmap="inferno", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(da.name or "R_wave")
        fig.tight_layout()
        fig.savefig(out_png, dpi=140)
        plt.close(fig)
        return out_png
    except Exception:
        return None


def build_risk_wave(ym: str, vclass: str = "all", params_yml: Optional[str] = None, dry_run: bool = True) -> Dict[str, Any]:
    if xr is None:
        raise RuntimeError("xarray required to build risk_wave")

    # 1) 对齐参考：P1 sic_fcst_<ym>.nc
    sic_path = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "ice_forecast", "merged", f"sic_fcst_{ym}.nc")
    if not os.path.exists(sic_path):
        raise FileNotFoundError(f"缺少 sic_fcst 文件: {sic_path}")
    ref_ds = xr.open_dataset(sic_path)
    # 选择坐标，兼容 1 帧与多帧
    ref_template = ref_ds

    # 2) 载入环境变量
    env_path = os.path.join(os.getcwd(), "ArcticRoute", "data_processed", "env_clean.nc")
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"缺少环境文件: {env_path}")
    hs, u10, tp = _load_env_components(env_path)

    # 3) 插值/重采样到参考网格与时间
    def _fit(da):
        if da is None:
            return None
        try:
            return da.interp_like(ref_template, method="nearest")
        except Exception:
            return da

    hs2 = _fit(hs)
    u102 = _fit(u10)
    tp2 = _fit(tp)

    # 4) 参数
    params = _load_params(params_yml, vclass)

    # 5) 构建表达式
    def _safe(x):
        if x is None:
            # 用 0 占位（保守）
            return 0.0
        try:
            return x
        except Exception:
            return 0.0

    # Hs/Tp，避免除 0
    ratio = None
    if tp2 is not None:
        try:
            ratio = hs2 / xr.where(tp2 <= 1e-6, 1e-6, tp2)
        except Exception:
            ratio = None

    lin = (
        (params.a * (hs2 if hs2 is not None else 0))
        + (params.b * (u102 if u102 is not None else 0))
        + (params.c * (ratio if ratio is not None else 0))
        + params.bias
    )
    # 统一到 numpy 后计算 sigmoid，再包装回 DataArray
    arr = np.asarray(lin.values if hasattr(lin, "values") else lin, dtype=np.float32)
    r = _sigmoid(arr)
    r = np.clip(r, 0.0, 1.0)

    # 构建 Dataset（沿用 ref_ds 的 y/x/time 维顺序）
    # 取一个变量作为模板（sic_pred 或首个变量）
    var_name = "sic_pred" if "sic_pred" in ref_ds else (list(ref_ds.data_vars)[0] if ref_ds.data_vars else None)
    if var_name:
        tpl = ref_ds[var_name]
        da = xr.DataArray(r, dims=tpl.dims, coords=tpl.coords, name="R_wave")
    else:
        # 退化：尝试 y/x/time 名称
        dims = [d for d in ("time", "y", "x") if d in ref_ds.dims]
        coords = {d: ref_ds.coords[d] for d in dims if d in ref_ds.coords}
        da = xr.DataArray(r, dims=tuple(dims), coords=coords, name="R_wave")

    ds_out = xr.Dataset({"R_wave": da})
    ds_out["R_wave"].attrs.update({
        "long_name": "Wave risk",
        "units": "1",
        "mapping": "sigmoid(a*Hs + b*U10 + c*Hs/Tp + bias)",
    })
    ds_out = ds_out.assign_attrs({
        "layer": "risk_wave",
        "ym": str(ym),
        "vclass": str(vclass),
        "params": {"a": params.a, "b": params.b, "c": params.c, "bias": params.bias},
    })

    out_dir, fig_dir = _ensure_dirs()
    out_nc = os.path.join(out_dir, f"risk_wave_{ym}.nc")
    png_path = os.path.join(fig_dir, f"risk_wave_{ym}.png")

    if not dry_run:
        comp = {"zlib": True, "complevel": 4}
        enc = {"R_wave": {**comp}}
        ds_out.to_netcdf(out_nc, encoding=enc)
        # PNG：取第一帧
        try:
            da2 = ds_out["R_wave"]
            if "time" in da2.dims:
                da2 = da2.isel(time=0)
            _render_png(da2, f"R_wave {ym}", png_path)
        except Exception:
            pass
        try:
            run_id = os.environ.get("RUN_ID", "") or __import__("time").strftime("%Y%m%dT%H%M%S")
        except Exception:
            run_id = ""
        try:
            register_artifact(run_id=run_id, kind="risk_wave", path=out_nc, attrs={"ym": ym, "vclass": vclass})
        except Exception:
            pass

    # 关闭文件
    try:
        ref_ds.close()
    except Exception:
        pass

    return {
        "out": out_nc,
        "png": png_path,
        "dry_run": bool(dry_run),
        "shape": {k: int(ds_out.sizes[k]) for k in ds_out.sizes},
        "attrs": dict(ds_out.attrs),
    }


__all__ = ["build_risk_wave"]

