from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

import numpy as np  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from ArcticRoute.core.prior.geo import to_xy


@dataclass
class FeatureConfig:
    crs: str = "EPSG:3413"  # 极区投影
    posenc: str = "rope"    # "rope" | "t2v"
    rope_base: float = 10000.0
    t2v_k: int = 8           # time2vec 频率数（输出维度为 k+1）
    standardize: bool = True # 是否进行稳健标准化（Median/MAD）


def _as_pandas(df_or_dict: Any):
    if pd is not None and isinstance(df_or_dict, pd.DataFrame):  # type: ignore[attr-defined]
        return df_or_dict
    # dict of arrays/lists
    if isinstance(df_or_dict, dict):
        if pd is None:
            raise RuntimeError("pandas not available to construct DataFrame")
        return pd.DataFrame(df_or_dict)
    raise TypeError("expect pandas.DataFrame or dict of arrays for sequence features")


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        return float(x)
    except Exception:
        return default


def _wrap_deg(deg: float) -> float:
    try:
        x = float(deg)
    except Exception:
        return float("nan")
    x = x % 360.0
    if x < 0:
        x += 360.0
    return x


def _median_mad(X: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """按列计算中位数与 MAD（Median Absolute Deviation）。mask=True 的位置参与统计。"""
    if mask is not None:
        valid = mask.reshape(-1, 1).astype(bool)
        # 将无效位置置为 NaN，使用 nan* 统计
        Xv = np.where(valid, X, np.nan)
        med = np.nanmedian(Xv, axis=0)
        mad = np.nanmedian(np.abs(Xv - med), axis=0)
    else:
        med = np.median(X, axis=0)
        mad = np.median(np.abs(X - med), axis=0)
    # 避免除零
    mad = np.where((mad == 0) | ~np.isfinite(mad), 1.0, mad)
    return med, mad


def _standardize(X: np.ndarray, med: np.ndarray, mad: np.ndarray) -> np.ndarray:
    # 常用系数 1.4826 使得 MAD 在正态分布下成为标准差估计
    return (X - med) / (mad * 1.4826)


def _time_components(ts: float) -> Tuple[float, float, float, float]:
    """从 UTC 秒时间戳得到小时和年内日编号的正弦余弦编码输入基。
    返回 hour_frac[0,1), day_of_year(1..366)
    """
    try:
        t = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    except Exception:
        return float("nan"), float("nan"), float("nan"), float("nan")
    hour = t.hour + t.minute / 60.0 + t.second / 3600.0
    doy = t.timetuple().tm_yday
    # sin/cos 的基础相位
    hour_theta = 2 * math.pi * (hour / 24.0)
    doy_theta = 2 * math.pi * (float(doy) / 366.0)
    return math.sin(hour_theta), math.cos(hour_theta), math.sin(doy_theta), math.cos(doy_theta)


def build_sequence_features(seq_any: Any, cfg: Optional[FeatureConfig] = None) -> Dict[str, Any]:
    """构建 transformer 序列输入特征与位置/时间编码。

    输入需要包含列：ts, lat, lon（必需）；可选：sog, cog。
    输出：
      - x: np.ndarray [T, C]，C=9（Δx, Δy, sog, cos(cog), sin(cog), hour_sin, hour_cos, doy_sin, doy_cos）
      - mask: np.ndarray [T]，bool，True 表示该时刻有效
      - stats: {"median": np.ndarray[C], "mad": np.ndarray[C]}
      - posenc: 若 cfg.posenc=="t2v" 则提供 {"type":"t2v", "emb": np.ndarray[T, k+1]}
                若 cfg.posenc=="rope" 则提供 {"type":"rope", "cos": np.ndarray[T, D], "sin": np.ndarray[T, D]}（D 按需由上层决定; 这里提供默认 D=64 的角频）
    缺失值处理：以 0 填充特征，并在 mask 中置 False；标准化统计仅基于 mask=True 的样本。
    """
    cfg = cfg or FeatureConfig()
    df = _as_pandas(seq_any)

    # 提取并做基础有效性判断
    lat = df.get("lat")
    lon = df.get("lon")
    ts = df.get("ts")
    sog = df.get("sog") if "sog" in df.columns else None
    cog = df.get("cog") if "cog" in df.columns else None

    T = int(len(df))
    lat_arr = np.array([_safe_float(v) for v in (lat.tolist() if lat is not None else [np.nan]*T)], dtype=float)
    lon_arr = np.array([_safe_float(v) for v in (lon.tolist() if lon is not None else [np.nan]*T)], dtype=float)
    ts_arr = np.array([_safe_float(v) for v in (ts.tolist() if ts is not None else [np.nan]*T)], dtype=float)
    sog_arr = np.array([_safe_float(v, 0.0) for v in (sog.tolist() if sog is not None else [0.0]*T)], dtype=float)
    cog_arr = np.array([_safe_float(v) for v in (cog.tolist() if cog is not None else [np.nan]*T)], dtype=float)

    # 有效掩码（lat/lon/ts 三者全具备）
    mask = np.isfinite(lat_arr) & np.isfinite(lon_arr) & np.isfinite(ts_arr)

    # 投影坐标
    xs = np.zeros(T, dtype=float)
    ys = np.zeros(T, dtype=float)
    for i in range(T):
        if mask[i]:
            try:
                x, y = to_xy(lat_arr[i], lon_arr[i], crs=cfg.crs)
                xs[i] = float(x)
                ys[i] = float(y)
            except Exception:
                mask[i] = False
    # Δx, Δy（首个置 0；缺失回填 0）
    dx = np.zeros(T, dtype=float)
    dy = np.zeros(T, dtype=float)
    if T >= 2:
        dx[1:] = np.diff(xs)
        dy[1:] = np.diff(ys)
        # 对于任一端点无效的差分，置 0
        inv = ~(mask[1:] & mask[:-1])
        dx[1:][inv] = 0.0
        dy[1:][inv] = 0.0

    # 航向分解（度→弧度）
    cog_deg = np.where(np.isfinite(cog_arr), cog_arr % 360.0, np.nan)
    cog_rad = np.deg2rad(cog_deg)
    cos_cog = np.where(np.isfinite(cog_rad), np.cos(cog_rad), 0.0)
    sin_cog = np.where(np.isfinite(cog_rad), np.sin(cog_rad), 0.0)

    # 时间编码基（小时/年内日）
    hour_sin = np.zeros(T, dtype=float)
    hour_cos = np.zeros(T, dtype=float)
    doy_sin = np.zeros(T, dtype=float)
    doy_cos = np.zeros(T, dtype=float)
    for i in range(T):
        if mask[i]:
            hs, hc, ds, dc = _time_components(ts_arr[i])
            hour_sin[i] = hs if np.isfinite(hs) else 0.0
            hour_cos[i] = hc if np.isfinite(hc) else 0.0
            doy_sin[i] = ds if np.isfinite(ds) else 0.0
            doy_cos[i] = dc if np.isfinite(dc) else 0.0

    # 汇总特征通道：
    X = np.stack([dx, dy, sog_arr, cos_cog, sin_cog, hour_sin, hour_cos, doy_sin, doy_cos], axis=1)

    # 稳健标准化（按通道）
    stats: Dict[str, Any] = {}
    if cfg.standardize:
        med, mad = _median_mad(X, mask=mask)
        Xz = _standardize(X, med, mad)
        # 对 mask=False 的位置，保持 0（更利于掩码）
        Xz[~mask, :] = 0.0
        X = Xz
        stats = {"median": med, "mad": mad}
    else:
        stats = {"median": np.zeros(X.shape[1], dtype=float), "mad": np.ones(X.shape[1], dtype=float)}

    # 位置/时间编码
    posenc: Dict[str, Any]
    if cfg.posenc.lower() == "t2v":
        # time2vec：输入使用 UTC 秒的归一化（到天），避免数值过大
        # 结构：g0(t) = w0 * t + b0; gi(t) = sin(wi * t + bi)
        # 这里简化为固定频率集，学习权重由模型完成，这里只提供固定基（正弦基 + 常数项）
        t_days = np.where(mask, ts_arr / 86400.0, 0.0)
        k = int(max(1, cfg.t2v_k))
        freqs = np.arange(1, k + 1, dtype=float)  # 1..k 周期基
        # [T, k]
        S = np.sin(t_days[:, None] * (2 * math.pi * freqs)[None, :])
        C = np.cos(t_days[:, None] * (2 * math.pi * freqs)[None, :])
        emb = np.concatenate([t_days[:, None], S, C], axis=1)  # [T, 1 + k + k]
        emb[~mask, :] = 0.0
        posenc = {"type": "t2v", "emb": emb}
    else:
        # RoPE：返回角频（cos/sin），用于对 Q/K 做旋转。
        # 这里给出固定维度 D=64 的旋转频带；若上层模型维度不同，可在使用时插值或重新生成。
        D = 64
        pos = np.arange(T, dtype=float)
        inv_freq = (cfg.rope_base) ** (-np.arange(0, D, 2).astype(float) / D)
        angles = pos[:, None] * inv_freq[None, :]  # [T, D/2]
        cos = np.cos(angles)
        sin = np.sin(angles)
        posenc = {"type": "rope", "cos": cos, "sin": sin, "d_rot": D}

    return {
        "x": X.astype(np.float32),
        "mask": mask.astype(bool),
        "stats": stats,
        "posenc": posenc,
    }


def make_features(lat: np.ndarray, lon: np.ndarray, sog: Optional[np.ndarray], cog: Optional[np.ndarray], ts: np.ndarray, cfg: Optional[FeatureConfig] = None) -> Dict[str, Any]:
    import pandas as _pd  # type: ignore
    data = {
        "lat": lat,
        "lon": lon,
        "ts": ts,
    }
    if sog is not None:
        data["sog"] = sog
    if cog is not None:
        data["cog"] = cog
    df = _pd.DataFrame(data)
    return build_sequence_features(df, cfg)


def apply_posenc(T: int, method: str = "rope", rope_base: float = 10000.0, t2v_k: int = 8, device: Optional[str] = None):
    import numpy as _np  # type: ignore
    import torch as _th  # type: ignore
    if method.lower() == "t2v":
        # 占位：仅生成零张量，建议从 features 中直接取 posenc
        emb = _np.zeros((T, 1 + 2 * int(max(1, t2v_k))), dtype=_np.float32)
        return {"type": "t2v", "emb": emb}
    # RoPE 生成 [T, D/2] cos/sin，默认 D=64
    D = 64
    pos = _np.arange(T, dtype=_np.float32)
    inv_freq = (rope_base) ** (-_np.arange(0, D, 2, dtype=_np.float32) / D)
    angles = _np.outer(pos, inv_freq)  # [T, D/2]
    return {"type": "rope", "cos": _np.cos(angles), "sin": _np.sin(angles), "d_rot": D}


__all__ = ["FeatureConfig", "build_sequence_features", "make_features", "apply_posenc"]

