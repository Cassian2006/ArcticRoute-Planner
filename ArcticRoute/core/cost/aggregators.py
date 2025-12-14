from __future__ import annotations

"""
Phase I | 风险聚合器

提供基于 Evidential/Beta 近似的聚合：mean / q(分位) / CVaR@α。
当仅有均值无方差时，退化为：
- mean: 直接均值
- q/cvar: 退化为均值（或轻微保守偏移，可在未来加入配置）

若提供 (mean,var) 且满足 Beta 可行域（var < mean*(1-mean)），则拟合 Beta(a,b) 并计算：
- 分位：scipy 或数值逼近；若无 scipy，则用简单牛顿迭代或 Clopper-Pearson 近似
- CVaR：对 Beta 使用尾部积分公式；若不可用则蒙特卡洛近似

注意：所有结果裁剪到 [0,1]。
"""

from typing import Optional, Tuple
import numpy as np

try:
    from scipy.stats import beta as sp_beta  # type: ignore
except Exception:  # pragma: no cover
    sp_beta = None  # type: ignore


def _safe_clip01(x: np.ndarray | float) -> np.ndarray | float:
    return np.clip(x, 0.0, 1.0)


def beta_from_mean_var(mean: np.ndarray | float, var: np.ndarray | float, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    m = np.asarray(mean, dtype=float)
    v = np.asarray(var, dtype=float)
    m = np.clip(m, eps, 1.0 - eps)
    v = np.maximum(v, 0.0)
    cap = m * (1.0 - m)
    # 为数值稳定，若 var 超界，按 0.95*cap 限制（表示较强不确定性）
    v_eff = np.where(v >= cap, 0.95 * cap, v)
    t = (m * (1.0 - m)) / (v_eff + eps) - 1.0
    a = m * t
    b = (1.0 - m) * t
    a = np.maximum(a, eps)
    b = np.maximum(b, eps)
    return a, b


def beta_quantile(mean: np.ndarray | float, var: np.ndarray | float, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    a, b = beta_from_mean_var(mean, var)
    if sp_beta is not None:
        try:
            return sp_beta.ppf(alpha, a, b)
        except Exception:
            pass
    # 后备：蒙特卡洛近似（小样本确保可用性）
    rng = np.random.default_rng(123)
    # 采样次数适中，避免过慢
    n = 256
    samples = rng.beta(a, b, size=(n,)) if np.ndim(a) == 0 else rng.beta(a[..., None], b[..., None], size=(a.shape + (n,)))
    return np.quantile(samples, alpha, axis=-1)


def beta_cvar(mean: np.ndarray | float, var: np.ndarray | float, alpha: float) -> np.ndarray:
    """右尾部 CVaR（条件在超出分位时的期望）。
    对风险语义（越大越坏），我们通常取高分位（如 0.95）。
    """
    q = beta_quantile(mean, var, alpha)
    a, b = beta_from_mean_var(mean, var)
    if sp_beta is not None:
        try:
            # 数值积分：E[X | X>=q] = (1/(1-alpha)) * ∫_q^1 x f(x) dx
            # 使用不完全 Beta 函数关系：∫ x^{a}(1-x)^{b-1} dx = -B_x(a+1,b)
            from mpmath import betainc  # type: ignore
            a1 = a + 1.0
            # B(a+1,b) - B_q(a+1,b)
            import mpmath as mp  # type: ignore
            B = mp.beta(a1, b)
            Bq = betainc(a1, b, 0, q, regularized=False)
            num = (B - Bq) / mp.beta(a, b)
            cvar = (num) / (1.0 - alpha)
            return _safe_clip01(np.asarray(cvar, dtype=float))
        except Exception:
            pass
    # 后备：蒙特卡洛
    rng = np.random.default_rng(123)
    n = 1024
    samples = rng.beta(a, b, size=(n,)) if np.ndim(a) == 0 else rng.beta(a[..., None], b[..., None], size=(a.shape + (n,)))
    kth = int(max(1, round(alpha * n)))
    part = np.partition(samples, kth, axis=-1)
    tail = part[..., kth:]
    cvar = tail.mean(axis=-1)
    return _safe_clip01(cvar)


def aggregate_risk(risk_mean: np.ndarray | float, risk_var: Optional[np.ndarray | float] = None, *, mode: str = "mean", alpha: float = 0.95) -> np.ndarray:
    mode = (mode or "mean").lower()
    m = np.asarray(risk_mean, dtype=float)
    if risk_var is None:
        if mode == "mean":
            return _safe_clip01(m)
        # 无方差时保守退化：把分位/CVaR都近似为均值
        return _safe_clip01(m)
    v = np.asarray(risk_var, dtype=float)
    if mode == "mean":
        return _safe_clip01(m)
    if mode in ("q", "quantile"):
        return _safe_clip01(beta_quantile(m, v, alpha))
    if mode in ("cvar", "es", "expected_shortfall"):
        return _safe_clip01(beta_cvar(m, v, alpha))
    # 未知模式回退
    return _safe_clip01(m)


__all__ = [
    "aggregate_risk",
    "beta_from_mean_var",
    "beta_quantile",
    "beta_cvar",
]



