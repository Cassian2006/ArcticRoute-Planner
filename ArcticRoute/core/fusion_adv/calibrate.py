from __future__ import annotations
"""
后校准（后处理）：支持 Logistic 与 Isotonic

提供：
- fit_calibrator(probs, labels, mask, method='isotonic'|'logistic') -> dict(model)
- save_calibrator(model, path) / load_calibrator(path)
- apply_calibration(prob2d, calib_path) -> np.ndarray
- plot_reliability(...) 保存到 reports/d_stage/phaseK

实现最小可用：
- logistic 使用 sklearn.linear_model.LogisticRegression(solver='lbfgs')
- isotonic 使用 sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')
- 模型持久化为 JSON：{'method': 'isotonic', 'x': [...], 'y': [...]} 或 {'coef': [w,b]}
"""
from typing import Any, Dict, Tuple, Optional
import os
import json
import numpy as np
import time

try:
    from sklearn.isotonic import IsotonicRegression  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
except Exception:  # pragma: no cover
    IsotonicRegression = None  # type: ignore
    LogisticRegression = None  # type: ignore

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore

PHASEK_DIR = os.path.join(os.getcwd(), "ArcticRoute", "reports", "d_stage", "phaseK")


def _masked_flat(probs: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = probs[mask > 0.5].astype(np.float64)
    y = labels[mask > 0.5].astype(np.float64)
    # 限制到[0,1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    y = np.clip(y, 0.0, 1.0)
    return p, y


def fit_calibrator(probs: np.ndarray, labels: np.ndarray, mask: np.ndarray, method: str = "isotonic") -> Dict[str, Any]:
    p, y = _masked_flat(probs, labels, mask)
    method = method.lower()
    if method == "logistic":
        if LogisticRegression is None:
            raise RuntimeError("scikit-learn 未安装，无法使用 logistic")
        lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        # 用 logit(p) 作特征更稳健
        x = np.log(p) - np.log(1 - p)
        lr.fit(x.reshape(-1,1), y.astype(int))
        w = float(lr.coef_.ravel()[0])
        b = float(lr.intercept_.ravel()[0])
        return {"method": "logistic", "w": w, "b": b}
    # 默认 isotonic
    if IsotonicRegression is None:
        raise RuntimeError("scikit-learn 未安装，无法使用 isotonic")
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(p, y)
    # 以拟合点对作为离散映射保存（保证单调）
    xs = np.linspace(0, 1, 256)
    ys = ir.predict(xs)
    ys = np.clip(ys, 0.0, 1.0)
    return {"method": "isotonic", "x": xs.tolist(), "y": ys.astype(float).tolist()}


def save_calibrator(model: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)
    return path


def load_calibrator(path: str) -> Dict[str, Any]:
    return json.loads(open(path, "r", encoding="utf-8").read())


def apply_calibration(prob2d: np.ndarray, calib_path: str) -> np.ndarray:
    model = load_calibrator(calib_path)
    p = prob2d.astype(np.float32)
    if model.get("method") == "logistic":
        w = float(model.get("w", 1.0)); b = float(model.get("b", 0.0))
        x = np.log(np.clip(p,1e-6,1-1e-6)) - np.log(np.clip(1-p,1e-6,1-1e-6))
        y = 1.0 / (1.0 + np.exp(-(w * x + b)))
        return y.astype(np.float32)
    # isotonic
    xs = np.array(model.get("x", [0.0, 1.0]), dtype=float)
    ys = np.array(model.get("y", [0.0, 1.0]), dtype=float)
    out = np.interp(np.clip(p, 0.0, 1.0), xs, ys)
    return out.astype(np.float32)


def plot_reliability(probs: np.ndarray, labels: np.ndarray, mask: np.ndarray, title: str, out_png: Optional[str] = None) -> Optional[str]:
    if plt is None:
        return None
    p, y = _masked_flat(probs, labels, mask)
    bins = np.linspace(0,1,11)
    xs = []; ys = []; ns = []
    for i in range(10):
        m = (p >= bins[i]) & (p < bins[i+1] + (1e-8 if i==9 else 0))
        if m.any():
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    plt.figure(figsize=(4.0,3.2))
    plt.plot([0,1],[0,1],'k--',alpha=0.5)
    plt.scatter(xs, ys, c='tab:blue')
    for i,(xx,yy,n) in enumerate(zip(xs,ys,ns)):
        plt.text(xx, yy, str(n), fontsize=7)
    plt.title(title)
    plt.xlabel('confidence'); plt.ylabel('accuracy')
    plt.tight_layout()
    os.makedirs(PHASEK_DIR, exist_ok=True)
    out_png = out_png or os.path.join(PHASEK_DIR, f"calibration_{int(time.time())}.png")
    plt.savefig(out_png, dpi=140)
    plt.close()
    return out_png


def ece_score(probs: np.ndarray, labels: np.ndarray, mask: np.ndarray, n_bins: int = 10) -> float:
    p, y = _masked_flat(probs, labels, mask)
    if p.size == 0:
        return 1.0
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i+1] + (1e-8 if i==n_bins-1 else 0))
        if m.any():
            conf = float(p[m].mean()); acc = float(y[m].mean())
            ece += (m.mean()) * abs(conf - acc)
    return float(ece)


def fit_by_bucket(probs: np.ndarray, labels: np.ndarray, mask: np.ndarray, buckets: np.ndarray, method: str = "isotonic") -> Dict[str, Any]:
    """按 bucket 拟合校准器，返回 {bucket: model}。
    probs/labels/mask/buckets 均应为相同形状的二维数组或展平后的一维数组。
    """
    assert probs.shape == labels.shape == mask.shape == buckets.shape, "shape mismatch"
    models: Dict[str, Any] = {}
    uniq = np.unique(buckets)
    for b in uniq:
        mb = (buckets == b) & (mask > 0.5)
        if not np.any(mb):
            continue
        model = fit_calibrator(probs[mb], labels[mb], np.ones_like(labels[mb]), method=method)
        models[str(b)] = model
    return models


__all__ = [
    "fit_calibrator", "save_calibrator", "load_calibrator", "apply_calibration", "plot_reliability", "ece_score", "fit_by_bucket"
]

