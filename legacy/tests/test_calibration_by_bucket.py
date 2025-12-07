from __future__ import annotations
import numpy as np
from ArcticRoute.core.fusion_adv.calibrate import fit_by_bucket, ece_score, apply_calibration, save_calibrator
import os, tempfile, json


def test_per_bucket_ece_not_worse():
    rng = np.random.default_rng(42)
    H, W = 40, 50
    # 构造两个 bucket，分布不同
    buckets = np.zeros((H, W), dtype=int)
    buckets[:, W//2:] = 1
    # 生成未校准概率与标签
    probs = rng.beta(2.0, 5.0, size=(H, W)).astype(float)
    labels = (rng.random((H, W)) < (probs * 0.8 + 0.1)).astype(float)  # 有偏差
    mask = np.ones_like(probs, dtype=float)

    # 全局 ECE
    ece_global = ece_score(probs, labels, mask)

    # 按桶拟合并应用
    models = fit_by_bucket(probs, labels, mask, buckets=buckets, method="isotonic")
    # 将每个 bucket 段分别校准
    probs_cal = probs.copy()
    for b, model in models.items():
        b0 = int(b)
        sel = (buckets == b0)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, f"cal_{b}.json")
            save_calibrator(model, path)
            probs_cal[sel] = apply_calibration(probs[sel], path)
    ece_bucket = ece_score(probs_cal, labels, mask)

    # 允许随机性，要求不比全局更差（小于等于 + 1e-3 容差）
    assert ece_bucket <= ece_global + 1e-3, (ece_bucket, ece_global)

