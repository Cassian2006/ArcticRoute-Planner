from __future__ import annotations

import numpy as np

from ArcticRoute.core.cost.aggregators import aggregate_risk


def test_aggregator_monotonic_mean_q_cvar():
    # 构造一组均值-方差，使其对应 Beta 可行
    mean = np.array([0.2, 0.5, 0.8], dtype=float)
    # 方差小于 m*(1-m) 即可；这里取 1/10 的可行上界
    var_cap = mean * (1 - mean)
    var = 0.1 * var_cap
    alpha = 0.9

    m = aggregate_risk(mean, var, mode="mean", alpha=alpha)
    q = aggregate_risk(mean, var, mode="q", alpha=alpha)
    c = aggregate_risk(mean, var, mode="cvar", alpha=alpha)

    assert np.all(m <= q + 1e-6), (m, q)
    assert np.all(q <= c + 1e-6), (q, c)


def test_aggregator_degenerate_no_variance():
    mean = np.linspace(0, 1, 5)
    m = aggregate_risk(mean, None, mode="mean", alpha=0.9)
    q = aggregate_risk(mean, None, mode="q", alpha=0.9)
    c = aggregate_risk(mean, None, mode="cvar", alpha=0.9)
    assert np.allclose(m, mean)
    assert np.allclose(q, mean)
    assert np.allclose(c, mean)

