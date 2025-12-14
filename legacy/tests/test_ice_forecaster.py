import math
import sys
import numpy as np
import pandas as pd
import pytest

try:
    from ArcticRoute.io.ice_sarima_lstm import IceForecaster, _HAS_TORCH, _HAS_STATSMODELS
except Exception as e:  # pragma: no cover
    pytest.skip("IceForecaster import failed: %s" % e, allow_module_level=True)


def _make_synthetic_series(n_months=120, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n_months)
    # 基本季节项 + 轻微线性趋势
    base = 0.5 + 0.3 * np.sin(2 * np.pi * t / 12.0) + 0.05 * (t / n_months)
    # 非线性扰动（周期 6 月的平方项，使其非线性且可由 LSTM 学到部分）
    nl = 0.1 * (np.sin(2 * np.pi * t / 6.0) ** 2 - 0.5)
    noise = 0.03 * rng.standard_normal(n_months)
    y = base + nl + noise
    y = np.clip(y, 0.0, 1.0)
    idx = pd.date_range("2010-01-01", periods=n_months, freq="MS")
    return pd.Series(y, index=idx)


@pytest.mark.skipif(not _HAS_TORCH, reason="需要 PyTorch 以训练 LSTM")
@pytest.mark.skipif(not _HAS_STATSMODELS, reason="需要 statsmodels 以训练 SARIMA")
def test_combination_better_than_sarima_on_synthetic():
    s = _make_synthetic_series(132)
    train, test = s.iloc[:108], s.iloc[108:]

    # SARIMA only: 通过 lstm_hidden=0 禁用 LSTM
    sarima_only = IceForecaster(epochs=1, lstm_hidden=0)
    sarima_only.fit(train)
    fc_sarima = sarima_only.predict(len(test))

    # SARIMA + LSTM（非线性残差）
    combo = IceForecaster(lstm_hidden=32, lstm_layers=1, epochs=25, lr=1e-3)
    combo.fit(train)
    fc_combo = combo.predict(len(test))

    y_true = test.values
    rmse_sarima = float(np.sqrt(np.mean((y_true - fc_sarima.values) ** 2)))
    rmse_combo = float(np.sqrt(np.mean((y_true - fc_combo.values) ** 2)))

    # 组合应优于仅 SARIMA（容忍微小差距浮动）
    assert rmse_combo <= rmse_sarima + 1e-4


def test_predict_horizon_and_range():
    s = _make_synthetic_series(48)
    # 允许无 torch/statsmodels 的降级：只要能输出预测并在 [0,1]
    model = IceForecaster(epochs=1, lstm_hidden=0)
    model.fit(s.iloc[:-12])

    out7 = model.predict(7)
    out12 = model.predict(12)

    assert len(out7) == 7 and len(out12) == 12
    assert isinstance(out7.index, pd.DatetimeIndex)
    assert out7.min() >= 0.0 - 1e-6 and out7.max() <= 1.0 + 1e-6
    assert out12.min() >= 0.0 - 1e-6 and out12.max() <= 1.0 + 1e-6

