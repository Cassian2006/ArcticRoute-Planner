from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# 可选依赖：statsmodels, torch
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
    _HAS_STATSMODELS = True
except Exception:  # pragma: no cover
    SARIMAX = None  # type: ignore
    _HAS_STATSMODELS = False

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore
    _HAS_TORCH = False


@dataclass
class _SavedState:
    seasonal_period: int
    sarima_order: Tuple[int, int, int]
    sarima_seasonal: Tuple[int, int, int, int]
    lstm_hidden: int
    lstm_layers: int
    lr: float
    window: int
    # 训练数据的信息
    last_timestamp: pd.Timestamp
    freq: str
    # SARIMA
    sarima_fit_bytes: Optional[bytes]  # 使用 pickle 序列化 statsmodels 结果
    # LSTM
    lstm_state_dict: Optional[dict]
    lstm_scaler: Optional[Tuple[float, float]]  # (mean, std) 用于残差标准化
    last_r_std_tail: Optional[np.ndarray]
    lambda_shrink: float


class _ResLSTM(nn.Module):
    def __init__(self, input_size: int, hidden: int, layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers, batch_first=True, dropout=0.0)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (B, T, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 最后时间步
        out = self.fc(out)
        return out  # (B,1)


class IceForecaster:
    def __init__(
        self,
        seasonal_period: int = 12,
        sarima_order: Tuple[int, int, int] = (1, 1, 1),
        sarima_seasonal: Tuple[int, int, int, int] = (1, 1, 1, 12),
        lstm_hidden: int = 32,
        lstm_layers: int = 1,
        epochs: int = 50,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.seasonal_period = seasonal_period
        self.sarima_order = sarima_order
        self.sarima_seasonal = sarima_seasonal
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.epochs = epochs
        self.lr = lr
        self.device = "cpu"  # 测试更稳定，强制 CPU

        # 训练后状态
        self._sarima_res = None
        self._lstm: Optional[_ResLSTM] = None
        self._lstm_scaler: Optional[Tuple[float, float]] = None
        self._window = max(6, seasonal_period)  # 兼容旧字段（保存用）
        self._seq_len = 24  # 滑窗长度固定 24
        self._last_timestamp: Optional[pd.Timestamp] = None
        self._freq: Optional[str] = None
        self._last_r_std_tail: Optional[np.ndarray] = None  # 用于滚动预测的初始窗口
        self.used_fallback: bool = False
        self.lambda_shrink: float = 1.0

    # --------------------- 内部工具 ---------------------
    @staticmethod
    def _prep_series(series: pd.Series) -> pd.Series:
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("series.index 必须是 DatetimeIndex")
        s = series.sort_index()
        # 月频对齐
        if s.index.freq is None:
            s = s.asfreq("MS")  # 月始
        # 缺测处理和裁剪
        s = s.astype(float).interpolate(limit=2).clip(0.0, 1.0)
        return s

    @staticmethod
    def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def _fit_sarima(self, y: pd.Series):
        try:
            if not _HAS_STATSMODELS:
                raise RuntimeError("statsmodels 不可用")
            model = SARIMAX(
                y.values,
                order=self.sarima_order,
                seasonal_order=self.sarima_seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)
            # 收敛性检查，不收敛则回退
            converged = True
            try:
                mle = getattr(res, "mle_retvals", {}) or {}
                converged = bool(mle.get("converged", True))
            except Exception:
                converged = True
            if not converged:
                self._sarima_res = ("seasonal_mean", self._seasonal_profile(y))
                self.used_fallback = True
            else:
                self._sarima_res = ("sarima", res)
        except Exception:
            # 回退：简单月别均值轮廓
            self._sarima_res = ("seasonal_mean", self._seasonal_profile(y))
            self.used_fallback = True

    def _seasonal_profile(self, y: pd.Series) -> np.ndarray:
        # 月别平均轮廓，用于回退预测
        months = y.index.month
        prof = np.zeros(12, dtype=float)
        for m in range(1, 13):
            v = y[months == m].values
            prof[m - 1] = float(v.mean()) if v.size else float(y.mean())
        return prof

    def _sarima_in_sample(self, y: pd.Series) -> np.ndarray:
        tag, obj = self._sarima_res
        if tag == "sarima":
            fitted = obj.fittedvalues  # type: ignore
            fv = np.asarray(fitted)
            if fv.shape[0] < len(y):
                pad = np.full(len(y) - fv.shape[0], fv[-1])
                fv = np.concatenate([pad, fv])
            return fv
        else:
            profile = obj  # type: ignore
            months = y.index.month
            return np.array([profile[m - 1] for m in months], dtype=float)

    def _sarima_forecast(self, steps: int, start_month: int) -> np.ndarray:
        tag, obj = self._sarima_res
        if tag == "sarima":
            fc = obj.get_forecast(steps=steps).predicted_mean  # type: ignore
            return np.asarray(fc)
        else:
            prof = obj  # type: ignore
            out = []
            m = start_month
            for _ in range(steps):
                out.append(prof[(m - 1) % 12])
                m += 1
            return np.asarray(out, dtype=float)

    def _build_residual_dataset(self, r_std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T = len(r_std)
        w = self._seq_len
        xs, ys = [], []
        for i in range(T - w):
            xs.append(r_std[i : i + w])
            ys.append(r_std[i + w])
        if not xs:
            return np.empty((0, w), dtype=np.float32), np.empty((0,), dtype=np.float32)
        X = np.asarray(xs, dtype=np.float32)
        y = np.asarray(ys, dtype=np.float32)
        return X, y

    def _fit_lstm(self, r_std: np.ndarray):
        # 条件：可用 torch 且开启 LSTM
        if not _HAS_TORCH or self.lstm_hidden <= 0:
            self._lstm = None
            return
        # 长度检查
        if len(r_std) < self._seq_len + 1:
            self._lstm = None
            self.used_fallback = True
            return

        # 构建样本，时间切分 90% 训练、10% 验证
        X, y = self._build_residual_dataset(r_std)
        if len(X) < 10:
            self._lstm = None
            self.used_fallback = True
            return
        n = len(X)
        n_train = max(1, int(n * 0.9))
        Xtr, ytr = X[:n_train], y[:n_train]
        Xval, yval = X[n_train:], y[n_train:]

        # 张量与设备
        device = torch.device("cpu")
        model = _ResLSTM(input_size=1, hidden=self.lstm_hidden, layers=self.lstm_layers).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        # 训练参数
        batch_size = 64
        best_val = float("inf")
        patience, bad = 5, 0

        def _to_batches(Xa, ya):
            idx = np.arange(len(Xa))
            np.random.shuffle(idx)
            for i in range(0, len(idx), batch_size):
                b = idx[i : i + batch_size]
                Xb = torch.from_numpy(Xa[b][:, :, None]).float().to(device)
                yb = torch.from_numpy(ya[b][:, None]).float().to(device)
                yield Xb, yb

        for ep in range(self.epochs):
            model.train()
            total_loss = 0.0
            for Xb, yb in _to_batches(Xtr, ytr):
                opt.zero_grad()
                pred = model(Xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += float(loss.item())

            # 验证
            model.eval()
            with torch.no_grad():
                if len(Xval) > 0:
                    Xv = torch.from_numpy(Xval[:, :, None]).float().to(device)
                    yv = torch.from_numpy(yval[:, None]).float().to(device)
                    pv = model(Xv)
                    vloss = float(loss_fn(pv, yv).item())
                else:
                    vloss = total_loss  # 无验证集时退化

            if vloss + 1e-8 < best_val:
                best_val = vloss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    # 早停
                    break

        # 恢复最佳
        if 'best_state' in locals():
            model.load_state_dict(best_state)
        self._lstm = model.eval()

    # --------------------- 公共接口 ---------------------
    def fit(self, series: pd.Series) -> "IceForecaster":
        # 确定性随机种子
        np.random.seed(42)
        if _HAS_TORCH:
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)

        s = self._prep_series(series)
        self._last_timestamp = s.index[-1]
        self._freq = s.index.freqstr or "MS"

        # 1) SARIMA 拟合（基于全部训练序列）
        self._fit_sarima(s)
        fitted = self._sarima_in_sample(s)
        # 2) 残差（严格使用 in-sample fittedvalues）
        r = (s.values - fitted).astype(float)
        # 标准化
        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=0))
        if not np.isfinite(sigma) or sigma < 1e-6:
            sigma = 1e-6
        r_std = (r - mu) / sigma
        self._lstm_scaler = (mu, sigma)
        # 保存滚动初始窗口
        if len(r_std) >= self._seq_len:
            self._last_r_std_tail = r_std[-self._seq_len :].astype(np.float32)
        else:
            self._last_r_std_tail = None

        # 3) 残差 LSTM 训练
        self._fit_lstm(r_std)

        # 4) HEDGE：用末尾 10% 作为验证段，估计 lambda_shrink
        self.lambda_shrink = 1.0
        try:
            H = max(1, int(round(len(s) * 0.1)))
            if self._lstm is not None and self._last_r_std_tail is not None and H >= 1 and len(s) > self._seq_len + H:
                # 切分：train_part / val_part
                s_train = s.iloc[:-H]
                s_val = s.iloc[-H:]
                # 临时 SARIMA 只基于 train_part
                if _HAS_STATSMODELS:
                    tmp_model = SARIMAX(
                        s_train.values,
                        order=self.sarima_order,
                        seasonal_order=self.sarima_seasonal,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    tmp_res = tmp_model.fit(disp=False)
                    sarima_val = np.asarray(tmp_res.get_forecast(steps=H).predicted_mean)
                else:
                    prof = self._seasonal_profile(s_train)
                    m0 = s_train.index[-1].month + 1
                    sarima_val = np.array([prof[(m0 - 1 + i) % 12] for i in range(H)], dtype=float)

                # 目标 a：真实值与 SARIMA 预测差
                a = s_val.values - sarima_val

                # 残差滚动预测（验证段）：窗口起点取“验证段开始前”的 r_std 序列
                # 为避免依赖 tmp_res 的残差，这里使用主模型的 r/std
                r_std_train = r_std[: -H]
                if len(r_std_train) >= self._seq_len:
                    win = r_std_train[-self._seq_len :].astype(np.float32).copy()
                else:
                    win = None

                if win is not None:
                    preds_std = []
                    device = torch.device("cpu")
                    with torch.no_grad():
                        for _ in range(H):
                            x = torch.from_numpy(win[None, :, None]).float().to(device)
                            y_std = float(self._lstm(x).cpu().numpy()[0, 0])
                            preds_std.append(y_std)
                            win = np.roll(win, -1)
                            win[-1] = y_std
                    b = np.asarray(preds_std, dtype=float) * sigma + mu  # 反标准化

                    # 最小二乘式缩放系数（夹在 [0,1]）
                    num = float(np.dot(a, b))
                    den = float(np.dot(b, b) + 1e-8)
                    lam = num / den if den > 0 else 0.0
                    lam = min(1.0, max(0.0, lam))

                    # 对比 RMSE，若无提升则置 0
                    rmse_base = self._rmse(s_val.values, sarima_val)
                    rmse_combo = self._rmse(s_val.values, sarima_val + lam * b)
                    if lam < 1e-6 or rmse_combo >= rmse_base - 1e-9:
                        lam = 0.0
                    self.lambda_shrink = float(lam)
                else:
                    self.lambda_shrink = 0.0
            else:
                # 无法计算 HEDGE，保守退化
                if self._lstm is None:
                    self.lambda_shrink = 0.0
        except Exception:
            # 出错则保守处理
            self.lambda_shrink = 0.0

        return self

    def predict(self, horizon_months: int) -> pd.Series:
        if self._last_timestamp is None:
            raise RuntimeError("请先 fit 再 predict")
        # SARIMA 预测线性部分
        start_next_month = (self._last_timestamp + pd.offsets.MonthBegin(1)).month
        sarima_fc = self._sarima_forecast(horizon_months, start_next_month)

        # 若禁用 LSTM 或回退，直接返回 SARIMA
        if self.lstm_hidden <= 0 or self._lstm is None or self._lstm_scaler is None or self._last_r_std_tail is None:
            y_fc = sarima_fc
        else:
            # 正确滚动预测残差（不喂真值）
            mu, sigma = self._lstm_scaler
            device = torch.device("cpu")
            win = self._last_r_std_tail.copy()  # 已在 fit 时标准化
            preds_std = []
            with torch.no_grad():
                for _ in range(horizon_months):
                    x = torch.from_numpy(win[None, :, None]).float().to(device)
                    y_std = float(self._lstm(x).cpu().numpy()[0, 0])
                    preds_std.append(y_std)
                    # 滚动窗口：只用模型自己的输出
                    win = np.roll(win, -1)
                    win[-1] = y_std
            resid_fc = np.asarray(preds_std, dtype=float) * sigma + mu
            y_fc = sarima_fc + self.lambda_shrink * resid_fc

        # 业务范围裁剪（SIC 取值范围）
        y_fc = np.clip(y_fc, 0.0, 1.0)
        idx = pd.date_range(self._last_timestamp + pd.offsets.MonthBegin(1), periods=horizon_months, freq=self._freq)
        return pd.Series(y_fc, index=idx)

    def save(self, path: str):
        # SARIMA 拟合结果若存在，用 pickle 序列化；回退模式存入 profile
        tag, obj = (None, None)
        if self._sarima_res is not None:
            tag, obj = self._sarima_res
        if tag == "sarima":
            sarima_bytes = pickle.dumps(obj)
        else:
            sarima_bytes = pickle.dumps((tag, obj))

        # LSTM 状态
        lstm_state = None
        if self._lstm is not None and _HAS_TORCH:
            lstm_state = self._lstm.state_dict()

        state = _SavedState(
            seasonal_period=self.seasonal_period,
            sarima_order=self.sarima_order,
            sarima_seasonal=self.sarima_seasonal,
            lstm_hidden=self.lstm_hidden,
            lstm_layers=self.lstm_layers,
            lr=self.lr,
            window=self._window,
            last_timestamp=self._last_timestamp,  # type: ignore
            freq=self._freq or "MS",
            sarima_fit_bytes=sarima_bytes,
            lstm_state_dict=lstm_state,
            lstm_scaler=self._lstm_scaler,
            last_r_std_tail=self._last_r_std_tail,
            lambda_shrink=self.lambda_shrink,
        )
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "IceForecaster":
        with open(path, "rb") as f:
            state: _SavedState = pickle.load(f)
        obj = cls(
            seasonal_period=state.seasonal_period,
            sarima_order=state.sarima_order,
            sarima_seasonal=state.sarima_seasonal,
            lstm_hidden=state.lstm_hidden,
            lstm_layers=state.lstm_layers,
            lr=state.lr,
        )
        obj._window = state.window
        obj._last_timestamp = state.last_timestamp
        obj._freq = state.freq

        # 恢复 SARIMA
        sarima_loaded = pickle.loads(state.sarima_fit_bytes) if state.sarima_fit_bytes is not None else None
        if isinstance(sarima_loaded, tuple) and sarima_loaded and sarima_loaded[0] == "seasonal_mean":
            obj._sarima_res = ("seasonal_mean", sarima_loaded[1])
        else:
            obj._sarima_res = ("sarima", sarima_loaded)

        # 恢复 LSTM
        obj._lstm_scaler = state.lstm_scaler
        obj._last_r_std_tail = state.last_r_std_tail
        obj.lambda_shrink = state.lambda_shrink if state.lambda_shrink is not None else 1.0
        if state.lstm_state_dict is not None and _HAS_TORCH and obj.lstm_hidden > 0:
            model = _ResLSTM(input_size=1, hidden=obj.lstm_hidden, layers=obj.lstm_layers)
            model.load_state_dict(state.lstm_state_dict)
            obj._lstm = model.eval()
        else:
            obj._lstm = None
        return obj
