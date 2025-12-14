from __future__ import annotations

"""
最小 EDL 训练脚本（E0.3）

- 从 data_real/edl_training/edl_train.parquet 读取一个小子集（默认 10k）
- 简单标准化（sic / wave_swh）
- 划分 train/val（8/2）
- 训练一个极小 MLP + Evidential（Dirichlet）分类头（2 类：risky=0, safe=1）
- 训练若干 epoch，打印 train/val loss 和 accuracy
- 保存模型为 models/edl_small_demo.pt（包含模型权重与标准化参数）

用法：
    python -m arcticroute.edl_training.train_small_edl --max-samples 10000 --epochs 5
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# 可选依赖：PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyTorch 未安装，请先安装 torch>=2.0.0") from e


DEFAULT_PARQUET = Path("data_real/edl_training/edl_train.parquet")
DEFAULT_MODEL_PATH = Path("models/edl_small_demo.pt")

# 我们仅使用推理时可获得的特征：sic, wave_swh, ice_thickness_m, lat, lon
FEATURE_ORDER: List[str] = [
    "sic", "wave_swh", "ice_thickness_m", "lat", "lon"
]
STD_COLS: List[str] = ["sic", "wave_swh"]  # 简单标准化列
LABEL_COL = "label_safe_risky"  # 0=risky, 1=safe
N_CLASSES = 2


@dataclass
class DataSplit:
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    mean: np.ndarray
    std: np.ndarray


class SmallMLP_EDL(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, int] = (32, 16), num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], num_classes)  # 输出 evidence logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        evidence = F.softplus(logits)  # evidence >= 0
        alpha = evidence + 1.0  # Dirichlet parameters
        return alpha


def edl_mse_loss(alpha: torch.Tensor, y_onehot: torch.Tensor, kl_strength: float = 1e-3) -> torch.Tensor:
    """EDL 经典损失（基于 MSE + KL 正则），参考 Sensoy et al., 2018。

    Args:
        alpha: Dirichlet 参数 (N, K)
        y_onehot: one-hot 标签 (N, K)
        kl_strength: KL 正则系数
    Returns:
        标量损失
    """
    S = torch.sum(alpha, dim=1, keepdim=True)  # (N,1)
    p = alpha / S  # 期望类别概率 (N,K)

    # Mean Square Error term
    mse = torch.sum((y_onehot - p) ** 2, dim=1, keepdim=True)
    var = torch.sum(p * (1 - p) / (S + 1), dim=1, keepdim=True)
    loss_err = mse + var

    # KL(Dir(alpha) || Dir(1))
    K = alpha.size(1)
    ones = torch.ones_like(alpha)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    sum_ones = torch.sum(ones, dim=1, keepdim=True)

    lnB_alpha = torch.lgamma(sum_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_one = torch.lgamma(sum_ones) - torch.sum(torch.lgamma(ones), dim=1, keepdim=True)

    digamma_sum = torch.digamma(sum_alpha)
    digamma_alpha = torch.digamma(alpha)

    kl = torch.sum((alpha - ones) * (digamma_alpha - digamma_sum), dim=1, keepdim=True) + lnB_alpha - lnB_one

    loss = torch.mean(loss_err + kl_strength * kl)
    return loss


def _prepare_data(parquet_path: Path, max_samples: int = 10_000, seed: int = 42) -> DataSplit:
    df = pd.read_parquet(parquet_path)

    # 仅保留需要的列
    cols = FEATURE_ORDER + [LABEL_COL]
    df = df[[c for c in cols if c in df.columns]].copy()

    # 丢弃缺失与异常
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df[LABEL_COL].isin([0, 1])]

    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed)

    # 简单标准化
    mean = df[STD_COLS].mean().values.astype(np.float32)
    std = df[STD_COLS].std(ddof=0).replace(0, 1.0).values.astype(np.float32)

    df_norm = df.copy()
    for i, col in enumerate(STD_COLS):
        df_norm[col] = (df_norm[col].values.astype(np.float32) - mean[i]) / std[i]

    X = df_norm[FEATURE_ORDER].values.astype(np.float32)
    y = df_norm[LABEL_COL].values.astype(np.int64)

    # 划分 train / val
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_train = int(len(X) * 0.8)
    tr_idx, va_idx = idx[:n_train], idx[n_train:]

    X_train = torch.from_numpy(X[tr_idx])
    y_train = torch.from_numpy(y[tr_idx])
    X_val = torch.from_numpy(X[va_idx])
    y_val = torch.from_numpy(y[va_idx])

    return DataSplit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, mean=mean, std=std)


def train_small_edl(
    parquet_path: Path = DEFAULT_PARQUET,
    model_out: Path = DEFAULT_MODEL_PATH,
    max_samples: int = 10_000,
    epochs: int = 5,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
) -> Path:
    torch.manual_seed(seed)
    ds = _prepare_data(parquet_path, max_samples=max_samples, seed=seed)

    model = SmallMLP_EDL(in_dim=len(FEATURE_ORDER), hidden=(32, 16), num_classes=N_CLASSES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def _accuracy(alpha: torch.Tensor, y: torch.Tensor) -> float:
        p = alpha / torch.sum(alpha, dim=1, keepdim=True)
        pred = torch.argmax(p, dim=1)
        return float((pred == y).float().mean().item())

    # 训练循环
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(ds.X_train.size(0))
        Xtr = ds.X_train[perm].to(device)
        ytr = ds.y_train[perm].to(device)

        total_loss = 0.0
        total_acc = 0.0
        n_batches = int(np.ceil(len(Xtr) / batch_size))

        for b in range(n_batches):
            s = b * batch_size
            e = min((b + 1) * batch_size, len(Xtr))
            xb = Xtr[s:e]
            yb = ytr[s:e]

            alpha = model(xb)
            y_onehot = F.one_hot(yb, num_classes=N_CLASSES).float()
            loss = edl_mse_loss(alpha, y_onehot, kl_strength=1e-3)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * (e - s)
            total_acc += _accuracy(alpha.detach(), yb) * (e - s)

        train_loss = total_loss / len(Xtr)
        train_acc = total_acc / len(Xtr)

        # 验证
        model.eval()
        with torch.no_grad():
            Xv = ds.X_val.to(device)
            yv = ds.y_val.to(device)
            alpha_v = model(Xv)
            yv_onehot = F.one_hot(yv, num_classes=N_CLASSES).float()
            val_loss = edl_mse_loss(alpha_v, yv_onehot, kl_strength=1e-3).item()
            val_acc = _accuracy(alpha_v, yv)

        print(f"[EDL_TRAIN] epoch {ep:02d}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    # 保存模型
    model_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "feature_order": FEATURE_ORDER,
        "std_cols": STD_COLS,
        "mean": ds.mean,
        "std": ds.std,
        "hidden": (32, 16),
        "num_classes": N_CLASSES,
    }
    torch.save(payload, model_out)
    print(f"[EDL_TRAIN] saved small EDL model to {model_out}")

    return model_out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train a small EDL model on edl_train.parquet")
    p.add_argument("--data", type=str, default=str(DEFAULT_PARQUET))
    p.add_argument("--out", type=str, default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--max-samples", type=int, default=10_000)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu")

    args = p.parse_args()

    train_small_edl(
        parquet_path=Path(args.data),
        model_out=Path(args.out),
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )



