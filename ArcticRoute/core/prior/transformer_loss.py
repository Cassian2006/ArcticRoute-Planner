from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """SimCLR 风格的投影头：MLP -> L2 归一化。

    结构：Linear(d, d_hidden) -> BN -> ReLU -> Linear(d_hidden, d_out)
    默认 d_hidden = d_out = d。
    """
    def __init__(self, d_in: int, d_hidden: Optional[int] = None, d_out: Optional[int] = None):
        super().__init__()
        d_hidden = d_hidden or d_in
        d_out = d_out or d_in
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D]
        y = self.net(x)
        y = F.normalize(y, dim=-1, eps=1e-6)
        return y


def info_nce(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.1, projector: Optional[ProjectionHead] = None) -> Tuple[torch.Tensor, Dict[str, Any], torch.Tensor, torch.Tensor]:
    """SimCLR 风格的 InfoNCE 损失（双视图）。

    输入：
      - z_i, z_j: [B, D] 两个视角的表征（同一批对齐）
      - temperature: 温度参数（缩放 logits）
      - projector: 可选投影头（若提供将先投影再计算；否则直接用 z_i/z_j）

    返回：
      (loss, logs, p_i, p_j)
      - loss: 标量
      - logs: {"temperature": T, "pos_mean": ..., "neg_mean": ...}
      - p_i, p_j: 投影后向量（已 L2 归一化）
    """
    assert z_i.shape == z_j.shape and z_i.dim() == 2, "z_i/z_j 需要形状 [B, D] 且相同"
    B, D = z_i.shape
    if projector is not None:
        p_i = projector(z_i)
        p_j = projector(z_j)
    else:
        p_i = F.normalize(z_i, dim=-1)
        p_j = F.normalize(z_j, dim=-1)

    # 构造 2B 个样本：cat(p_i, p_j)
    Z = torch.cat([p_i, p_j], dim=0)  # [2B, D]
    sim = torch.matmul(Z, Z.t())  # [2B, 2B]
    # 去除对角自身匹配
    mask = torch.eye(2 * B, dtype=torch.bool, device=Z.device)
    sim = sim / float(temperature)

    # 对于每个 i，正样本是 i 与 i^B（视角切换）
    labels = torch.arange(B, device=Z.device)
    labels_j = labels + B
    labels = torch.cat([labels_j, labels], dim=0)  # 前半部分的正样本在后半部分对应行，后半部分反之

    # 构造 logits：对每一行，排除自身；
    logits = sim.clone()
    logits = logits.masked_fill(mask, float('-inf'))

    # 取正样本得分与负样本均值用于日志
    pos_scores = torch.diag(sim, diagonal=B)
    pos_scores = torch.cat([pos_scores, torch.diag(sim, diagonal=-B)], dim=0)  # [2B]
    neg_mask = ~mask
    neg_scores = torch.masked_select(sim, neg_mask).view(2 * B, 2 * B - 1)
    logs = {
        "temperature": float(temperature),
        "pos_mean": float(pos_scores.mean().detach().cpu().item()),
        "neg_mean": float(neg_scores.mean().detach().cpu().item()),
    }

    loss = F.cross_entropy(logits, labels)
    return loss, logs, p_i, p_j


def masked_imputation_loss(x: torch.Tensor, x_hat: torch.Tensor, mask: Optional[torch.Tensor] = None, mask_ratio: float = 0.15) -> Tuple[torch.Tensor, Dict[str, Any], torch.Tensor]:
    """掩码重建（MIM）损失：对 15% token 的特征向量做重建 MSE。

    输入：
      - x: [B,T,C] 原始特征
      - x_hat: [B,T,C] 重建特征
      - mask: [B,T] 可选有效掩码（True 表示有效位置）
      - mask_ratio: 掩码比例（默认 0.15）

    返回：(loss, logs, used_mask)
      - used_mask: 实际用于损失的位置 bool 掩码（[B,T]），便于上游可视化/统计
    """
    assert x.shape == x_hat.shape and x.dim() == 3, "x/x_hat 需要形状 [B,T,C] 且一致"
    B, T, C = x.shape
    device = x.device

    if mask is None:
        base_valid = torch.ones(B, T, dtype=torch.bool, device=device)
    else:
        base_valid = mask.bool()

    # 随机选择需要重建的位置（在有效 token 内均匀选择）
    num_valid = base_valid.sum().item()
    k = max(1, int(round(num_valid * float(mask_ratio))))
    # 构造概率向量并采样索引
    probs = base_valid.float().view(-1)
    if probs.sum() <= 0:
        # 没有有效位置，返回 0 损失
        zero = torch.zeros((), device=device, dtype=x.dtype)
        return zero, {"mse": 0.0, "mask_ratio": 0.0, "used": 0}, base_valid
    probs = probs / probs.sum()
    idx = torch.multinomial(probs, num_samples=k, replacement=False)  # [k]
    used_mask = torch.zeros(B * T, dtype=torch.bool, device=device)
    used_mask[idx] = True
    used_mask = used_mask.view(B, T)

    # 选择位置计算 MSE（按 C 聚合）
    diff = (x - x_hat) ** 2  # [B,T,C]
    mse_tok = diff.mean(dim=-1)  # [B,T]
    loss = (mse_tok * used_mask.float()).sum() / used_mask.float().sum()

    logs = {
        "mse": float(loss.detach().cpu().item()),
        "mask_ratio": float(mask_ratio),
        "used": int(used_mask.sum().detach().cpu().item()),
    }
    return loss, logs, used_mask


@dataclass
class LossConfig:
    temperature: float = 0.1
    lambda_mim: float = 0.5  # 总损失中的 MIM 系数


def total_loss(z_i: torch.Tensor, z_j: torch.Tensor, x: Optional[torch.Tensor] = None, x_hat: Optional[torch.Tensor] = None, *, projector: Optional[ProjectionHead] = None, cfg: Optional[LossConfig] = None, token_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """联合损失：L = L_NCE + λ·L_MIM。两者可独立/联合训练（缺失一项即仅另一项）。

    输入：
      - z_i, z_j: [B,D] 表征（两视角）
      - x, x_hat: [B,T,C]（可选）用于 MIM
      - projector: 投影头（可选），用于 NCE
      - cfg: LossConfig，提供 temperature 与 lambda_mim
      - token_mask: [B,T]（可选）MIM 有效 token 掩码
    返回：
      (loss, logs)；logs 含 nce/mim 分项与总损失
    """
    cfg = cfg or LossConfig()
    logs: Dict[str, Any] = {}

    # NCE
    loss_nce = None
    if z_i is not None and z_j is not None:
        loss_nce, log_nce, _, _ = info_nce(z_i, z_j, temperature=cfg.temperature, projector=projector)
        logs.update({f"nce.{k}": v for k, v in log_nce.items()})
        logs["nce.loss"] = float(loss_nce.detach().cpu().item())

    # MIM
    loss_mim = None
    if x is not None and x_hat is not None:
        loss_mim, log_mim, used_mask = masked_imputation_loss(x, x_hat, mask=token_mask)
        logs.update({f"mim.{k}": v for k, v in log_mim.items()})
        logs["mim.used"] = int(used_mask.sum().detach().cpu().item())
        logs["mim.loss"] = float(loss_mim.detach().cpu().item())

    # 汇总
    if loss_nce is not None and loss_mim is not None:
        loss = loss_nce + float(cfg.lambda_mim) * loss_mim
    elif loss_nce is not None:
        loss = loss_nce
    elif loss_mim is not None:
        loss = loss_mim
    else:
        raise ValueError("必须至少提供 NCE (z_i,z_j) 或 MIM (x,x_hat) 之一")

    logs["loss.total"] = float(loss.detach().cpu().item())
    logs["lambda_mim"] = float(cfg.lambda_mim)
    return loss, logs


__all__ = [
    "ProjectionHead",
    "info_nce",
    "masked_imputation_loss",
    "LossConfig",
    "total_loss",
]




