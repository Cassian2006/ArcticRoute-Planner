from __future__ import annotations
"""
MoE 门控（Gate）
- 输入：prior embedding 向量 + 上下文 one-hot（region/season/vessel）
- 输出：K 个专家的 softmax 权重，可带温度缩放与掩码

REUSE: 仅用 PyTorch MLP，小巧稳定。
"""
from typing import Optional, Dict, Any
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


class GateMLP(nn.Module):
    def __init__(self, in_dim: int, k: int = 3, hidden: int = 64, temperature: float = 1.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, k)
        self.temperature = float(max(1e-3, temperature))
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = F.silu(self.fc1(x))
        logits = self.fc2(h) / self.temperature
        if mask is not None:
            logits = logits.masked_fill(mask <= 0, float('-inf'))
        w = torch.softmax(logits, dim=-1)
        # 数值安全：NaN -> 均匀
        w = torch.where(torch.isfinite(w), w, torch.full_like(w, 1.0 / w.shape[-1]))
        return w


def make_context(region_1h, season_1h, vessel_1h, prior_embed: Optional[Any] = None) -> Any:
    import numpy as np
    parts = []
    if prior_embed is not None:
        pe = np.asarray(prior_embed, dtype=np.float32).ravel()
        parts.append(pe)
    parts.append(np.asarray(region_1h, dtype=np.float32).ravel())
    parts.append(np.asarray(season_1h, dtype=np.float32).ravel())
    parts.append(np.asarray(vessel_1h, dtype=np.float32).ravel())
    x = np.concatenate(parts, axis=0)
    return x


def gate_weights(prior_embed, region_1h, season_1h, vessel_1h, k: int = 3, temperature: float = 1.0, mask: Optional[Any] = None, model_state: Optional[Dict[str, Any]] = None) -> Any:
    """一次性计算单样本门控权重（无训练路径）。
    - model_state: 可选，传入已训练 GateMLP 的 state_dict；若为空使用随机初始化但稳定输出（softmax 仍有效）
    - mask: 可选 [K] 0/1 掩码
    返回：np.ndarray[K]，和为1。
    """
    import numpy as np
    x = make_context(region_1h=region_1h, season_1h=season_1h, vessel_1h=vessel_1h, prior_embed=prior_embed)
    if torch is None:
        # 退化：均匀分配
        w = np.ones((k,), dtype=np.float32) / float(k)
        if mask is not None:
            m = np.asarray(mask, dtype=np.float32)
            m = (m > 0).astype(np.float32)
            if m.sum() > 0:
                w = (w * m); w = w / max(1e-6, w.sum())
        return w
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
        model = GateMLP(in_dim=x_t.shape[-1], k=k, hidden=64, temperature=temperature)
        if model_state:
            try:
                model.load_state_dict(model_state)  # type: ignore
            except Exception:
                pass
        model.eval()
        mask_t = None
        if mask is not None:
            mask_t = torch.tensor(mask, dtype=torch.float32).view(1, -1)
        w = model(x_t, mask=mask_t)[0].cpu().numpy()
        return w.astype(np.float32)


__all__ = ["GateMLP", "gate_weights", "make_context"]

