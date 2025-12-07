from __future__ import annotations
"""
Phase L Experts 占位：
- 提供轻量 Adapter 接口用于每个 bucket 的微调/缩放
- 如无专用权重，回退到全局专家（UNet-Former）
"""
from typing import Optional, Dict, Any

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


class LogitAffineAdapter(nn.Module):
    """对概率先做 logit 再做仿射，再 sigmoid 回写。
    y = sigmoid(a * logit(p) + b)
    仅作为占位，真实实现可替换为 LoRA/Adapter 层。
    """
    def __init__(self, a: float = 1.0, b: float = 0.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a)))  # type: ignore
        self.b = nn.Parameter(torch.tensor(float(b)))  # type: ignore

    def forward(self, prob_map):  # prob_map: [B,1,H,W]
        import torch.nn.functional as F  # type: ignore
        eps = 1e-6
        p = prob_map.clamp(eps, 1 - eps)
        x = torch.log(p) - torch.log(1 - p)
        y = self.a * x + self.b
        return torch.sigmoid(y)


def load_adapter_for_bucket(bucket: str, state_dir: Optional[str] = None) -> Optional[nn.Module]:
    """按桶加载 Adapter 权重。若不存在，返回 None。
    state_dir: 目录下期望有 expert_<bucket>.ckpt
    """
    import os
    if torch is None:
        return None
    if not state_dir:
        return None
    ckpt = os.path.join(state_dir, f"expert_{bucket}.ckpt")
    if not os.path.exists(ckpt):
        return None
    try:
        state = torch.load(ckpt, map_location="cpu")
        model = LogitAffineAdapter()
        model.load_state_dict(state)
        return model
    except Exception:
        return None


__all__ = ["LogitAffineAdapter", "load_adapter_for_bucket"]

