from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchinfo import summary as torchinfo_summary  # type: ignore
except Exception:  # pragma: no cover
    torchinfo_summary = None  # type: ignore


@dataclass
class ModelConfig:
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    ffn_mult: int = 4
    dropout: float = 0.1
    rope: bool = True           # True -> 使用 RoPE；False -> 期望上游提供 t2v 并与 x 拼接或加法
    rope_dim: int = 64          # 参与旋转的维度（偶数，<= d_model）
    checkpoint: bool = True     # 激活检查点以省显存


class PreNorm(nn.Module):
    def __init__(self, d: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 q,k 应用 RoPE 旋转。
    输入：
      - q,k: [B, nH, T, Hd]
      - cos,sin: [T, rope_dim/2]
    仅对前 rope_dim 进行旋转（必须偶数），其余维度保持不变。
    """
    if rope_dim <= 0:
        return q, k
    Hd = q.size(-1)
    rd = min(rope_dim, Hd)
    rd = rd - (rd % 2)
    if rd <= 0:
        return q, k
    # 取前 rd 维，拆成偶奇配对
    def rot(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :rd]
        x2 = x[..., rd:]
        x1 = x1.view(*x1.shape[:-1], rd // 2, 2)
        x_even = x1[..., 0]
        x_odd = x1[..., 1]
        # 广播 cos/sin -> [1,1,T, rd/2]
        # 兼容上游提供更长的 RoPE 维度（例如按 d_model/2 生成），此处裁剪到 rd/2
        c_full = cos.view(1, 1, cos.shape[0], cos.shape[1])
        s_full = sin.view(1, 1, sin.shape[0], sin.shape[1])
        c = c_full[..., : (rd // 2)]
        s = s_full[..., : (rd // 2)]
        x_rot_even = x_even * c - x_odd * s
        x_rot_odd = x_even * s + x_odd * c
        x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1).view(*x.shape[:-1], rd)
        return torch.cat([x_rot, x2], dim=-1)
    return rot(q), rot(k)


class MHA_SDPA(nn.Module):
    """自定义多头注意力，使用 PyTorch 2 F.scaled_dot_product_attention，支持 RoPE。"""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, rope: bool = False, rope_dim: int = 64):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout
        self.use_rope = rope
        self.rope_dim = rope_dim
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, rope_cos: Optional[torch.Tensor] = None, rope_sin: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B,T,D]
        B, T, D = x.shape
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        # [B, nH, T, Hd]
        def reshape_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.nhead, self.head_dim).permute(0, 2, 1, 3)
        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)
        if self.use_rope and rope_cos is not None and rope_sin is not None:
            q, k = _apply_rope(q, k, rope_cos, rope_sin, self.rope_dim)
        # key padding mask -> attention mask of shape [B, 1, 1, T]
        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: [B, T] True 表示有效；需要无效位置为 -inf
            invalid = ~key_padding_mask.bool()
            attn_mask = invalid.view(B, 1, 1, T)
        # scaled dot product attention
        # F.scaled_dot_product_attention 接受 attn_mask bool 时会屏蔽；或浮点掩码。这里传 bool。
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        # [B, nH, T, Hd] -> [B,T,D]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        out = self.Wo(out)
        out = self.attn_drop(out)
        return out


class FFN(nn.Module):
    def __init__(self, d_model: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = d_model * mult
        self.net = nn.Sequential(
            nn.Linear(d_model, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = PreNorm(cfg.d_model, MHA_SDPA(cfg.d_model, cfg.nhead, cfg.dropout, rope=cfg.rope, rope_dim=cfg.rope_dim))
        self.ffn = PreNorm(cfg.d_model, FFN(cfg.d_model, cfg.ffn_mult, cfg.dropout))

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, rope_cos: Optional[torch.Tensor] = None, rope_sin: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(x, key_padding_mask=key_padding_mask, rope_cos=rope_cos, rope_sin=rope_sin)
        x = x + self.ffn(x)
        return x


class TrajectoryTransformer(nn.Module):
    """轻量 Encoder-only Transformer（Pre-Norm）。

    输入：
      - x: [B,T,C] 特征（来自 transformer_feats.build_sequence_features）
      - mask: [B,T] True 有效
      - posenc: {"type":"rope","cos":[T,D/2],"sin":[T,D/2]} 或 {"type":"t2v","emb":[T,K]}
    """
    def __init__(self, d_in: int, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.proj = nn.Linear(d_in, self.cfg.d_model)
        self.drop = nn.Dropout(self.cfg.dropout)
        self.layers = nn.ModuleList([EncoderLayer(self.cfg) for _ in range(self.cfg.num_layers)])
        self.norm = nn.LayerNorm(self.cfg.d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, posenc: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        # x: [B,T,C]
        B, T, C = x.shape
        h = self.proj(x)
        h = self.drop(h)
        key_padding_mask = mask if mask is not None else torch.ones(B, T, dtype=torch.bool, device=x.device)
        rope_cos = rope_sin = None
        if self.cfg.rope and posenc is not None and posenc.get("type") == "rope":
            # [T, D/2]
            rope_cos = torch.as_tensor(posenc.get("cos"), dtype=h.dtype, device=h.device)
            rope_sin = torch.as_tensor(posenc.get("sin"), dtype=h.dtype, device=h.device)
        # 逐层（可选激活检查点）
        if self.cfg.checkpoint and torch.is_grad_enabled():
            for layer in self.layers:
                def _fn(inp):
                    return layer(inp, key_padding_mask=key_padding_mask, rope_cos=rope_cos, rope_sin=rope_sin)
                h = torch.utils.checkpoint.checkpoint(_fn, h, use_reentrant=False)
        else:
            for layer in self.layers:
                h = layer(h, key_padding_mask=key_padding_mask, rope_cos=rope_cos, rope_sin=rope_sin)
        h = self.norm(h)
        return h  # [B,T,D]


def model_summary_example() -> Optional[str]:
    """返回一个基于 torchinfo 的模型概览（如可用）。用于 DOD 检查。
    示例设定：batch=16, seq=512, d_in=9。
    """
    if torchinfo_summary is None:
        return None
    cfg = ModelConfig()
    m = TrajectoryTransformer(d_in=9, cfg=cfg)
    dummy = torch.zeros(16, 512, 9)
    mask = torch.ones(16, 512, dtype=torch.bool)
    # 生成 RoPE 位置编码
    D = cfg.rope_dim if cfg.rope_dim > 0 else 64
    pos = torch.arange(512, dtype=torch.float32)
    inv_freq = (cfg.rope and 10000.0) ** (-torch.arange(0, D, 2).float() / D)
    if cfg.rope:
        angles = pos[:, None] * inv_freq[None, :]
        posenc = {"type": "rope", "cos": torch.cos(angles), "sin": torch.sin(angles)}
    else:
        posenc = None
    txt = str(torchinfo_summary(m, input_data=(dummy, mask, posenc), verbose=0))
    return txt


__all__ = ["ModelConfig", "TrajectoryTransformer", "model_summary_example"]


