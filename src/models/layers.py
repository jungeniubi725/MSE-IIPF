"""模型复用组件。对应论文中反复出现的几个算子。

  masked_mean       Pool_mean(·)，带 padding 掩码            E5, E13
  pool_cls          Pool_CLS(·)，取 <s> 位置                 E5, E13
  AttentionPool     α_i = softmax(wᵀtanh(W_u u_i)); Σα_i u_i  E17, E18
  ConsistencyGate   g = σ(W[a;b;|a−b|;a⊙b]+b)                E14, E15, E51
  CrossAttentionInject  LN(X + MHA(X, S, S))                 E11, E12
  ResidualMLP       两层 MLP + 残差，用作 R_m(·) 重建器       E44
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """(B, N, D) -> (B, D)。mask 为 (B, N)，1 有效 0 padding。"""
    if mask is None:
        return x.mean(dim=1)
    m = mask.to(x.dtype).unsqueeze(-1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)


def pool_cls(x: torch.Tensor) -> torch.Tensor:
    """RoBERTa 的 <s> 在位置 0（见 text_roberta.py 的落盘约定）。"""
    return x[:, 0]


class ProjectionHead(nn.Module):
    """E5 的 P_v / P_a / P_s：投影头 + L2 归一化。"""

    def __init__(self, d_in: int, d_out: int, hidden: int | None = None) -> None:
        super().__init__()
        h = hidden or d_in
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2norm(self.net(x))


class AttentionPool(nn.Module):
    """E17–E18 的注意力池化。padding 位在 softmax 前置 -inf。"""

    def __init__(self, d: int, hidden: int | None = None) -> None:
        super().__init__()
        h = hidden or d
        self.W_u = nn.Linear(d, h)
        self.w = nn.Linear(h, 1, bias=False)

    def forward(
        self, u: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """u: (B, N, D) -> (pooled (B,D), attn (B,N))"""
        score = self.w(torch.tanh(self.W_u(u))).squeeze(-1)  # (B, N)
        if mask is not None:
            score = score.masked_fill(mask == 0, float("-inf"))
            # 整行全 padding 时 softmax 会出 NaN，退化为均匀权重
            dead = (mask.sum(dim=1) == 0)
            if dead.any():
                score = score.masked_fill(dead.unsqueeze(-1), 0.0)
        attn = score.softmax(dim=-1)
        return torch.einsum("bn,bnd->bd", attn, u), attn


class ConsistencyGate(nn.Module):
    """E14/E15 的一致性门控，也用于 E51 的代理注入门控。

    输入两个全局向量 a、b，拼 [a; b; |a−b|; a⊙b]（4d）后映射到 d 维**通道级**门控
    （非标量），使模型可按维度决定注入强度。
    """

    def __init__(self, d: int, d_out: int | None = None) -> None:
        super().__init__()
        self.proj = nn.Linear(4 * d, d_out or d)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([a, b, (a - b).abs(), a * b], dim=-1)
        return torch.sigmoid(self.proj(feat))


class CrossAttentionInject(nn.Module):
    """E11–E12：以原始模态为 Query，语义流为 Key/Value，残差 + LN。"""

    def __init__(self, d: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.mha = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d)

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        s_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: (B,Nx,D) 原始流; s: (B,Ns,D) 语义流; s_mask: (B,Ns) 1 有效。"""
        kpm = None
        if s_mask is not None:
            kpm = s_mask == 0
            # 全 padding 行会让 MHA 输出 NaN，放开第 0 位兜底
            dead = kpm.all(dim=1)
            if dead.any():
                kpm = kpm.clone()
                kpm[dead, 0] = False
        c, _ = self.mha(self.q(x), self.k(s), self.v(s), key_padding_mask=kpm,
                        need_weights=False)
        return self.ln(x + c)


class ResidualMLP(nn.Module):
    """E44 的 R_m(·)：轻量两层感知机 + 残差。也用作转移预测器与循环映射网络。"""

    def __init__(self, d_in: int, d_out: int, hidden: int | None = None,
                 dropout: float = 0.0) -> None:
        super().__init__()
        h = hidden or max(d_in, d_out)
        self.net = nn.Sequential(
            nn.Linear(d_in, h), nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(h, d_out),
        )
        self.skip = nn.Identity() if d_in == d_out else nn.Linear(d_in, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class TransformerFusionBlock(nn.Module):
    """E28 的 F_fusion：标准 Transformer Encoder Block（MHSA + Res/LN + FFN）。"""

    def __init__(self, d: int, num_heads: int = 4, ffn_mult: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_mult * d), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ffn_mult * d, d),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, M, D)，M 为模态数。返回同形状。"""
        h, _ = self.mha(tokens, tokens, tokens, need_weights=False)
        x = self.ln1(tokens + h)
        return self.ln2(x + self.ffn(x))
