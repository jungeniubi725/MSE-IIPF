"""对比损失。论文中共出现四处形式，此处统一实现。

  bidirectional_infonce  E7, E8    双向，系数 1/(2N)，用于 L_DSA
  single_infonce         E26       单向，系数 1/B，用于 L_ITAC 的三个配对
  predictive_infonce     E43, E45  预测/重建 vs. stop-grad 目标，负样本取 batch 内其他样本
                                   用于 L_TPC 与 L_rec^m

注意 E7/E8 与 E26 的**系数与方向不同**，论文里是两种写法，不要合并：
  E7 : L_v,sv = −1/(2N) Σ_i [ log p(i→i) + log p'(i→i) ]   双向对称
  E26: L_v,t  = −1/B   Σ_i   log p(i→i)                    单向
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _sim(a: torch.Tensor, b: torch.Tensor, tau: float) -> torch.Tensor:
    """s_ij = a_i·b_j / τ。a、b 须已 L2 归一化（E6/E25 的前置条件）。"""
    return a @ b.t() / max(tau, 1e-6)


def single_infonce(
    a: torch.Tensor,
    b: torch.Tensor,
    tau: float = 0.07,
    false_neg_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """E26：单向 InfoNCE，系数 1/B。对角为正样本。

    false_neg_mask: (B,B) bool，True 表示该位置是**伪负样本**（不同样本但同标签），
    在 softmax 前置 -inf 屏蔽掉，对应原文 L98 的伪负样本抑制要求。
    """
    logits = _sim(a, b, tau)
    if false_neg_mask is not None:
        eye = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(false_neg_mask & ~eye, float("-inf"))
    target = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, target)


def bidirectional_infonce(
    a: torch.Tensor,
    b: torch.Tensor,
    tau: float = 0.07,
    false_neg_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """E7/E8：双向对称 InfoNCE。

    F.cross_entropy 内含 1/N 平均，两向求和后除 2 即得原文的 1/(2N)。
    """
    ab = single_infonce(a, b, tau, false_neg_mask)
    ba = single_infonce(b, a, tau, false_neg_mask)
    return 0.5 * (ab + ba)


def predictive_infonce(
    pred: torch.Tensor,
    target: torch.Tensor,
    tau: float = 0.07,
    normalize: bool = True,
) -> torch.Tensor:
    """E43/E45：预测值对 stop-grad 目标的对比损失。

    pred[b] 与 target[b] 构成正样本，pred[b] 与 target[j≠b] 构成负样本。
    target 在此**必须已 detach**（调用方负责，对应原文的 sg(·)）。
    normalize=True 时先 L2 归一化，使 sim 为余弦相似度（原文 E43 的 sim(·,·)）。
    """
    if normalize:
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
    logits = _sim(pred, target, tau)
    idx = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, idx)


def cosine_align_loss(
    x: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """E38/E47 的 1 − cos(x, sg(target)) 形式，可带逐样本权重 ω。

    x, target: (..., D)；weight: (...,) 与前置维度对齐。
    """
    c = 1.0 - F.cosine_similarity(x, target, dim=-1)
    if weight is None:
        return c.mean()
    w = weight.to(c.dtype)
    return (w * c).sum() / w.sum().clamp_min(1e-6)


def build_false_neg_mask(labels: torch.Tensor) -> torch.Tensor:
    """(B,) 标签 -> (B,B) bool，True 表示同标签（潜在伪负样本）。

    回归标签（float）不适用，调用方需自行判断是否传入。
    """
    return labels.view(-1, 1) == labels.view(1, -1)
