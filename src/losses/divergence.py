"""散度与一致性度量。

  js_divergence   E31: D_JS(p,q) = ½D_KL(p‖m) + ½D_KL(q‖m), m=(p+q)/2
  js_distance     E50 用的是「Jensen-Shannon **距离**」= sqrt(D_JS)
  conflict_kappa  E50: κ_t = ⅓[D_JS(p_v,p_a)+D_JS(p_v,p_l)+D_JS(p_a,p_l)]

关于归一化到 [0,1]（原文 E51 要求 κ_t ∈ [0,1]）
------------------------------------------------
以 **log2 为底** 时 D_JS ∈ [0,1]，其平方根同样 ∈ [0,1]，无需再额外缩放。
若用自然对数，上界是 ln2 ≈ 0.693，κ 会系统性偏小。
（注：主模型 L_cal 已改为正确性硬目标，κ 仅作 T7 的诊断量，不再进入训练。）
故本实现固定 log2 底，`use_sqrt` 控制用距离(E50 字面)还是散度。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_LOG2 = 0.6931471805599453


def js_divergence(
    p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8, base2: bool = True
) -> torch.Tensor:
    """D_JS(p,q)，最后一维为类别维。返回 (...) 形状（少一维）。

    p、q 须是概率分布（已 softmax）。base2=True 时值域 [0,1]。
    """
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    js = 0.5 * (kl_pm + kl_qm)
    if base2:
        js = js / _LOG2
    return js.clamp(0.0, 1.0)


def js_distance(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """E50 字面意义的 Jensen-Shannon 距离 = sqrt(D_JS)，log2 底下值域 [0,1]。"""
    return js_divergence(p, q, eps=eps, base2=True).clamp_min(0.0).sqrt()


def conflict_kappa(
    p_v: torch.Tensor,
    p_a: torch.Tensor,
    p_l: torch.Tensor,
    use_sqrt: bool = True,
) -> torch.Tensor:
    """E50：三模态冲突程度 κ_t ∈ [0,1]。三个 JS 项取均值。

    use_sqrt=True 用 JS 距离（原文写的是「距离」），False 用 JS 散度。
    """
    f = js_distance if use_sqrt else js_divergence
    return ((f(p_v, p_a) + f(p_v, p_l) + f(p_a, p_l)) / 3.0).clamp(0.0, 1.0)


def dist_consistency_loss(
    p_modal: torch.Tensor,
    p_proxy: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """E39：L_dist = 1/(BT) Σ Σ_m ω_m,t · D_JS(p_m,t, sg(p_z,t))。

    p_modal: (..., M, K) 各模态分布；p_proxy: (..., K) 代理分布（调用方已 detach）。
    weight:  (..., M) 模态可靠性权重 ω。
    """
    js = js_divergence(p_modal, p_proxy.unsqueeze(-2).expand_as(p_modal))  # (..., M)
    if weight is None:
        return js.mean()
    w = weight.to(js.dtype)
    return (w * js).sum() / w.sum().clamp_min(1e-6)


def kl_dirichlet_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """E47：D_KL(Dir(α) ‖ Dir(1))，闭式解。alpha: (..., K) -> (...)。

        log[Γ(Σα_k) / (Γ(K)·Π Γ(α_k))] + Σ(α_k − 1)[ψ(α_k) − ψ(Σα_j)]

    全部用 lgamma/digamma 在 log 域计算，避免 Gamma 溢出。
    """
    K = alpha.size(-1)
    S = alpha.sum(dim=-1, keepdim=True)  # (..., 1)
    term1 = (
        torch.lgamma(S.squeeze(-1))
        - torch.lgamma(torch.tensor(float(K), device=alpha.device, dtype=alpha.dtype))
        - torch.lgamma(alpha).sum(dim=-1)
    )
    term2 = ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S))).sum(dim=-1)
    return term1 + term2
