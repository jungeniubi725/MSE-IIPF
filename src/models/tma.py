"""Tri-Modal Affective Alignment（论文 §3.3，公式 E20–E32）。

三者是**串行数据流**（原文 L99 明确），不是并列堆叠：
    MAM（调输入质量）→ ITAC（三模态语义是否同一情感）→ ITAM（整体组合是否一致）

关于 L_MAM
---------
论文正文只在 E32 的加权和里出现 L_MAM，未给显式定义。本实现按两项落地：
  1. 保留率正则 ‖mean(r_m) − ρ_target‖²：防止退化为「全保留」或「全丢弃」；
  2. 掩码前后辅助情感分布的 JS 一致性：对应原文「在部分可靠信息缺失情况下
     仍保持情感语义一致性」的设计意图（L83）。
硬阈值 M = 1[r > γ] 不可导，用 straight-through estimator 让梯度流回 r。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses.contrastive import build_false_neg_mask, single_infonce
from ..losses.divergence import js_divergence
from .layers import TransformerFusionBlock, l2norm, masked_mean

MODALITIES = ("v", "a", "l")


class ModalityAwareMasking(nn.Module):
    """E21–E23 + L_MAM。逐 token 估计可靠性并硬掩码。"""

    def __init__(self, d: int, gamma: float = 0.5, keep_target: float = 0.8) -> None:
        super().__init__()
        self.gamma = gamma
        self.keep_target = keep_target
        self.W_r = nn.Linear(d, 1)  # E21: W_r ∈ R^{1×d}, b_r

    def forward(
        self, z: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """z: (B,N,d) -> (掩码后特征, 可靠性 r (B,N), 保留门 (B,N))"""
        r = torch.sigmoid(self.W_r(z)).squeeze(-1)  # E21
        hard = (r > self.gamma).to(z.dtype)  # E22
        # straight-through：前向用硬 0/1，反向走 r 的梯度
        keep = hard + r - r.detach()
        if mask is not None:
            keep = keep * mask.to(z.dtype)
        # 整行被全部掩掉会让后续池化出 NaN，至少保留可靠性最高的一个 token
        if mask is not None:
            valid = mask.to(z.dtype)
        else:
            valid = torch.ones_like(r)
        dead = (keep.detach() * valid).sum(dim=1) < 1e-6
        if dead.any():
            top = (r.masked_fill(valid == 0, -1e4)).argmax(dim=1)
            rescue = F.one_hot(top, r.size(1)).to(z.dtype)
            keep = torch.where(dead.unsqueeze(-1), keep + rescue, keep)
        return z * keep.unsqueeze(-1), r, keep  # E23


class TriModalAlignment(nn.Module):
    def __init__(
        self,
        d_in: int = 256,
        d: int = 256,
        num_heads: int = 4,
        tau: float = 0.07,
        gamma: float = 0.5,
        keep_target: float = 0.8,
        itac_weights: tuple[float, float, float] = (0.4, 0.4, 0.2),  # α, β, η
        num_classes: int = 8,
        use_mam: bool = True,  # 消融 A4
        use_itac: bool = True,  # 消融 A5
        use_itam: bool = True,  # 消融 A6
        suppress_false_neg: bool = True,  # 原文 L98 伪负样本抑制
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.alpha, self.beta, self.eta = itac_weights
        self.use_mam, self.use_itac, self.use_itam = use_mam, use_itac, use_itam
        self.suppress_false_neg = suppress_false_neg

        # E20：三模态线性投影到共享情感语义空间
        self.proj = nn.ModuleDict({m: nn.Linear(d_in, d) for m in MODALITIES})
        # E21–E23：每模态独立的 MAM
        self.mam = nn.ModuleDict(
            {m: ModalityAwareMasking(d, gamma, keep_target) for m in MODALITIES}
        )
        # L_MAM 第二项用的共享辅助头（掩码前后分布一致性）
        self.aux_head = nn.Linear(d, num_classes)
        # E28–E29：ITAM 的融合块与匹配分类器
        self.fusion = TransformerFusionBlock(d, num_heads, dropout=dropout)
        self.match_head = nn.Linear(d, 1)

    # ------------------------------------------------------------ MAM
    def _apply_mam(
        self, z: dict[str, torch.Tensor], masks: dict[str, torch.Tensor | None]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        out, reg_terms = {}, []
        for m in MODALITIES:
            zm, mk = z[m], masks.get(m)
            zm_masked, r, keep = self.mam[m](zm, mk)
            out[m] = zm_masked
            # 项 1：保留率正则
            valid = mk.to(r.dtype) if mk is not None else torch.ones_like(r)
            keep_rate = (keep.detach() * valid).sum() / valid.sum().clamp_min(1e-6)
            rate = (r * valid).sum() / valid.sum().clamp_min(1e-6)
            reg_terms.append((rate - self.mam[m].keep_target) ** 2)
            # 项 2：掩码前后辅助分布 JS 一致性
            p_before = self.aux_head(masked_mean(zm, mk)).softmax(-1)
            p_after = self.aux_head(masked_mean(zm_masked, mk)).softmax(-1)
            reg_terms.append(js_divergence(p_after, p_before.detach()).mean())
            out[f"keep_rate_{m}"] = keep_rate.detach()  # 仅作日志，不参与反传
        loss = torch.stack([t for t in reg_terms]).mean()
        return out, loss

    # ------------------------------------------------------------ ITAM
    def _itam(
        self,
        g: dict[str, torch.Tensor],
        labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """E28–E31。正样本为同样本三元组，负样本替换 1 个模态。

        负样本优先取**不同情感类别**的样本（原文 L98），避免伪负样本。
        """
        B = g["v"].size(0)
        dev = g["v"].device
        # 为每个 anchor 选一个替换源 j
        if labels is not None and self.suppress_false_neg and labels.dtype in (
            torch.int32, torch.int64
        ):
            diff = labels.view(-1, 1) != labels.view(1, -1)  # (B,B) 不同标签
        else:
            diff = ~torch.eye(B, dtype=torch.bool, device=dev)
        # 每行按均匀分布在候选中采样；无候选则退化为 roll(1)
        w = diff.float()
        no_cand = w.sum(dim=1) < 1e-6
        if no_cand.any():
            fallback = (~torch.eye(B, dtype=torch.bool, device=dev)).float()
            w = torch.where(no_cand.unsqueeze(-1), fallback, w)
        if w.sum() < 1e-6:  # B == 1，无法构负样本
            j = torch.zeros(B, dtype=torch.long, device=dev)
        else:
            j = torch.multinomial(w.clamp_min(1e-8), 1).squeeze(-1)

        # 正样本 + 两种负样本（替换 a / 替换 l），对应原文 (V_i,A_j,T_i)、(V_i,A_i,T_j)
        pos = torch.stack([g["v"], g["a"], g["l"]], dim=1)
        neg_a = torch.stack([g["v"], g["a"][j], g["l"]], dim=1)
        neg_l = torch.stack([g["v"], g["a"], g["l"][j]], dim=1)
        trips = torch.cat([pos, neg_a, neg_l], dim=0)  # (3B, 3, d)
        h = self.fusion(trips)[:, 0]  # E28：取首 token
        logit = self.match_head(h).squeeze(-1)  # E29 的 logit（σ 交给 BCE）
        y = torch.cat([
            torch.ones(B, device=dev), torch.zeros(2 * B, device=dev)
        ])
        return F.binary_cross_entropy_with_logits(logit, y), torch.sigmoid(logit[:B])

    # ------------------------------------------------------------ forward
    def forward(
        self,
        x_v: torch.Tensor,
        x_a: torch.Tensor,
        x_t: torch.Tensor,
        mask_v: torch.Tensor | None = None,
        mask_a: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """x_v/x_a: (B,N,d_in) 序列；x_t: (B,d_in) 单向量（E18 已池化）。"""
        # 文本按长度为 1 的序列处理，使三模态走同一套代码
        z = {
            "v": self.proj["v"](x_v),
            "a": self.proj["a"](x_a),
            "l": self.proj["l"](x_t.unsqueeze(1)),
        }
        masks: dict[str, torch.Tensor | None] = {"v": mask_v, "a": mask_a, "l": None}
        losses: dict[str, torch.Tensor] = {}
        extra: dict[str, torch.Tensor] = {}

        # ---- MAM ----
        if self.use_mam:
            masked, l_mam = self._apply_mam(z, masks)
            for m in MODALITIES:
                z[m] = masked[m]
                extra[f"keep_rate_{m}"] = masked[f"keep_rate_{m}"]
            losses["L_MAM"] = l_mam
        else:
            losses["L_MAM"] = z["v"].new_zeros(())

        # ---- 全局表示（E24：序列用 mean pool）----
        g = {m: masked_mean(z[m], masks.get(m)) for m in MODALITIES}
        gn = {m: l2norm(g[m]) for m in MODALITIES}

        # ---- ITAC（E25–E27）----
        if self.use_itac:
            fnm = None
            if labels is not None and self.suppress_false_neg and labels.dtype in (
                torch.int32, torch.int64
            ):
                fnm = build_false_neg_mask(labels)
            l_vt = single_infonce(gn["v"], gn["l"], self.tau, fnm)
            l_at = single_infonce(gn["a"], gn["l"], self.tau, fnm)
            l_va = single_infonce(gn["v"], gn["a"], self.tau, fnm)
            losses["L_v_t"], losses["L_a_t"], losses["L_v_a"] = l_vt, l_at, l_va
            losses["L_ITAC"] = self.alpha * l_vt + self.beta * l_at + self.eta * l_va
        else:
            losses["L_ITAC"] = z["v"].new_zeros(())

        # ---- ITAM（E28–E31）----
        if self.use_itam and g["v"].size(0) > 1:
            l_itam, p_match = self._itam(g, labels)
            losses["L_ITAM"] = l_itam
            extra["p_match"] = p_match
        else:
            losses["L_ITAM"] = z["v"].new_zeros(())

        out = {"z_v": z["v"], "z_a": z["a"], "z_l": z["l"],
               "g_v": g["v"], "g_a": g["a"], "g_l": g["l"],
               "mask_v": mask_v, "mask_a": mask_a, **extra}
        return out, losses
