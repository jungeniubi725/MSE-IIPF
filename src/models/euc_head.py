"""证据式不确定性感知情感分类头（论文 §3.5，公式 E42–E52）。

两个输出
-------
  forward         : 融合表示 f_t -> (p̂_t, ŷ_t, u_t, r_t, α_t) + L_head
  forward_modality: 三模态辅助分支 -> (p_v, p_a, p_l)，用于 Krippendorff α 分析

校准损失 L_cal
--------------
L_cal 把证据不确定性 u_t 回归到「预测是否正确」的硬标签 1[ŷ_t ≠ y_t]：
预测正确时 u 应低，错误时 u 应高。这与 experiment.txt 的「Lcauc 硬正确性
版本」一致，是本方法选定的主模型形式（早先曾用三模态冲突度 κ 作软目标，
RAVDESS 上 κ 与预测正确性负相关、无 ECE 收益，已弃用）。

三个单模态辅助头（modal_head_v/a/l）不再参与 L_cal，仅由有标签锚点 L_uni
用真实情感标签监督，使其预测具有类别语义，供 §5.1 的 Krippendorff α 模态
一致性分析（T6 表）使用。forward_modality 在推理时单独调用返回这三个分布。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses.divergence import kl_dirichlet_uniform
from ..utils.evidential import aggregate_dirichlet_evidence


class EvidentialHead(nn.Module):
    def __init__(
        self,
        d_f: int = 256,
        d_modal: int = 256,
        num_classes: int = 8,
        anneal_epochs: int = 10,
        lambda_cal: float = 0.5,
        lambda_ce: float = 4.0,  # CE 辅助项权重（校准修复）
        lambda_uni: float = 0.5,
        lambda_seq: float = 0.0,
    ) -> None:
        super().__init__()
        self.K = num_classes
        self.anneal_epochs = anneal_epochs
        self.lambda_cal = lambda_cal
        self.lambda_ce = lambda_ce
        self.lambda_uni = lambda_uni
        self.lambda_seq = lambda_seq
        # E42：融合表示 -> 证据（Softplus 保证非负）
        self.evidence_net = nn.Sequential(nn.Linear(d_f, d_f), nn.ReLU(),
                                          nn.Linear(d_f, num_classes))
        # 三模态辅助头（Softmax 输出），供 Krippendorff α（T6）与 L_uni
        self.modal_head_v = nn.Linear(d_modal, num_classes)
        self.modal_head_a = nn.Linear(d_modal, num_classes)
        self.modal_head_l = nn.Linear(d_modal, num_classes)

    # ------------------------------------------------------------------ 证据与不确定性
    def _forward_evidential(
        self, f: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """f: (B,T,d_f) -> (α, p̂, u, ŷ)。α:(B,T,K), 其余:(B,T)。"""
        e = F.softplus(self.evidence_net(f)).clamp(max=1e4)  # E42：防溢出
        alpha = e + 1.0  # (B,T,K)
        S = alpha.sum(dim=-1, keepdim=True)  # (B,T,1)
        p_hat = alpha / S  # E42 期望概率
        u = self.K / S.squeeze(-1)  # E42 不确定性
        y_pred = p_hat.argmax(dim=-1)
        return alpha, p_hat, u, y_pred

    # ------------------------------------------------------------------ 损失
    def _edl_loss(
        self, alpha: torch.Tensor, y: torch.Tensor, mask_t: torch.Tensor | None
    ) -> torch.Tensor:
        """E45：Dirichlet 期望交叉熵。α:(B,T,K), y:(B,T) long。"""
        S = alpha.sum(dim=-1)
        y_oh = F.one_hot(y, self.K).to(alpha.dtype)  # (B,T,K)
        term = y_oh * (torch.digamma(S.unsqueeze(-1)) - torch.digamma(alpha))
        if mask_t is not None:
            term = term * mask_t.unsqueeze(-1).to(term.dtype)
            n = mask_t.sum().clamp_min(1e-6)
        else:
            n = float(y.numel())
        return term.sum() / n

    def _kl_loss(
        self, alpha: torch.Tensor, y: torch.Tensor, mask_t: torch.Tensor | None
    ) -> torch.Tensor:
        """E46–E47：KL 正则，修正 Dirichlet 参数后算与均匀先验的 KL 散度。"""
        y_oh = F.one_hot(y, self.K).to(alpha.dtype)
        alpha_tilde = y_oh + (1.0 - y_oh) * alpha  # E46
        kl = kl_dirichlet_uniform(alpha_tilde)  # E47
        if mask_t is not None:
            kl = kl * mask_t.to(kl.dtype)
            return kl.sum() / mask_t.sum().clamp_min(1e-6)
        return kl.mean()

    def _ce_loss(
        self, p_hat: torch.Tensor, y: torch.Tensor, mask_t: torch.Tensor | None
    ) -> torch.Tensor:
        """CE 辅助项（校准修复）：-log p̂_y。

        L_EDL 的 digamma 形式在 evidence 稍大后梯度趋近 0（trigamma(S)−trigamma(α_y)→0），
        正确类 evidence 涨不动，导致 p̂ 接近均匀、u=K/Σα 虚高、confidence 虚低、ECE 差。
        CE 对正确类 evidence 的梯度不饱和，能持续驱动 evidence 增长；其 log S 项还带
        温和正则作用，抑制过拟合（logits 版 CE 在短训下易过拟合到 val，test 反而更差）。
        作为 L_EDL 的收敛加速器。
        """
        log_p = torch.log(p_hat.clamp_min(1e-8))
        y_oh = F.one_hot(y, self.K).to(p_hat.dtype)
        ce = -(y_oh * log_p).sum(dim=-1)
        if mask_t is not None:
            ce = ce * mask_t.to(ce.dtype)
            return ce.sum() / mask_t.sum().clamp_min(1e-6)
        return ce.mean()

    def _calibration_loss(
        self,
        u: torch.Tensor,
        y: torch.Tensor,
        y_pred: torch.Tensor,
        mask_t: torch.Tensor | None,
    ) -> torch.Tensor:
        """L_cal：正确性硬目标校准，u_t 回归到 1[ŷ_t ≠ y_t]。

        目标用 y_pred.detach() 构造，使「是否正确」作为固定教师信号，
        校准项只更新证据不确定性分支，不通过目标反向影响分类边界。
        """
        target = (y_pred.detach() != y).to(u.dtype)  # 1 错误, 0 正确
        reg = (u - target) ** 2
        if mask_t is not None:
            return (reg * mask_t.to(reg.dtype)).sum() / mask_t.sum().clamp_min(1e-6)
        return reg.mean()

    def _unimodal_loss(
        self,
        p_v: torch.Tensor,
        p_a: torch.Tensor,
        p_l: torch.Tensor,
        y: torch.Tensor,
        mask_t: torch.Tensor | None,
    ) -> torch.Tensor:
        """三个辅助头的有监督锚点：L_uni = ⅓ Σ_m CE(p_m, y)，供 T6 分析。"""
        losses = []
        for p_m in (p_v, p_a, p_l):
            nll = -p_m.clamp_min(1e-8).log().gather(-1, y.unsqueeze(-1)).squeeze(-1)
            if mask_t is not None:
                w = mask_t.to(nll.dtype)
                losses.append((nll * w).sum() / w.sum().clamp_min(1e-6))
            else:
                losses.append(nll.mean())
        return torch.stack(losses).mean()

    def _sequence_loss(
        self,
        alpha: torch.Tensor,
        seq_y: torch.Tensor | None,
        epoch: int,
        mask_t: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Sequence EDL/CE on evidence aggregated with one unit prior.

        ``alpha`` is (B,T,K), while ``seq_y`` is (B,).  This objective aligns
        training and checkpoint selection with RAVDESS's sample granularity.
        """
        alpha_seq = aggregate_dirichlet_evidence(alpha, dim=1, mask=mask_t)
        strength = alpha_seq.sum(dim=-1, keepdim=True)
        p_seq = alpha_seq / strength
        u_seq = self.K / strength.squeeze(-1)
        y_seq_pred = p_seq.argmax(dim=-1)
        out = {
            "alpha_seq": alpha_seq,
            "p_seq": p_seq,
            "u_seq": u_seq,
            "y_seq_pred": y_seq_pred,
        }

        zero = alpha.new_zeros(())
        if seq_y is None:
            return out, {
                "L_seq_EDL": zero, "L_seq_KL": zero,
                "L_seq_CE": zero, "L_seq": zero,
            }
        seq_y = seq_y.to(device=alpha.device, dtype=torch.long).reshape(-1)
        if seq_y.numel() != alpha_seq.size(0):
            raise ValueError(
                f"seq_y has {seq_y.numel()} labels, expected {alpha_seq.size(0)}"
            )
        y = seq_y.unsqueeze(1)
        a = alpha_seq.unsqueeze(1)
        p = p_seq.unsqueeze(1)
        l_edl = self._edl_loss(a, y, None)
        l_kl = self._kl_loss(a, y, None)
        l_ce = self._ce_loss(p, y, None)
        lambda_kl = min(1.0, epoch / max(self.anneal_epochs, 1))
        l_seq = l_edl + lambda_kl * l_kl + self.lambda_ce * l_ce
        return out, {
            "L_seq_EDL": l_edl,
            "L_seq_KL": l_kl,
            "L_seq_CE": l_ce,
            "L_seq": l_seq,
        }

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        f: torch.Tensor,
        g_v: torch.Tensor,
        g_a: torch.Tensor,
        g_l: torch.Tensor,
        y: torch.Tensor,
        epoch: int = 0,
        mask_t: torch.Tensor | None = None,
        seq_y: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """f: (B,T,d_f) 融合；g_*: (B,T,d_modal) 三模态全局；y: (B,T) 标签。

        返回 (输出字典, 损失字典)。
        """
        alpha, p_hat, u, y_pred = self._forward_evidential(f)
        r = 1.0 - u  # 可靠性

        # 三模态辅助分支（Softmax），供 Krippendorff α（T6）与 L_uni
        p_v = self.modal_head_v(g_v).softmax(-1)
        p_a = self.modal_head_a(g_a).softmax(-1)
        p_l = self.modal_head_l(g_l).softmax(-1)

        # E45–E48
        l_edl = self._edl_loss(alpha, y, mask_t)
        l_kl = self._kl_loss(alpha, y, mask_t)
        lambda_kl = min(1.0, epoch / max(self.anneal_epochs, 1))  # E48 退火
        l_euc = l_edl + lambda_kl * l_kl  # E48 的 L_EUC（原文未分配符号，本文记作此）

        # 正确性硬目标校准（L_cal）
        l_cal = self._calibration_loss(u, y, y_pred, mask_t)
        # 三个单模态头由 L_uni 监督，保持类别语义，供 T6 的 Krippendorff α 分析。
        l_uni = self._unimodal_loss(p_v, p_a, p_l, y, mask_t)
        # 校准修复：CE 辅助项，加速正确类 evidence 增长
        l_ce = self._ce_loss(p_hat, y, mask_t)
        seq_out, seq_losses = self._sequence_loss(alpha, seq_y, epoch, mask_t)
        l_unit = (
            l_euc
            + self.lambda_cal * l_cal
            + self.lambda_uni * l_uni
            + self.lambda_ce * l_ce
        )
        l_head = l_unit + self.lambda_seq * seq_losses["L_seq"]

        losses = {
            "L_EDL": l_edl,
            "L_KL": l_kl,
            "L_EUC": l_euc,
            "L_cal": l_cal,
            "L_uni": l_uni,
            "L_CE": l_ce,
            "L_unit": l_unit,
            **seq_losses,
            "L_head": l_head,
        }
        out = {
            "alpha": alpha,  # (B,T,K)
            "p_hat": p_hat,
            "u": u,
            "r": r,
            "y_pred": y_pred,
            "p_v": p_v,
            "p_a": p_a,
            "p_l": p_l,
            "lambda_kl": lambda_kl,
            **seq_out,
        }
        return out, losses

    def forward_modality(
        self, g_v: torch.Tensor, g_a: torch.Tensor, g_l: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """推理时单独调，返回三模态 Softmax 概率（用于 Krippendorff α）。"""
        return (
            self.modal_head_v(g_v).softmax(-1),
            self.modal_head_a(g_a).softmax(-1),
            self.modal_head_l(g_l).softmax(-1),
        )
