"""MSE-IIPF：总装。串起四大模块，输出预测与总损失。"""

from __future__ import annotations

import torch
import torch.nn as nn

from .euc_head import EvidentialHead
from .iipf import IIPF
from .sg_dsfe import SGDSFE
from .tma import TriModalAlignment


class MSEIIPF(nn.Module):
    """论文 E1 的完整数据流：

    (V, A) → SG-DSFE → (X_v*, X_a*, X_t*) → TMA → (Z_v, Z_a, Z_l)
         → IIPF → f_t → Evidential Head → (p̂_t, ŷ_t, u_t, r_t)
    """

    def __init__(
        self,
        d_visual: int = 768,
        d_audio: int = 768,
        d_text: int = 768,
        d: int = 256,
        d_p: int = 128,
        d_f: int = 256,
        num_classes: int = 8,
        # 损失权重（E1 总损失的系数）
        w_dsa: float = 1.0,
        w_tma: tuple[float, float, float] = (0.1, 1.0, 0.5),  # λ1, λ2, λ3
        w_iipf: tuple[float, float, float] = (1.0, 1.0, 1.0),  # λ_cmc, λ_tpc, λ_prr
        # 各模块超参
        tau_dsa: float = 0.07,
        tau_itac: float = 0.07,
        gamma: float = 0.5,
        itac_weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
        tau_r: float = 1.0,
        tau_z: float = 0.07,
        tau_q: float = 0.07,
        lambda_d: float = 0.5,
        lambda_cyc: float = 0.5,
        mu: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
        agg: str = "gru_stable",
        anneal_epochs: int = 10,
        lambda_cal: float = 0.5,  # L_cal 正确性硬目标权重
        lambda_ce: float = 4.0,  # CE 辅助项权重（校准修复）
        lambda_uni: float = 0.5,  # 三模态辅助头监督（供 T6 的 Krippendorff α）
        lambda_seq: float = 0.0,  # sequence 级 EDL/CE 权重（RAVDESS）
        # 消融开关
        use_semantic_stream: bool = True,  # A1
        use_gate: bool = True,  # A3
        use_mam: bool = True,  # A4
        use_itac: bool = True,  # A5
        use_itam: bool = True,  # A6
        use_iipf_module: bool = True,  # A7
        use_cmc: bool = True,  # A8
        use_tpc: bool = True,  # A9
        use_prr: bool = True,  # A10
        use_proxy_weight: bool = True,  # A11
        use_evidential: bool = True,  # A12（关掉则退化为 Softmax 头）
        use_cal: bool = True,  # A13
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.w_dsa = w_dsa
        self.w_tma_1, self.w_tma_2, self.w_tma_3 = w_tma
        self.w_iipf_cmc, self.w_iipf_tpc, self.w_iipf_prr = w_iipf
        self.use_iipf_module = use_iipf_module
        self.use_evidential = use_evidential

        self.sg_dsfe = SGDSFE(d_visual, d_audio, d_text, d, num_heads, tau_dsa,
                              dropout, use_semantic_stream, use_gate)
        self.tma = TriModalAlignment(d, d, num_heads, tau_itac, gamma, 0.8,
                                    itac_weights, num_classes, use_mam, use_itac,
                                    use_itam, suppress_false_neg=True, dropout=dropout)
        self.iipf = IIPF(d, d_p, d_f, num_classes, tau_r, tau_z, tau_q, lambda_d,
                        lambda_cyc, mu, agg, use_cmc, use_tpc, use_prr, use_proxy_weight)
        self.head = EvidentialHead(
            d_f=d_f,
            d_modal=d,
            num_classes=num_classes,
            anneal_epochs=anneal_epochs,
            lambda_cal=lambda_cal if use_cal else 0.0,
            lambda_ce=lambda_ce,
            lambda_uni=lambda_uni,
            lambda_seq=lambda_seq,
        )
        # A12：Softmax 退化头
        self.softmax_head = nn.Linear(d_f, num_classes)
        # A7（关掉 IIPF）时融合输出维度是 d，证据头期望 d_f，需投影对齐
        self.a7_proj = nn.Identity() if d == d_f else nn.Linear(d, d_f)

    def forward(
        self,
        h_v: torch.Tensor,
        h_a: torch.Tensor,
        h_sv: torch.Tensor,
        h_sa: torch.Tensor,
        y: torch.Tensor,
        epoch: int = 0,
        mask_v: torch.Tensor | None = None,
        mask_a: torch.Tensor | None = None,
        mask_sv: torch.Tensor | None = None,
        mask_sa: torch.Tensor | None = None,
        mask_t: torch.Tensor | None = None,
        labels_for_itam: torch.Tensor | None = None,
        seq_labels: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """输入四路特征 (B, N_*, d_in)；y: (B, T) 标签。返回 (输出, 损失)。

        labels_for_itam: (B,) sequence 级标签（RAVDESS 用），TMA 的伪负样本抑制。
        """
        # ---- SG-DSFE ----
        o1, l1 = self.sg_dsfe(h_v, h_a, h_sv, h_sa, mask_v, mask_a, mask_sv, mask_sa)
        # ---- TMA ----
        o2, l2 = self.tma(o1["x_v"], o1["x_a"], o1["x_t"], o1.get("mask_v"),
                         o1.get("mask_a"), labels_for_itam)
        # ---- reshape：SG-DSFE/TMA 在 B_eff = B×T 上跑，IIPF 起按 (B,T,·) ----
        # collate_fn 把每个时间单元展平进 batch 维（见 data/dataset.py 的说明），
        # 故这里要把 (B*T, d) 折回 (B, T, d)，而不是取 token 维前 T 个。
        B, T = y.shape
        d_model = o2["g_v"].size(-1)
        assert o2["g_v"].size(0) == B * T, (
            f"batch 维不匹配：g_v 有 {o2['g_v'].size(0)} 行，期望 B*T={B*T}。"
            "检查 collate_fn 是否按 B×T 展平。"
        )
        g_v_t = o2["g_v"].view(B, T, d_model)
        g_a_t = o2["g_a"].view(B, T, d_model)
        g_l_t = o2["g_l"].view(B, T, d_model)

        # ---- IIPF ----
        if not self.use_iipf_module:
            # 消融 A7：IIPF 整块关掉，f_t 退化为三模态注意力融合
            g_stack = torch.stack([g_v_t, g_a_t, g_l_t], dim=2)  # (B,T,3,d)
            attn = g_stack.mean(dim=-1, keepdim=True).softmax(dim=2)
            o3 = {"f": self.a7_proj((attn * g_stack).sum(dim=2))}  # (B,T,d_f)
            l3 = {"L_CMC": h_v.new_zeros(()), "L_TPC": h_v.new_zeros(()),
                  "L_PRR": h_v.new_zeros(())}
        else:
            o3, l3 = self.iipf(g_v_t, g_a_t, g_l_t, mask_t)
        # ---- 证据头 ----
        f_t = o3["f"]
        if self.use_evidential:
            o4, l4 = self.head(
                f_t, g_v_t, g_a_t, g_l_t, y, epoch, mask_t, seq_labels
            )
        else:  # 消融 A12
            logits = self.softmax_head(f_t)
            p_hat = logits.softmax(-1)
            y_pred = p_hat.argmax(-1)
            w = mask_t.to(p_hat.dtype).unsqueeze(-1) if mask_t is not None \
                else torch.ones_like(p_hat[..., :1])
            p_seq = (p_hat * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)
            o4 = {"p_hat": p_hat, "y_pred": y_pred, "u": None, "r": None,
                  "alpha": None, "p_seq": p_seq,
                  "y_seq_pred": p_seq.argmax(dim=-1), "u_seq": None}
            ce = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
            ).view_as(y)
            l4 = {"L_head": (ce * w.squeeze(-1)).sum() / w.sum().clamp_min(1e-6)}
        # ---- 总损失 ----
        loss_total = (
            l4["L_head"]
            + self.w_dsa * l1["L_DSA"]
            + self.w_tma_1 * l2["L_MAM"]
            + self.w_tma_2 * l2["L_ITAC"]
            + self.w_tma_3 * l2["L_ITAM"]
            + self.w_iipf_cmc * l3["L_CMC"]
            + self.w_iipf_tpc * l3["L_TPC"]
            + self.w_iipf_prr * l3["L_PRR"]
        )
        out = {**o1, **o2, **o3, **o4}
        losses = {"loss": loss_total, **l1, **l2, **l3, **l4}
        return out, losses
