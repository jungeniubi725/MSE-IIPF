"""SG-DSFE：语义引导双流特征增强（论文 §3.2，公式 E2–E19）。

数据流
------
    H_v, H_a (原始流)  +  H_sv, H_sa (模态转化语义流)
      → 统一投影 d 维                                    E4
      → 双流语义对齐 L_DSA                                E5–E10
      → 跨注意力语义注入                                  E11–E12
      → 一致性门控融合                                    E13–E16
      → 统一结构化语义表示 X_t                            E17–E18
    输出 O_SG = (X_v*, X_a*, X_t*)                        E19

注意 X_t* 是**单向量**（E18 已池化），而 X_v*、X_a* 仍是序列。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..losses.contrastive import bidirectional_infonce
from .layers import (
    AttentionPool,
    ConsistencyGate,
    CrossAttentionInject,
    ProjectionHead,
    masked_mean,
    pool_cls,
)


class SGDSFE(nn.Module):
    def __init__(
        self,
        d_visual: int = 768,
        d_audio: int = 768,
        d_text: int = 768,
        d: int = 256,
        num_heads: int = 4,
        tau: float = 0.07,
        dropout: float = 0.1,
        use_semantic_stream: bool = True,  # 消融 A1：关掉则不用语义流
        use_gate: bool = True,  # 消融 A3：关掉则 g ≡ 1（语义直接注入）
    ) -> None:
        super().__init__()
        self.d = d
        self.tau = tau
        self.use_semantic_stream = use_semantic_stream
        self.use_gate = use_gate

        # E4：四路线性投影到统一 d 维
        self.W_v = nn.Linear(d_visual, d, bias=False)
        self.W_a = nn.Linear(d_audio, d, bias=False)
        self.W_sv = nn.Linear(d_text, d, bias=False)
        self.W_sa = nn.Linear(d_text, d, bias=False)

        # E5：投影头。P_s 由视觉/音频两个语义分支**共享**（原文两式同用 P_s）
        self.P_v = ProjectionHead(d, d)
        self.P_a = ProjectionHead(d, d)
        self.P_s = ProjectionHead(d, d)

        # E11–E12：跨注意力注入
        self.inject_v = CrossAttentionInject(d, num_heads, dropout)
        self.inject_a = CrossAttentionInject(d, num_heads, dropout)

        # E14–E15：一致性门控（通道级）
        self.gate_v = ConsistencyGate(d)
        self.gate_a = ConsistencyGate(d)

        # E17–E18：统一语义表示的注意力池化
        self.attn_pool = AttentionPool(d)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        h_v: torch.Tensor,
        h_a: torch.Tensor,
        h_sv: torch.Tensor,
        h_sa: torch.Tensor,
        mask_v: torch.Tensor | None = None,
        mask_a: torch.Tensor | None = None,
        mask_sv: torch.Tensor | None = None,
        mask_sa: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """输入均为 (B, N_*, d_in)。返回 (输出字典, 损失字典)。"""
        # ---- E4 统一维度 ----
        x_v = self.W_v(h_v)
        x_a = self.W_a(h_a)
        x_sv = self.W_sv(h_sv)
        x_sa = self.W_sa(h_sa)

        losses: dict[str, torch.Tensor] = {}

        if not self.use_semantic_stream:
            # 消融 A1：无语义流。X_t 退化为零向量，下游文本模态被置空。
            out = {
                "x_v": x_v,
                "x_a": x_a,
                "x_t": x_v.new_zeros(x_v.size(0), self.d),
                "mask_v": mask_v,
                "mask_a": mask_a,
            }
            losses["L_DSA"] = x_v.new_zeros(())
            return out, losses

        # ---- E5 池化 + 投影 + L2 归一化 ----
        z_v = self.P_v(masked_mean(x_v, mask_v))
        z_a = self.P_a(masked_mean(x_a, mask_a))
        z_sv = self.P_s(pool_cls(x_sv))
        z_sa = self.P_s(pool_cls(x_sa))

        # ---- E6–E10 双流语义对齐 ----
        l_v_sv = bidirectional_infonce(z_v, z_sv, self.tau)
        l_a_sa = bidirectional_infonce(z_a, z_sa, self.tau)
        losses["L_v_sv"] = l_v_sv
        losses["L_a_sa"] = l_a_sa
        losses["L_DSA"] = l_v_sv + l_a_sa  # E9

        # ---- E11–E12 跨注意力注入 ----
        x_v_tilde = self.inject_v(x_v, x_sv, mask_sv)
        x_a_tilde = self.inject_a(x_a, x_sa, mask_sa)

        # ---- E13–E16 一致性门控 ----
        if self.use_gate:
            r_v = masked_mean(x_v, mask_v)
            r_a = masked_mean(x_a, mask_a)
            r_sv = pool_cls(x_sv)
            r_sa = pool_cls(x_sa)
            g_v = self.gate_v(r_v, r_sv).unsqueeze(1)  # (B,1,d) 沿 token 广播
            g_a = self.gate_a(r_a, r_sa).unsqueeze(1)
            x_v_star = g_v * x_v_tilde + (1.0 - g_v) * x_v
            x_a_star = g_a * x_a_tilde + (1.0 - g_a) * x_a
        else:  # 消融 A3
            g_v = g_a = None
            x_v_star, x_a_star = x_v_tilde, x_a_tilde

        # ---- E17–E18 统一结构化语义表示 ----
        u = torch.cat([x_sv, x_sa], dim=1)
        u_mask = None
        if mask_sv is not None or mask_sa is not None:
            mv = mask_sv if mask_sv is not None else x_sv.new_ones(x_sv.shape[:2])
            ma = mask_sa if mask_sa is not None else x_sa.new_ones(x_sa.shape[:2])
            u_mask = torch.cat([mv, ma], dim=1)
        x_t_star, attn = self.attn_pool(u, u_mask)

        out = {
            "x_v": x_v_star,
            "x_a": x_a_star,
            "x_t": x_t_star,  # (B, d) 单向量
            "mask_v": mask_v,
            "mask_a": mask_a,
            "text_attn": attn,
            "z_v": z_v,
            "z_a": z_a,
            "z_sv": z_sv,
            "z_sa": z_sa,
        }
        if g_v is not None:
            out["gate_v"] = g_v.squeeze(1)
            out["gate_a"] = g_a.squeeze(1)
        return out, losses
