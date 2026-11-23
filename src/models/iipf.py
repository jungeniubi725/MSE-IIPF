"""IIPF：隐式内部状态代理学习与融合（论文 §3.4，公式 E33–E52 前段）。

四个子块
-------
  a) 跨模态一致性驱动的代理初始化   E34–E41  -> L_CMC = L_rep + λ_d·L_dist
  b) 情绪转移感知的时序演化建模     E42–E43  -> L_TPC
  c) 代理—外显表征重建约束          E44–E48  -> L_PRR = L_rec + λ_cyc·L_cycle
  d) 代理引导的外显表达融合         E49–E52  -> f_t

张量约定
-------
本模块按 (B, T, ·) 组织，T 为时间单元数（RAVDESS 的片段索引 / MOSI 的 utterance）。
下标 l 表示文本模态（与 §3.2 的 t 同指），避免与时间下标 t 混淆。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..losses.contrastive import cosine_align_loss, predictive_infonce
from ..losses.divergence import dist_consistency_loss
from .layers import AttentionPool, ConsistencyGate, ResidualMLP

MODALITIES = ("v", "a", "l")


class ProxyInit(nn.Module):
    """E34–E41：可靠性加权的共享情感证据 -> 初始代理 z_t^(0)。"""

    def __init__(self, d: int, d_p: int, num_classes: int, tau_r: float = 1.0,
                 lambda_d: float = 0.5) -> None:
        super().__init__()
        self.tau_r = tau_r
        self.lambda_d = lambda_d
        # E34: q_m,t = LN(W_m^p h_m,t + b_m^p)
        self.W_p = nn.ModuleDict({m: nn.Linear(d, d_p) for m in MODALITIES})
        self.ln_q = nn.ModuleDict({m: nn.LayerNorm(d_p) for m in MODALITIES})
        # E35: r_m,t = σ((w_m^r)ᵀ q_m,t + b_m^r)
        self.w_r = nn.ModuleDict({m: nn.Linear(d_p, 1) for m in MODALITIES})
        # E37 后: z_t^(0) = LN(W_c c_t + b_c)
        self.W_c = nn.Linear(d_p, d_p)
        self.ln_z = nn.LayerNorm(d_p)
        # E39 的共享辅助情感头（模态分支与代理分支共用）
        self.aux_head = nn.Linear(d_p, num_classes)

    def forward(
        self, h: dict[str, torch.Tensor], mask_t: torch.Tensor | None = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """h[m]: (B,T,d) 各模态时间单元级表示。mask_t: (B,T) 1 有效。"""
        q = {m: self.ln_q[m](self.W_p[m](h[m])) for m in MODALITIES}  # E34
        r = torch.cat([torch.sigmoid(self.w_r[m](q[m])) for m in MODALITIES], dim=-1)
        omega = (r / self.tau_r).softmax(dim=-1)  # E36  (B,T,3)
        q_stack = torch.stack([q[m] for m in MODALITIES], dim=2)  # (B,T,3,d_p)
        c = (omega.unsqueeze(-1) * q_stack).sum(dim=2)  # E37  (B,T,d_p)
        z0 = self.ln_z(self.W_c(c))  # 初始代理

        # E38：表示一致性（ω 加权，代理侧 stop-gradient 防坍塌）
        w = omega
        if mask_t is not None:
            w = w * mask_t.unsqueeze(-1).to(w.dtype)
        l_rep = cosine_align_loss(
            q_stack, z0.detach().unsqueeze(2).expand_as(q_stack), weight=w
        )
        # E39–E40：分布一致性
        p_modal = self.aux_head(q_stack).softmax(dim=-1)  # (B,T,3,K)
        p_proxy = self.aux_head(z0).softmax(dim=-1)  # (B,T,K)
        l_dist = dist_consistency_loss(p_modal, p_proxy.detach(), weight=w)

        losses = {
            "L_rep": l_rep,
            "L_dist": l_dist,
            "L_CMC": l_rep + self.lambda_d * l_dist,  # E41
        }
        return {"q": q, "q_stack": q_stack, "omega": omega, "c": c, "z0": z0,
                "p_modal": p_modal, "p_proxy": p_proxy}, losses


class TemporalEvolution(nn.Module):
    """E42–E43：GRU 时序化 + 转移强度门控 + 对比预测。"""

    def __init__(self, d_p: int, tau_z: float = 0.07) -> None:
        super().__init__()
        self.tau_z = tau_z
        self.gru = nn.GRU(d_p, d_p, batch_first=True)
        # ρ_t = σ(w_ρᵀ[Δc_t; ‖Δc_t‖] + b_ρ)
        self.w_rho = nn.Linear(d_p + 1, 1)
        # 转移预测器：两层 MLP + ReLU（原文 L125）
        self.delta = ResidualMLP(2 * d_p, d_p)

    def forward(
        self, z0: torch.Tensor, c: torch.Tensor, mask_t: torch.Tensor | None = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        z, _ = self.gru(z0)  # E42: z_t = GRU(z_t^(0), z_{t−1})
        T = z.size(1)
        if T < 2:  # 单时间单元，无转移可学
            return {"z": z, "rho": None}, {"L_TPC": z.new_zeros(())}

        dc = c[:, 1:] - c[:, :-1]  # Δc_t
        rho = torch.sigmoid(
            self.w_rho(torch.cat([dc, dc.norm(dim=-1, keepdim=True)], dim=-1))
        )  # (B,T-1,1)
        dz = self.delta(torch.cat([z[:, :-1], dc], dim=-1))
        z_pred = z[:, :-1] + rho * dz  # ρ 小则保持，ρ 大则允许跃变

        # E43：对比预测，负样本为 batch 内其他样本的下一状态
        target = z[:, 1:].detach()
        losses_t = []
        for t in range(T - 1):
            if mask_t is not None and mask_t[:, t + 1].sum() < 2:
                continue  # 有效样本不足 2 无法构负样本
            losses_t.append(predictive_infonce(z_pred[:, t], target[:, t], self.tau_z))
        l_tpc = (
            torch.stack(losses_t).mean() if losses_t else z.new_zeros(())
        )
        return {"z": z, "rho": rho, "z_pred": z_pred}, {"L_TPC": l_tpc}


class ProxyReconstruction(nn.Module):
    """E44–E48：对比式重建 + 循环一致性。"""

    def __init__(self, d_p: int, tau_q: float = 0.07,
                 mu: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
                 lambda_cyc: float = 0.5) -> None:
        super().__init__()
        assert abs(sum(mu) - 1.0) < 1e-6, "E46 要求 μ_v + μ_a + μ_l = 1"
        self.tau_q = tau_q
        self.mu = dict(zip(MODALITIES, mu))
        self.lambda_cyc = lambda_cyc
        self.R = nn.ModuleDict({m: ResidualMLP(d_p, d_p) for m in MODALITIES})
        self.R_z = ResidualMLP(3 * d_p, d_p)  # 循环映射网络

    def forward(
        self, z: torch.Tensor, q: dict[str, torch.Tensor],
        mask_t: torch.Tensor | None = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        B, T, _ = z.shape
        q_hat = {m: self.R[m](z) for m in MODALITIES}  # E44
        losses: dict[str, torch.Tensor] = {}

        # E45–E46：逐模态对比式重建，batch 内其他样本为负
        l_rec = z.new_zeros(())
        for m in MODALITIES:
            per_t = []
            for t in range(T):
                if B < 2:
                    continue
                per_t.append(
                    predictive_infonce(q_hat[m][:, t], q[m][:, t].detach(), self.tau_q)
                )
            lm = torch.stack(per_t).mean() if per_t else z.new_zeros(())
            losses[f"L_rec_{m}"] = lm
            l_rec = l_rec + self.mu[m] * lm
        losses["L_rec"] = l_rec

        # E47–E48：循环映射回代理空间 + 一致性
        z_cyc = self.R_z(torch.cat([q_hat[m] for m in MODALITIES], dim=-1))
        w = mask_t.to(z.dtype) if mask_t is not None else None
        l_cycle = cosine_align_loss(z_cyc, z.detach(), weight=w)
        losses["L_cycle"] = l_cycle
        losses["L_PRR"] = l_rec + self.lambda_cyc * l_cycle
        return {"q_hat": q_hat, "z_cyc": z_cyc}, losses


class ProxyGuidedFusion(nn.Module):
    """E49–E52：稳定先验 + 代理引导的模态权重 + 注入门控。"""

    def __init__(self, d: int, d_p: int, d_f: int = 256,
                 agg: str = "gru_stable", use_proxy_weight: bool = True) -> None:
        super().__init__()
        self.agg = agg
        self.use_proxy_weight = use_proxy_weight  # 消融 A11
        # 稳定性门控（原文 L140）：差异小则累积，转移明显则快更新
        self.stable_gate = ConsistencyGate(d_p, d_p)
        self.attn_agg = AttentionPool(d_p)  # 独立视频样本的片段级注意力聚合
        # E49：三模态与代理映射到统一融合空间
        self.W_f = nn.ModuleDict({m: nn.Linear(d, d_f) for m in MODALITIES})
        self.W_zf = nn.Linear(d_p, d_f)
        # E50：a_m,t = w_gᵀ tanh(W_e e_m,t + W_z e_z,t + b_g)
        self.W_e = nn.Linear(d_f, d_f)
        self.W_z = nn.Linear(d_f, d_f, bias=False)
        self.w_g = nn.Linear(d_f, 1, bias=False)
        # E51–E52：代理注入门控 + 输出 LN
        self.inject_gate = ConsistencyGate(d_f, d_f)
        self.ln_out = nn.LayerNorm(d_f)

    def _stabilize(self, z: torch.Tensor, mask_t: torch.Tensor | None) -> torch.Tensor:
        """得到稳定内部状态先验 z̃_t。"""
        if self.agg == "attn":
            pooled, _ = self.attn_agg(z, mask_t)
            return pooled.unsqueeze(1).expand_as(z)
        # gru_stable：沿时间递推，门控决定累积还是快更新
        outs, prev = [], z[:, 0]
        outs.append(prev)
        for t in range(1, z.size(1)):
            g = self.stable_gate(prev, z[:, t])
            prev = g * z[:, t] + (1.0 - g) * prev
            outs.append(prev)
        return torch.stack(outs, dim=1)

    def forward(
        self, h: dict[str, torch.Tensor], z: torch.Tensor,
        mask_t: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        z_tilde = self._stabilize(z, mask_t)
        e = {m: self.W_f[m](h[m]) for m in MODALITIES}  # E49
        e_z = self.W_zf(z_tilde)

        e_stack = torch.stack([e[m] for m in MODALITIES], dim=2)  # (B,T,3,d_f)
        if self.use_proxy_weight:
            a = self.w_g(torch.tanh(self.W_e(e_stack) + self.W_z(e_z).unsqueeze(2)))
        else:  # 消融 A11：权重不看代理
            a = self.w_g(torch.tanh(self.W_e(e_stack)))
        alpha = a.softmax(dim=2)  # E50  (B,T,3,1)
        e_ext = (alpha * e_stack).sum(dim=2)  # (B,T,d_f)

        g_t = self.inject_gate(e_ext, e_z)  # E51
        f_t = self.ln_out(e_ext + g_t * e_z)  # E52
        return {"f": f_t, "e_ext": e_ext, "e_z": e_z, "z_tilde": z_tilde,
                "alpha": alpha.squeeze(-1), "gate_inject": g_t}


class IIPF(nn.Module):
    """IIPF 总装。输入三模态时间单元级表示，输出融合表示 f_t 与三项损失。"""

    def __init__(
        self,
        d: int = 256,
        d_p: int = 128,
        d_f: int = 256,
        num_classes: int = 8,
        tau_r: float = 1.0,
        tau_z: float = 0.07,
        tau_q: float = 0.07,
        lambda_d: float = 0.5,
        lambda_cyc: float = 0.5,
        mu: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
        agg: str = "gru_stable",
        use_cmc: bool = True,  # 消融 A8
        use_tpc: bool = True,  # 消融 A9
        use_prr: bool = True,  # 消融 A10
        use_proxy_weight: bool = True,  # 消融 A11
    ) -> None:
        super().__init__()
        self.use_cmc, self.use_tpc, self.use_prr = use_cmc, use_tpc, use_prr
        self.init = ProxyInit(d, d_p, num_classes, tau_r, lambda_d)
        self.temporal = TemporalEvolution(d_p, tau_z)
        self.recon = ProxyReconstruction(d_p, tau_q, mu, lambda_cyc)
        self.fusion = ProxyGuidedFusion(d, d_p, d_f, agg, use_proxy_weight)

    def forward(
        self,
        h_v: torch.Tensor,
        h_a: torch.Tensor,
        h_l: torch.Tensor,
        mask_t: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """h_*: (B, T, d)。返回 (输出字典, 损失字典)。"""
        h = {"v": h_v, "a": h_a, "l": h_l}
        out: dict[str, torch.Tensor] = {}
        losses: dict[str, torch.Tensor] = {}

        o_init, l_init = self.init(h, mask_t)
        out.update({k: o_init[k] for k in ("omega", "z0", "p_modal", "p_proxy")})
        losses["L_CMC"] = l_init["L_CMC"] if self.use_cmc else h_v.new_zeros(())
        losses.update({k: v for k, v in l_init.items() if k != "L_CMC"})

        o_tmp, l_tmp = self.temporal(o_init["z0"], o_init["c"], mask_t)
        z = o_tmp["z"]
        out["z"] = z
        losses["L_TPC"] = l_tmp["L_TPC"] if self.use_tpc else h_v.new_zeros(())

        o_rec, l_rec = self.recon(z, o_init["q"], mask_t)
        losses["L_PRR"] = l_rec["L_PRR"] if self.use_prr else h_v.new_zeros(())
        losses.update({k: v for k, v in l_rec.items() if k != "L_PRR"})

        o_fus = self.fusion(h, z, mask_t)
        out.update(o_fus)
        return out, losses
