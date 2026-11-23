# models/model_tri_albef_align.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用 ALBEF 官方的 all_gather 实现（分布式下用）
import torch.distributed as dist

@torch.no_grad()
def concat_all_gather(tensor):
    """
    在 DDP 下把各卡 tensor gather 后拼接；
    单卡 / 未初始化分布式时，直接原样返回。
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


@dataclass
class TriCfg:
    # ITC/Queue/Momentum
    embed_dim: int = 256
    temp_init: float = 0.07
    queue_size: int = 8192
    momentum: float = 0.995
    distill: bool = True  # 是否使用 momentum teacher soft targets

    # Transformer encoders
    d_model: int = 256
    depth: int = 4
    nhead: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    max_tokens: int = 256

    # 每个模态如果输入是 [B, D] 向量，则会切分成 num_tokens_*
    num_tokens_a: int = 16
    num_tokens_b: int = 16
    num_tokens_c: int = 16

    input_type_a: str = "vector"
    input_type_b: str = "sequence"
    input_type_c: str = "sequence"

    label_dim: int = 1


import math

class FeatureTokenizer(nn.Module):
    """
    支持两种输入：
    - vector 模态: x [B, D]，按 num_tokens 切块
    - sequence 模态: x [B, T, D0]，直接线性映射
    """
    def __init__(self, input_dim: int, num_tokens: int, d_model: int, input_type: str):
        super().__init__()
        assert input_type in ["vector", "sequence"]
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.input_type = input_type

        if self.input_type == "vector":
            # 例如 1585 / 16 -> 100
            token_dim = math.ceil(input_dim / num_tokens)
            self.token_dim = token_dim
            self.proj = nn.Linear(token_dim, d_model)
        else:
            # sequence 输入的每个 token 维度固定就是 input_dim
            self.token_dim = input_dim
            self.proj = nn.Linear(input_dim, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        if self.input_type == "vector":
            if x.dim() != 2:
                raise ValueError(f"Expected vector input [B,D], got {tuple(x.shape)}")
            B, D = x.shape
            T = self.num_tokens
            pad = T * self.token_dim - D
            if pad > 0:
                x = F.pad(x, (0, pad), value=0.0)
            x = x.view(B, T, self.token_dim)
            return self.proj(x), T

        else:
            if x.dim() != 3:
                raise ValueError(f"Expected sequence input [B,T,D], got {tuple(x.shape)}")
            B, T, D0 = x.shape
            if D0 != self.input_dim:
                raise ValueError(f"Sequence feature dim mismatch: expected {self.input_dim}, got {D0}")
            return self.proj(x), T


class TransformerEncoder(nn.Module):
    """
    输出 token embeddings:
      - embeds: [B, 1+T, d_model]
      - atts:   [B, 1+T]
    """
    def __init__(self, in_dim: int, num_tokens: int, cfg: TriCfg, input_type: str):
        super().__init__()
        self.cfg = cfg
        self.input_type = input_type

        self.tokenizer = FeatureTokenizer(
            input_dim=in_dim,
            num_tokens=num_tokens,
            d_model=cfg.d_model,
            input_type=input_type
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.max_tokens + 1, cfg.d_model))
        self.pos_drop = nn.Dropout(cfg.dropout)

        dim_ffn = int(cfg.d_model * cfg.mlp_ratio)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=dim_ffn,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.depth)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        tokens, T = self.tokenizer(x)

        if T > self.cfg.max_tokens:
            tokens = tokens[:, :self.cfg.max_tokens, :]
            T = self.cfg.max_tokens
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask[:, :T]

        B = tokens.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x_all = torch.cat([cls, tokens], dim=1)

        pos = self.pos_emb[:, : (1 + T), :]
        x_all = self.pos_drop(x_all + pos)

        if key_padding_mask is not None:
            cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=x_all.device)
            mask_all = torch.cat([cls_mask, key_padding_mask], dim=1)
        else:
            mask_all = None

        x_all = self.encoder(x_all, src_key_padding_mask=mask_all)

        if mask_all is None:
            atts = torch.ones((B, 1 + T), dtype=torch.long, device=x_all.device)
        else:
            atts = (~mask_all).long()

        return x_all, atts


class TriALBEFAlign(nn.Module):
    """
    三模态版 ALBEF-style：
    - 每个模态一个 TransformerEncoder（输出 token embeds）
    - ITC：三对 (A,B)(A,C)(B,C) 都做对比学习 + queue + (可选) teacher distill
    - ITM：三对都做 hard negative mining + fusion（这里用简单的“跨模态融合token”做二分类）
    - momentum teacher：每个 encoder + proj 各自一份动量模型
    """

    def __init__(self, dim_a: int, dim_b: int, dim_c: int, cfg: TriCfg):
        super().__init__()
        self.cfg = cfg
        self.distill = cfg.distill

        # ------------------------ student encoders ------------------------
        self.enc_a = TransformerEncoder(
            in_dim=dim_a,
            num_tokens=cfg.num_tokens_a,
            cfg=cfg,
            input_type=cfg.input_type_a,
        )
        self.enc_b = TransformerEncoder(
            in_dim=dim_b,
            num_tokens=cfg.num_tokens_b,
            cfg=cfg,
            input_type=cfg.input_type_b,
        )
        self.enc_c = TransformerEncoder(
            in_dim=dim_c,
            num_tokens=cfg.num_tokens_c,
            cfg=cfg,
            input_type=cfg.input_type_c,
        )

        # ------------------------ student projection heads ------------------------
        self.proj_a = nn.Linear(cfg.d_model, cfg.embed_dim)
        self.proj_b = nn.Linear(cfg.d_model, cfg.embed_dim)
        self.proj_c = nn.Linear(cfg.d_model, cfg.embed_dim)

        # ------------------------ temperature / queue / momentum ------------------------
        self.temp = nn.Parameter(torch.ones([]) * cfg.temp_init)
        self.queue_size = cfg.queue_size
        self.momentum = cfg.momentum

        # ------------------------ ITM head ------------------------
        self.itm_head = nn.Linear(cfg.d_model, 2)
        self.sup_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model * 3),
            nn.Linear(cfg.d_model * 3, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.label_dim),
        )

        # ------------------------ teacher encoders ------------------------
        self.enc_a_m = TransformerEncoder(
            in_dim=dim_a,
            num_tokens=cfg.num_tokens_a,
            cfg=cfg,
            input_type=cfg.input_type_a,
        )
        self.enc_b_m = TransformerEncoder(
            in_dim=dim_b,
            num_tokens=cfg.num_tokens_b,
            cfg=cfg,
            input_type=cfg.input_type_b,
        )
        self.enc_c_m = TransformerEncoder(
            in_dim=dim_c,
            num_tokens=cfg.num_tokens_c,
            cfg=cfg,
            input_type=cfg.input_type_c,
        )

        # ------------------------ teacher projection heads ------------------------
        self.proj_a_m = nn.Linear(cfg.d_model, cfg.embed_dim)
        self.proj_b_m = nn.Linear(cfg.d_model, cfg.embed_dim)
        self.proj_c_m = nn.Linear(cfg.d_model, cfg.embed_dim)

        # ------------------------ student-teacher pairs ------------------------
        self.model_pairs = [
            [self.enc_a, self.enc_a_m],
            [self.enc_b, self.enc_b_m],
            [self.enc_c, self.enc_c_m],
            [self.proj_a, self.proj_a_m],
            [self.proj_b, self.proj_b_m],
            [self.proj_c, self.proj_c_m],
        ]

        # 调试用：确认 student / teacher 的 tokenizer 维度一致
        print("enc_a tokenizer proj:", self.enc_a.tokenizer.proj.weight.shape)
        print("enc_a_m tokenizer proj:", self.enc_a_m.tokenizer.proj.weight.shape)
        print("enc_b tokenizer proj:", self.enc_b.tokenizer.proj.weight.shape)
        print("enc_b_m tokenizer proj:", self.enc_b_m.tokenizer.proj.weight.shape)
        print("enc_c tokenizer proj:", self.enc_c.tokenizer.proj.weight.shape)
        print("enc_c_m tokenizer proj:", self.enc_c_m.tokenizer.proj.weight.shape)

        self.copy_params()

        # ------------------------ queues ------------------------
        self.register_buffer("a_queue", torch.randn(cfg.embed_dim, self.queue_size))
        self.register_buffer("b_queue", torch.randn(cfg.embed_dim, self.queue_size))
        self.register_buffer("c_queue", torch.randn(cfg.embed_dim, self.queue_size))
        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.a_queue = F.normalize(self.a_queue, dim=0)
        self.b_queue = F.normalize(self.b_queue, dim=0)
        self.c_queue = F.normalize(self.c_queue, dim=0)

        # ------------------------ lightweight fusion encoder for ITM ------------------------
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=int(cfg.d_model * cfg.mlp_ratio),
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fuser = nn.TransformerEncoder(layer, num_layers=1)

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for p, p_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                p_m.data.copy_(p.data)
                p_m.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for p, p_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                p_m.data = p_m.data * self.momentum + p.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, a_feat, b_feat, c_feat, idx):
        a_feats = concat_all_gather(a_feat)
        b_feats = concat_all_gather(b_feat)
        c_feats = concat_all_gather(c_feat)
        idxs = concat_all_gather(idx)

        bs = a_feats.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % bs == 0, "queue_size must be divisible by batch_size (for simplicity)"

        self.a_queue[:, ptr:ptr + bs] = a_feats.T
        self.b_queue[:, ptr:ptr + bs] = b_feats.T
        self.c_queue[:, ptr:ptr + bs] = c_feats.T
        self.idx_queue[:, ptr:ptr + bs] = idxs.T

        ptr = (ptr + bs) % self.queue_size
        self.queue_ptr[0] = ptr

    # ------------------------ helper: ITC pair loss ------------------------
    def _itc_pair(
        self,
        x_feat, y_feat,
        x_feat_m, y_feat_m,
        y_queue, x_queue,
        idx, idx_all, alpha: float, soft_pos_mask: Optional[torch.Tensor] = None
    ):
        x_feat = torch.nan_to_num(x_feat, nan=0.0, posinf=0.0, neginf=0.0)
        y_feat = torch.nan_to_num(y_feat, nan=0.0, posinf=0.0, neginf=0.0)
        x_feat_m = torch.nan_to_num(x_feat_m, nan=0.0, posinf=0.0, neginf=0.0)
        y_feat_m = torch.nan_to_num(y_feat_m, nan=0.0, posinf=0.0, neginf=0.0)
        """
        """
        # targets: 如果队列里出现相同 idx，也视为正（检索式写法）
        pos_idx = torch.eq(idx, idx_all).float()
        if soft_pos_mask is not None:
            bs = idx.shape[0]
            pos_idx[:, :bs] = torch.maximum(pos_idx[:, :bs], soft_pos_mask.float())
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        # teacher all
        with torch.no_grad():
            x_all_m = torch.cat([x_feat_m.t(), x_queue.clone().detach()], dim=1)
            y_all_m = torch.cat([y_feat_m.t(), y_queue.clone().detach()], dim=1)

            if self.distill:
                sim_x2y_m = x_feat_m @ y_all_m / self.temp
                sim_y2x_m = y_feat_m @ x_all_m / self.temp
                sim_x2y_targets = alpha * F.softmax(sim_x2y_m, dim=1) + (1 - alpha) * sim_targets
                sim_y2x_targets = alpha * F.softmax(sim_y2x_m, dim=1) + (1 - alpha) * sim_targets

        # student logits (use teacher queues)
        y_all = torch.cat([y_feat_m.t(), y_queue.clone().detach()], dim=1)
        x_all = torch.cat([x_feat_m.t(), x_queue.clone().detach()], dim=1)

        sim_x2y = x_feat @ y_all / self.temp
        sim_y2x = y_feat @ x_all / self.temp

        if self.distill:
            loss_x2y = -torch.sum(F.log_softmax(sim_x2y, dim=1) * sim_x2y_targets, dim=1).mean()
            loss_y2x = -torch.sum(F.log_softmax(sim_y2x, dim=1) * sim_y2x_targets, dim=1).mean()
        else:
            loss_x2y = -torch.sum(F.log_softmax(sim_x2y, dim=1) * sim_targets, dim=1).mean()
            loss_y2x = -torch.sum(F.log_softmax(sim_y2x, dim=1) * sim_targets, dim=1).mean()

        return 0.5 * (loss_x2y + loss_y2x), sim_x2y, sim_y2x

    # ------------------------ helper: ITM for a pair ------------------------
    def _itm_pair(self, x_tokens, x_atts, y_tokens, y_atts, sim_x2y, sim_y2x, idx, false_neg_mask: Optional[torch.Tensor] = None):
        """
        ITM with robust negative sampling.
        """
        device = x_tokens.device
        bs = x_tokens.size(0)

        with torch.no_grad():
            weights_x2y = F.softmax(sim_x2y[:, :bs], dim=1)
            weights_y2x = F.softmax(sim_y2x[:, :bs], dim=1)
            if not torch.isfinite(weights_x2y).all():
                print("[ITM Debug] weights_x2y has NaN/Inf")
            if not torch.isfinite(weights_y2x).all():
                print("[ITM Debug] weights_y2x has NaN/Inf")
            if not torch.isfinite(sim_x2y).all():
                print("[ITM Debug] sim_x2y has NaN/Inf")
            if not torch.isfinite(sim_y2x).all():
                print("[ITM Debug] sim_y2x has NaN/Inf")

            mask = torch.eq(idx, idx.T)  # same idx cannot be negative
            if false_neg_mask is not None:
                mask = mask | false_neg_mask.bool()
            weights_x2y.masked_fill_(mask, 0.0)
            weights_y2x.masked_fill_(mask, 0.0)
            if (weights_x2y.sum(dim=1) <= 0).any() or (weights_y2x.sum(dim=1) <= 0).any():
                return x_tokens.new_zeros(())

        def _safe_sample(row: torch.Tensor, self_idx: int) -> int:
            row = row.clone()

            # 1) 把 NaN / Inf 清掉
            row = torch.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)

            # 2) 不允许采自己
            if 0 <= self_idx < row.numel():
                row[self_idx] = 0.0

            # 3) 如果整行无效，则退化为均匀采样（除自己外）
            s = row.sum()
            if (not torch.isfinite(s)) or s <= 0:
                row = torch.ones_like(row)
                if 0 <= self_idx < row.numel():
                    row[self_idx] = 0.0
                s = row.sum()

                # 极端情况：batch_size == 1
                if (not torch.isfinite(s)) or s <= 0:
                    return self_idx

            # 4) 归一化后再采样
            row = row / row.sum()
            return torch.multinomial(row, 1).item()

        # y_neg for each x
        y_neg_tokens = []
        y_neg_atts = []
        for b in range(bs):
            neg_idx = _safe_sample(weights_x2y[b], b)
            y_neg_tokens.append(y_tokens[neg_idx])
            y_neg_atts.append(y_atts[neg_idx])
        y_neg_tokens = torch.stack(y_neg_tokens, dim=0)
        y_neg_atts = torch.stack(y_neg_atts, dim=0)

        # x_neg for each y
        x_neg_tokens = []
        x_neg_atts = []
        for b in range(bs):
            neg_idx = _safe_sample(weights_y2x[b], b)
            x_neg_tokens.append(x_tokens[neg_idx])
            x_neg_atts.append(x_atts[neg_idx])
        x_neg_tokens = torch.stack(x_neg_tokens, dim=0)
        x_neg_atts = torch.stack(x_neg_atts, dim=0)

        def fuse(x_tok, y_tok):
            xy = torch.cat([x_tok, y_tok], dim=1)
            return self.fuser(xy)[:, 0, :]

        vl_pos = fuse(x_tokens, y_tokens)
        vl_neg1 = fuse(x_tokens, y_neg_tokens)
        vl_neg2 = fuse(x_neg_tokens, y_tokens)

        vl = torch.cat([vl_pos, vl_neg1, vl_neg2], dim=0)
        logits = self.itm_head(vl)

        labels = torch.cat([
            torch.ones(bs, dtype=torch.long),
            torch.zeros(2 * bs, dtype=torch.long),
        ], dim=0).to(device)

        return F.cross_entropy(logits, labels)

    def _supervised_loss(self, a_tokens, b_tokens, c_tokens, targets: Optional[torch.Tensor]):
        pred = self.sup_head(torch.cat([a_tokens[:, 0, :], b_tokens[:, 0, :], c_tokens[:, 0, :]], dim=-1))
        if targets is None:
            return pred.new_zeros(()), pred
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        targets = targets.to(device=pred.device, dtype=pred.dtype)
        return F.smooth_l1_loss(pred, targets), pred

    def _inbatch_itc_pair(self, x_feat, y_feat, pos_mask: Optional[torch.Tensor] = None):
        logits_x2y = x_feat @ y_feat.t() / self.temp
        logits_y2x = y_feat @ x_feat.t() / self.temp
        bs = x_feat.size(0)
        if pos_mask is None:
            targets = torch.eye(bs, device=x_feat.device, dtype=x_feat.dtype)
        else:
            targets = pos_mask.to(device=x_feat.device, dtype=x_feat.dtype)
            targets = torch.maximum(targets, torch.eye(bs, device=x_feat.device, dtype=x_feat.dtype))
        targets = targets / targets.sum(1, keepdim=True).clamp_min(1.0)
        loss_x2y = -torch.sum(F.log_softmax(logits_x2y, dim=1) * targets, dim=1).mean()
        loss_y2x = -torch.sum(F.log_softmax(logits_y2x, dim=1) * targets, dim=1).mean()
        return 0.5 * (loss_x2y + loss_y2x)

    def forward_inbatch_itc(
        self,
        xa, xb, xc,
        mask_a: Optional[torch.Tensor] = None,
        mask_b: Optional[torch.Tensor] = None,
        mask_c: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        soft_pos_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        a_tokens, _ = self.enc_a(xa, key_padding_mask=mask_a)
        b_tokens, _ = self.enc_b(xb, key_padding_mask=mask_b)
        c_tokens, _ = self.enc_c(xc, key_padding_mask=mask_c)

        a_feat = F.normalize(self.proj_a(a_tokens[:, 0, :]), dim=-1)
        b_feat = F.normalize(self.proj_b(b_tokens[:, 0, :]), dim=-1)
        c_feat = F.normalize(self.proj_c(c_tokens[:, 0, :]), dim=-1)

        loss_ab = self._inbatch_itc_pair(a_feat, b_feat, soft_pos_mask)
        loss_ac = self._inbatch_itc_pair(a_feat, c_feat, soft_pos_mask)
        loss_bc = self._inbatch_itc_pair(b_feat, c_feat, soft_pos_mask)
        loss_itc = (loss_ab + loss_ac + loss_bc) / 3.0
        loss_sup, pred_y = self._supervised_loss(a_tokens, b_tokens, c_tokens, targets)

        return {
            "loss_itc": loss_itc,
            "loss_sup": loss_sup,
            "pred_y": pred_y,
            "temp": self.temp.detach(),
        }

    # ------------------------ main forward ------------------------
    def forward(
        self,
        xa, xb, xc,
        alpha: float,
        idx: torch.Tensor,
        mask_a: Optional[torch.Tensor] = None,
        mask_b: Optional[torch.Tensor] = None,
        mask_c: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        soft_pos_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        xa/xb/xc:
          - [B, D] or [B, T, D]
        mask_*:
          - 若输入为序列 [B,T,D] 并且需要 padding，提供 key_padding_mask: [B,T] True=padding
        idx: [B] long，每个样本唯一 id（用于 queue positives）
        """
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)  # [1, B+Q]

        # student tokens
        a_tokens, a_atts = self.enc_a(xa, key_padding_mask=mask_a)
        b_tokens, b_atts = self.enc_b(xb, key_padding_mask=mask_b)
        c_tokens, c_atts = self.enc_c(xc, key_padding_mask=mask_c)

        a_feat = F.normalize(self.proj_a(a_tokens[:, 0, :]), dim=-1)
        b_feat = F.normalize(self.proj_b(b_tokens[:, 0, :]), dim=-1)
        c_feat = F.normalize(self.proj_c(c_tokens[:, 0, :]), dim=-1)

        # momentum teacher tokens
        # momentum teacher tokens
        with torch.no_grad():
            if self.training:
                self._momentum_update()

                a_tokens_m, _ = self.enc_a_m(xa, key_padding_mask=mask_a)
                b_tokens_m, _ = self.enc_b_m(xb, key_padding_mask=mask_b)
                c_tokens_m, _ = self.enc_c_m(xc, key_padding_mask=mask_c)

                a_feat_m = F.normalize(self.proj_a_m(a_tokens_m[:, 0, :]), dim=-1)
                b_feat_m = F.normalize(self.proj_b_m(b_tokens_m[:, 0, :]), dim=-1)
                c_feat_m = F.normalize(self.proj_c_m(c_tokens_m[:, 0, :]), dim=-1)
            else:
                # eval 时不更新 teacher，直接用当前 student 特征代替
                a_feat_m = a_feat.detach()
                b_feat_m = b_feat.detach()
                c_feat_m = c_feat.detach()

        # -------- ITC: three pairs --------
        loss_ab_itc, sim_a2b, sim_b2a = self._itc_pair(
            a_feat, b_feat, a_feat_m, b_feat_m,
            y_queue=self.b_queue, x_queue=self.a_queue,
            idx=idx, idx_all=idx_all, alpha=alpha, soft_pos_mask=soft_pos_mask
        )
        loss_ac_itc, sim_a2c, sim_c2a = self._itc_pair(
            a_feat, c_feat, a_feat_m, c_feat_m,
            y_queue=self.c_queue, x_queue=self.a_queue,
            idx=idx, idx_all=idx_all, alpha=alpha, soft_pos_mask=soft_pos_mask
        )
        loss_bc_itc, sim_b2c, sim_c2b = self._itc_pair(
            b_feat, c_feat, b_feat_m, c_feat_m,
            y_queue=self.c_queue, x_queue=self.b_queue,
            idx=idx, idx_all=idx_all, alpha=alpha, soft_pos_mask=soft_pos_mask
        )
        loss_itc = (loss_ab_itc + loss_ac_itc + loss_bc_itc) / 3.0

        if self.training:
            a_feat_m = torch.nan_to_num(a_feat_m, nan=0.0, posinf=0.0, neginf=0.0)
            b_feat_m = torch.nan_to_num(b_feat_m, nan=0.0, posinf=0.0, neginf=0.0)
            c_feat_m = torch.nan_to_num(c_feat_m, nan=0.0, posinf=0.0, neginf=0.0)
            self._dequeue_and_enqueue(a_feat_m, b_feat_m, c_feat_m, idx)

        # -------- ITM: three pairs (hard negatives in-batch) --------
        false_neg_mask = soft_pos_mask
        loss_ab_itm = self._itm_pair(a_tokens, a_atts, b_tokens, b_atts, sim_a2b, sim_b2a, idx, false_neg_mask)
        loss_ac_itm = self._itm_pair(a_tokens, a_atts, c_tokens, c_atts, sim_a2c, sim_c2a, idx, false_neg_mask)
        loss_bc_itm = self._itm_pair(b_tokens, b_atts, c_tokens, c_atts, sim_b2c, sim_c2b, idx, false_neg_mask)
        loss_itm = (loss_ab_itm + loss_ac_itm + loss_bc_itm) / 3.0
        loss_sup, pred_y = self._supervised_loss(a_tokens, b_tokens, c_tokens, targets)

        return {
            "loss_itc": loss_itc,
            "loss_itm": loss_itm,
            "loss_sup": loss_sup,
            "pred_y": pred_y,
            "loss": loss_itc + loss_itm,
            "temp": self.temp.detach(),
        }
