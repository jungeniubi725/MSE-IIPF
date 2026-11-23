"""训练循环。AMP + 梯度裁剪 + 辅助损失 warmup + 按 val macro-F1 选最优。

辅助损失 warmup 的理由
--------------------
L_DSA、L_ITAC、L_TPC 等辅助项在训练初期数值远大于分类损失（实测 L_DSA≈3.8
而 L_head≈2.7），若一开始就全权重施加，会压制分类边界的学习。故前 N 个
epoch 对所有辅助项线性升权，分类损失（L_head）始终全权重。

KL 退火（E48）由模型内部按 epoch 处理，此处只负责传 epoch。
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from .evaluator import _expand_seq_labels, evaluate, format_metrics

# 辅助损失键（warmup 作用于这些，L_head 不参与）
AUX_KEYS = ("L_DSA", "L_MAM", "L_ITAC", "L_ITAM", "L_CMC", "L_TPC", "L_PRR")


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        cfg: dict,
        device: str = "cuda",
        out_dir: str = "runs/exp",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        t = cfg.get("train", {})
        self.epochs = int(t.get("epochs", 30))
        self.warmup_epochs = int(t.get("aux_warmup_epochs", 3))
        self.grad_clip = float(t.get("grad_clip", 1.0))
        self.accum = int(t.get("grad_accum", 1))
        if self.accum < 1 or self.epochs < 1:
            raise ValueError("epochs and grad_accum must be positive")
        self.amp = bool(t.get("amp", True)) and device.startswith("cuda")
        self.patience = int(t.get("early_stop_patience", 10))
        self.select_metric = t.get("select_metric", "macro_f1")

        self.optimizer = self._build_optimizer(t)
        self.scheduler = self._build_scheduler(t)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

        self.num_classes = int(cfg.get("data", {}).get("num_classes", 8))
        self.class_names = cfg.get("data", {}).get("class_names")
        self.history: list[dict] = []
        self.best = {"score": -1.0, "epoch": -1}

    def _build_optimizer(self, t: dict) -> torch.optim.Optimizer:
        lr = float(t.get("lr", 1e-4))
        wd = float(t.get("weight_decay", 1e-2))
        # LayerNorm 与 bias 不做权重衰减（常规做法）
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": wd},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=lr, betas=(0.9, 0.999),
        )

    def _build_scheduler(self, t: dict):
        name = str(t.get("scheduler", "cosine")).lower()
        if name == "none":
            return None
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(self.epochs, 1),
                eta_min=float(t.get("min_lr", 1e-6)),
            )
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=3,
            )
        raise ValueError(f"未知 scheduler: {name}")

    def _aux_scale(self, epoch: int) -> float:
        """辅助损失 warmup 系数，线性从 0 升到 1。"""
        if self.warmup_epochs <= 0:
            return 1.0
        return min(1.0, (epoch + 1) / self.warmup_epochs)

    # ------------------------------------------------------------------ 单 epoch
    def train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        scale = self._aux_scale(epoch)
        agg: dict[str, float] = {}
        n, t0 = 0, time.time()

        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(self.train_loader):
            h = {k: batch[k].to(self.device, non_blocking=True)
                 for k in ("h_v", "h_a", "h_sv", "h_sa", "mask_v", "mask_a",
                           "mask_sv", "mask_sa", "mask_t", "labels", "seq_labels")}
            y = h["labels"]
            with torch.autocast("cuda", enabled=self.amp):
                _, losses = self.model(
                    h["h_v"], h["h_a"], h["h_sv"], h["h_sa"], y, epoch=epoch,
                    mask_v=h["mask_v"], mask_a=h["mask_a"],
                    mask_sv=h["mask_sv"], mask_sa=h["mask_sa"], mask_t=h["mask_t"],
                    labels_for_itam=_expand_seq_labels(h["seq_labels"], y.shape[1]),
                    seq_labels=h["seq_labels"],
                )
                # 重算总损失以施加 warmup（模型内已算好各项，这里只调辅助权重）
                loss = losses["L_head"]
                for k in AUX_KEYS:
                    if k in losses:
                        w = self._loss_weight(k)
                        loss = loss + scale * w * losses[k]
                group_start = (step // self.accum) * self.accum
                group_size = min(self.accum, len(self.train_loader) - group_start)
                loss_scaled = loss / group_size

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}: {float(loss)}")

            self.scaler.scale(loss_scaled).backward()
            if (step + 1) % self.accum == 0 or step + 1 == len(self.train_loader):
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            n += 1
            agg["loss"] = agg.get("loss", 0.0) + float(loss.detach())
            for k, v in losses.items():
                if k != "loss" and v.dim() == 0:
                    agg[k] = agg.get(k, 0.0) + float(v.detach())

        out = {k: v / max(n, 1) for k, v in agg.items()}
        out["aux_scale"] = scale
        out["lr"] = self.optimizer.param_groups[0]["lr"]
        out["sec"] = time.time() - t0
        return out

    def _loss_weight(self, key: str) -> float:
        """从模型属性取该辅助损失的权重（与 mse_iipf.py 的定义保持一致）。"""
        m = self.model
        return {
            "L_DSA": getattr(m, "w_dsa", 1.0),
            "L_MAM": getattr(m, "w_tma_1", 0.1),
            "L_ITAC": getattr(m, "w_tma_2", 1.0),
            "L_ITAM": getattr(m, "w_tma_3", 0.5),
            "L_CMC": getattr(m, "w_iipf_cmc", 1.0),
            "L_TPC": getattr(m, "w_iipf_tpc", 1.0),
            "L_PRR": getattr(m, "w_iipf_prr", 1.0),
        }.get(key, 1.0)

    # ------------------------------------------------------------------ 主循环
    def fit(self) -> dict:
        bad = 0
        for epoch in range(self.epochs):
            tr = self.train_one_epoch(epoch)
            va = evaluate(self.model, self.val_loader, self.device,
                          self.num_classes, self.class_names, epoch, self.amp)
            if self.select_metric not in va:
                raise KeyError(
                    f"train.select_metric={self.select_metric!r} 未出现在验证指标中；"
                    f"可用标量指标: "
                    f"{sorted(k for k, v in va.items() if np.isscalar(v))}"
                )
            score = float(va[self.select_metric])

            if self.scheduler is not None:
                if isinstance(self.scheduler,
                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(score)
                else:
                    self.scheduler.step()

            print(f"  ep {epoch:3d}  train loss {tr['loss']:.4f}  "
                  f"aux×{tr['aux_scale']:.2f}  lr {tr['lr']:.2e}  "
                  f"{tr['sec']:.1f}s | " + format_metrics(va, "val "))

            rec = {"epoch": epoch, "train": tr,
                   "val": {k: v for k, v in va.items() if k != "_raw"}}
            self.history.append(rec)
            self.save_history()

            if score > self.best["score"]:
                self.best = {"score": score, "epoch": epoch,
                             "metric": self.select_metric}
                self.save_ckpt("best.pt", epoch, score)
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    print(f"  早停：{self.patience} 个 epoch 无提升")
                    break

        self.save_history()
        return self.best

    # ------------------------------------------------------------------ IO
    def save_ckpt(self, name: str, epoch: int, score: float) -> str:
        p = os.path.join(self.out_dir, name)
        torch.save({"model": self.model.state_dict(), "epoch": epoch,
                    "score": score, "cfg": self.cfg}, p)
        return p

    def save_history(self) -> str:
        p = os.path.join(self.out_dir, "history.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"best": self.best, "history": self.history}, f,
                      ensure_ascii=False, indent=2, default=_json_safe)
        return p


def _json_safe(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
