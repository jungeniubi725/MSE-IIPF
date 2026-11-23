"""S3b：加载训练好的 MSE-IIPF，前向提取 TMA 对齐后的三模态特征。

用途
----
训练得到的 best.pt 里，TMA 已经把视觉/音频/文本投影到**统一情感语义空间**
（d=256，论文 E20–E24 的 g_m）。本脚本对指定 split 前向，把这三路对齐特征
落盘，供对齐效果可视化（三模态一致性、类别聚类）与外部分析。

输出（写到 --out-dir，默认 runs/<exp-name>/tma_feats）：
    aligned.npz      unit 级  g_v/g_a/g_l (N,256) + label (N,) + seq_idx (N,)
    aligned_seq.npz  seq 级   g_v/g_a/g_l (S,256) + seq_label (S,) + seq_id (S,)
    meta.json        形状与配置记录

对齐后的 g 是 unit 级（B×T 展平），mask_t 为 0 的 padding 位已剔除。
sequence 级由该 seq 的 T 个 unit 在 mask_t 上做 mean 得到（E24 的 Pool_mean）。

用法：
    python scripts/s3b_extract_tma.py --config configs/ravdess.yaml \
        --exp-name main --split test
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.evaluator import _expand_seq_labels
from src.models.mse_iipf import MSEIIPF
from src.utils.config import load_config
from src.utils.console import setup_console

setup_console()

# 复用 s3_train 的建模/装载逻辑（避免复制粘贴漂移）
import scripts.s3_train as s3  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S3b 提取 TMA 对齐特征")
    p.add_argument("--config", required=True)
    p.add_argument("--exp-name", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--ckpt", default=None, help="默认 runs/<exp-name>/best.pt")
    p.add_argument("--out-dir", default=None,
                   help="默认 runs/<exp-name>/tma_feats")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_model(cfg: dict) -> MSEIIPF:
    m = dict(cfg["model"])
    m["num_classes"] = cfg["data"]["num_classes"]
    for k in ("itac_weights", "mu", "w_tma", "w_iipf"):
        if k in m and isinstance(m[k], list):
            m[k] = tuple(m[k])
    return MSEIIPF(**m)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    ckpt = args.ckpt or os.path.join("runs", args.exp_name, "best.pt")
    if not os.path.exists(ckpt):
        print(f"[S3b] checkpoint 不存在：{ckpt}，先跑 s3_train.py", file=sys.stderr)
        return 1
    out_dir = args.out_dir or os.path.join("runs", args.exp_name, "tma_feats")
    os.makedirs(out_dir, exist_ok=True)

    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg = state.get("cfg", cfg)
    loader = s3.build_loaders(cfg, evaluation=True, only_split=args.split)[args.split]
    model = build_model(cfg).to(args.device)
    state = torch.load(ckpt, map_location=args.device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[S3b] 载入 {ckpt}（epoch {state.get('epoch')}, "
          f"{state.get('metric')}={state.get('score', float('nan')):.4f}）")
    print(f"[S3b] split={args.split}  {len(loader.dataset)} sequences  "
          f"device={args.device}")

    g_v, g_a, g_l, labs, seq_of_unit = [], [], [], [], []
    sv, sa, sl, seq_lab, sids = [], [], [], [], []
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            B, T = batch["B"], batch["T"]
            mask_t = batch["mask_t"].to(args.device)  # (B, T)
            out, _ = model(
                batch["h_v"].to(args.device), batch["h_a"].to(args.device),
                batch["h_sv"].to(args.device), batch["h_sa"].to(args.device),
                batch["labels"].to(args.device), epoch=state.get("epoch", 0),
                mask_v=batch["mask_v"].to(args.device),
                mask_a=batch["mask_a"].to(args.device),
                mask_sv=batch["mask_sv"].to(args.device),
                mask_sa=batch["mask_sa"].to(args.device),
                mask_t=mask_t,
                labels_for_itam=_expand_seq_labels(batch["seq_labels"], T).to(args.device),
                seq_labels=batch["seq_labels"].to(args.device),
            )
            d = out["g_v"].size(-1)
            gv = out["g_v"].view(B, T, d)
            ga = out["g_a"].view(B, T, d)
            gl = out["g_l"].view(B, T, d)
            lab = batch["labels"]  # (B, T)

            keep = mask_t.view(-1) > 0.5                      # (B*T,)
            keep2d = mask_t > 0.5                             # (B, T)

            # ---- unit 级：mask_t 过滤 padding ----
            g_v.append(gv.reshape(-1, d)[keep].cpu().numpy())
            g_a.append(ga.reshape(-1, d)[keep].cpu().numpy())
            g_l.append(gl.reshape(-1, d)[keep].cpu().numpy())
            labs.append(lab.reshape(-1)[keep.cpu()].numpy())
            # 每个真实 unit 属于哪个全局 seq（0..S-1）
            seqid = torch.arange(B).unsqueeze(1).expand(B, T).to(mask_t.device)
            seq_of_unit.append((seqid.reshape(-1)[keep] + offset).cpu().numpy())

            # ---- sequence 级：T 上 mask_t 加权 mean ----
            w = keep2d.unsqueeze(-1).float()
            sv.append((gv * w).sum(1).div(w.sum(1).clamp_min(1)).cpu().numpy())
            sa.append((ga * w).sum(1).div(w.sum(1).clamp_min(1)).cpu().numpy())
            sl.append((gl * w).sum(1).div(w.sum(1).clamp_min(1)).cpu().numpy())
            seq_lab.append(batch["seq_labels"].cpu().numpy())
            sids.extend(batch["seq_ids"])
            offset += B

    g_v = np.concatenate(g_v); g_a = np.concatenate(g_a); g_l = np.concatenate(g_l)
    labs = np.concatenate(labs).astype(np.int64)
    seq_of_unit = np.concatenate(seq_of_unit).astype(np.int64)
    sv = np.concatenate(sv); sa = np.concatenate(sa); sl = np.concatenate(sl)
    seq_lab = np.concatenate(seq_lab).astype(np.int64)
    sids = np.asarray(sids)

    np.savez_compressed(
        os.path.join(out_dir, "aligned.npz"),
        g_v=g_v, g_a=g_a, g_l=g_l, label=labs, seq_idx=seq_of_unit)
    np.savez_compressed(
        os.path.join(out_dir, "aligned_seq.npz"),
        g_v=sv, g_a=sa, g_l=sl, seq_label=seq_lab, seq_id=sids)

    meta = {
        "split": args.split, "ckpt": ckpt, "epoch": state.get("epoch"),
        "score": float(state.get("score", float("nan"))), "d": int(d),
        "n_unit": int(g_v.shape[0]), "n_seq": int(sv.shape[0]),
        "n_class": cfg["data"]["num_classes"],
        "class_names": cfg["data"].get("class_names"),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 简洁的模态间余弦一致性汇总，供快速判断对齐程度
    def _cos(a, b):
        a = a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-8)
        b = b / np.linalg.norm(b, axis=1, keepdims=True).clip(1e-8)
        return float((a * b).sum(1).mean())
    print(f"[S3b] unit 级 N={g_v.shape[0]}  对齐后三模态余弦一致性：")
    print(f"      cos(g_v,g_a)={_cos(g_v, g_a):.4f}  "
          f"cos(g_v,g_l)={_cos(g_v, g_l):.4f}  "
          f"cos(g_a,g_l)={_cos(g_a, g_l):.4f}")
    print(f"[S3b] 完成 -> {out_dir}（unit {g_v.shape[0]} / seq {sv.shape[0]}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
