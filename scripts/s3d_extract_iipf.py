"""S3d：提取 IIPF 互补融合特征 f_t 并落盘 CSV，供后续直接分类。

数据流
------
best.pt 里 IIPF 把三模态（TMA 对齐后的 g_v/g_a/g_l）互补融合成 f_t
（论文 E49–E52，d_f=256）。分类头的证据网络 evidence_net 只吃 f_t，
故 f_t 是「融合完成、可直接分类」的表征。本脚本把它保存为 CSV，
每行一个时间单元：id + 256 维特征 + label。

同时把 evidence_net 的权重单独存为 .pt，并用它直接对这批 f_t 分类，
输出预测，验证「f_t -> 分类头」这条捷径与原模型完全一致。

输出（默认 runs/<exp-name>/iipf_feats）：
    fusion.csv        unit 级  id, f0..f255, label
    evidence_net.pt   分类头权重（Linear(256,256)->ReLU->Linear(256,K)）
    meta.json         形状与配置
    终端打印          捷径分类 vs 原模型预测的一致性核对

用法：
    python scripts/s3d_extract_iipf.py --config configs/ravdess.yaml \
        --exp-name main --split test
"""
from __future__ import annotations

import argparse
import csv
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

import scripts.s3_train as s3  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S3d 提取 IIPF 融合特征 f_t")
    p.add_argument("--config", required=True)
    p.add_argument("--exp-name", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--ckpt", default=None, help="默认 runs/<exp-name>/best.pt")
    p.add_argument("--out-dir", default=None, help="默认 runs/<exp-name>/iipf_feats")
    p.add_argument("--float-fmt", default="%.6g")
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
        print(f"[S3d] checkpoint 不存在：{ckpt}，先跑 s3_train.py", file=sys.stderr)
        return 1
    out_dir = args.out_dir or os.path.join("runs", args.exp_name, "iipf_feats")
    os.makedirs(out_dir, exist_ok=True)

    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg = state.get("cfg", cfg)
    loader = s3.build_loaders(cfg, evaluation=True, only_split=args.split)[args.split]
    ds = loader.dataset
    # seq_id -> unit_ids（unit_index 升序），用于把 batch 里的 unit 还原成 id
    seq2units = {g["seq_id"]: g["unit_ids"] for g in ds.groups}

    model = build_model(cfg).to(args.device)
    state = torch.load(ckpt, map_location=args.device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    d_f = cfg["model"]["d_f"]
    print(f"[S3d] 载入 {ckpt}（epoch {state.get('epoch')}, d_f={d_f}）")
    print(f"[S3d] split={args.split}  {len(ds)} sequences  device={args.device}")

    csv_path = os.path.join(out_dir, "fusion.csv")
    n_written, n_zero = 0, 0
    ids_out, feats_out, labels_out = [], [], []
    pred_shortcut, pred_model = [], []
    with torch.inference_mode():
        for batch in loader:
            B, T = batch["B"], batch["T"]
            mask_t = batch["mask_t"].to(args.device)
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
            f = out["f"]                              # (B, T, d_f)
            y_model = out["y_pred"]                   # (B, T) 原模型 unit 级预测
            keep = mask_t > 0.5                       # (B, T)

            # 捷径分类：f_t -> evidence_net，应得到与原模型一致的预测
            p_short = model.head._forward_evidential(f)[1].argmax(-1)

            ids_b, labs_b = [], []
            for b, sid in enumerate(batch["seq_ids"]):
                units = seq2units[sid]
                for t in range(T):
                    if keep[b, t]:
                        ids_b.append(units[t] if t < len(units) else f"{sid}#{t}")
                        labs_b.append(int(batch["labels"][b, t]))
            ids_out.extend(ids_b)
            labels_out.extend(labs_b)
            feats_out.append(f[keep].cpu().numpy())
            pred_model.append(y_model[keep].cpu().numpy())
            pred_shortcut.append(p_short[keep].cpu().numpy())

    feats = np.concatenate(feats_out, axis=0)
    pred_model = np.concatenate(pred_model)
    pred_shortcut = np.concatenate(pred_shortcut)
    labels_out = np.asarray(labels_out, dtype=np.int64)

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["id"] + [f"f{i}" for i in range(d_f)] + ["label"])
        for i, (uid, vec, lab) in enumerate(zip(ids_out, feats, labels_out)):
            if not np.any(vec):
                n_zero += 1
            w.writerow([uid] + [args.float_fmt % x for x in vec] + [int(lab)])
            n_written += 1

    # 分类头权重单独存，供「只加载分类头」的后续流程
    torch.save(model.head.evidence_net.state_dict(),
               os.path.join(out_dir, "evidence_net.pt"))

    agree = float((pred_shortcut == pred_model).mean())
    acc = float((pred_model == labels_out).mean())
    meta = {
        "split": args.split, "ckpt": ckpt, "epoch": state.get("epoch"),
        "d_f": d_f, "n_unit": n_written, "n_zero_rows": n_zero,
        "n_class": cfg["data"]["num_classes"],
        "class_names": cfg["data"].get("class_names"),
        "shortcut_vs_model_agree": agree, "unit_acc": acc,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[S3d] 写 {n_written} 行 -> {csv_path}（全零行 {n_zero}）")
    print(f"[S3d] 捷径 f_t->evidence_net 与原模型预测一致率：{agree:.4f}")
    print(f"[S3d] unit 级 acc（原模型）：{acc:.4f}  "
          f"（捷径 acc 应相等：{(pred_shortcut == labels_out).mean():.4f}）")
    print(f"[S3d] 完成 -> {out_dir}（fusion.csv + evidence_net.pt + meta.json）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
