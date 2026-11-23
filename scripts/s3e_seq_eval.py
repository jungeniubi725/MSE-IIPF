"""S3e：样本级（sequence 级）评估。

unit 级预测 -> sequence 级聚合，一个样本（sequence）对应一个预测，再算
样本级指标（Accuracy / macro·micro·weighted P·R·F1 / per-class / 混淆矩阵 /
不确定性）。RAVDESS 的正确评估粒度是 sequence 级（208 个视频），而非 unit 级
（832 个片段）——evaluate() 默认按 unit 报，本脚本补上 sequence 级口径。

聚合方式（两种，主报告用证据聚合）：
  evidence（主）  EDL 的 Dirichlet 证据可加性：α_seq = 1 + Σ_t(α_t−1)，
                  p_seq = α_seq / Σα_seq，u_seq = K / Σα_seq。
                  —— 证据理论下最自然的序列聚合，同时给出序列级不确定性。
  probmean（对照） p_seq = mean_t p_t，直接对 unit 概率求平均。

输出（默认 runs/<exp-name>/seq_eval）：
    seq_predictions.csv   每个 sequence 一行：seq_id, true, pred, u, p0..p{K-1}
    seq_metrics.json      样本级指标全量
    终端打印              两种聚合方式的 acc / macro-F1 对比

用法：
    python scripts/s3e_seq_eval.py --config configs/ravdess.yaml \
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
from src.metrics.classification import classification_metrics, confusion_mat, per_class_metrics
from src.metrics.uncertainty import uncertainty_report
from src.models.mse_iipf import MSEIIPF
from src.utils.config import load_config
from src.utils.console import setup_console
from src.utils.evidential import aggregate_dirichlet_evidence

setup_console()

import scripts.s3_train as s3  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S3e 样本级（sequence 级）评估")
    p.add_argument("--config", required=True)
    p.add_argument("--exp-name", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--ckpt", default=None, help="默认 runs/<exp-name>/best.pt")
    p.add_argument("--out-dir", default=None, help="默认 runs/<exp-name>/seq_eval")
    p.add_argument("--seed", type=int, default=None,
                   help="CV 折的种子，配合 --fold 使用")
    p.add_argument("--fold", type=int, default=None,
                   help="CV 折号；给定后 ckpt/loader 指向 seed{seed}/fold{fold}")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_model(cfg: dict) -> MSEIIPF:
    m = dict(cfg["model"])
    m["num_classes"] = cfg["data"]["num_classes"]
    for k in ("itac_weights", "mu", "w_tma", "w_iipf"):
        if k in m and isinstance(m[k], list):
            m[k] = tuple(m[k])
    return MSEIIPF(**m)


def load_effective_config(base_path: str, ckpt: str) -> tuple[dict, str]:
    """Load the exact saved run config, falling back to the base YAML.

    Some ablation flags alter inference (for example
    ``use_semantic_stream``). Training writes a complete ``config.json`` at
    the run/seed directory, so the closest saved config above the checkpoint
    must take precedence when rebuilding the model for evaluation.
    """
    cfg = load_config(base_path)
    current = os.path.abspath(os.path.dirname(ckpt))
    while True:
        saved = os.path.join(current, "config.json")
        if os.path.isfile(saved):
            with open(saved, encoding="utf-8") as f:
                return json.load(f), saved
        parent = os.path.dirname(current)
        if parent == current or os.path.basename(current).lower() == "runs":
            break
        current = parent
    return cfg, os.path.abspath(base_path)


def main() -> int:
    args = parse_args()
    base_cfg = load_config(args.config)

    if args.fold is not None:
        seed = args.seed if args.seed is not None else int(base_cfg["train"].get("seed", 42))
        sub = f"seed{seed}/fold{args.fold}"
        fold = args.fold
    else:
        sub = ""
        fold = None

    ckpt = args.ckpt or os.path.join("runs", args.exp_name, sub, "best.pt")
    if not os.path.exists(ckpt):
        print(f"[S3e] checkpoint 不存在：{ckpt}，先跑 s3_train.py", file=sys.stderr)
        return 1
    cfg, cfg_source = load_effective_config(args.config, ckpt)
    K = cfg["data"]["num_classes"]
    class_names = cfg["data"].get("class_names") or [str(i) for i in range(K)]
    out_dir = args.out_dir or os.path.join("runs", args.exp_name, sub, "seq_eval")
    os.makedirs(out_dir, exist_ok=True)

    loader = s3.build_loaders(cfg, fold=fold, evaluation=True, only_split=args.split)[args.split]
    model = build_model(cfg).to(args.device)
    state = torch.load(ckpt, map_location=args.device, weights_only=True)
    if not cfg["model"].get("use_evidential", True):
        raise ValueError("Evidence aggregation needs an evidential head; use scripts/evaluate.py for Softmax")
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[S3e] 载入 {ckpt}（epoch {state.get('epoch')}）  split={args.split}  "
          f"{len(loader.dataset)} sequences  config={cfg_source}")

    # 顺序收集（test 集 shuffle=False，seq 顺序确定、跨 batch 不重复）
    seq_ids: list[str] = []
    seq_label: list[int] = []
    alpha_ev: list[np.ndarray] = []   # 证据聚合：每 seq 的 1+Σ(α−1)
    p_mean: list[np.ndarray] = []     # 概率平均：每 seq 的 mean p
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
            for b, sid in enumerate(batch["seq_ids"]):
                valid = mask_t[b] > 0.5
                a = out["alpha"][b][valid]            # (T', K)
                p = out["p_hat"][b][valid]            # (T', K)
                seq_ids.append(sid)
                seq_label.append(int(batch["seq_labels"][b]))
                alpha_seq = aggregate_dirichlet_evidence(a, dim=0)
                alpha_ev.append(alpha_seq.float().cpu().numpy())
                p_mean.append(p.mean(0).float().cpu().numpy())

    y_true = np.asarray(seq_label, dtype=np.int64)
    alpha_ev = np.stack(alpha_ev)
    p_mean = np.stack(p_mean)

    # 证据聚合：α_seq -> p_seq -> y_pred，以及序列级不确定性 u_seq = K / Σα
    p_ev = alpha_ev / alpha_ev.sum(-1, keepdims=True)
    y_ev = p_ev.argmax(-1)
    u_ev = K / alpha_ev.sum(-1)
    # 概率平均对照
    y_pm = p_mean.argmax(-1)

    # ---- 指标（主报告用证据聚合）----
    m = classification_metrics(y_true, y_ev, K)
    pc = per_class_metrics(y_true, y_ev, K, class_names)
    cm = confusion_mat(y_true, y_ev, K)
    unc = uncertainty_report(y_true, y_ev, p_ev, u_ev, kappa=None)
    metrics = {
        "n_sequences": int(len(y_true)),
        "split": args.split, "epoch": state.get("epoch"),
        "config_source": os.path.abspath(cfg_source),
        "aggregation": "evidence_sum_single_prior",
        "aggregation_formula": "alpha_seq = 1 + sum_t(alpha_t - 1)",
        **m, "per_class": pc,
        "confusion_matrix": cm.tolist(),
        "uncertainty": unc,
        # 概率平均对照，仅关键项
        "probmean_acc": float((y_pm == y_true).mean()),
        "probmean_macro_f1": float(
            __import__("sklearn.metrics", fromlist=["f1_score"])
            .f1_score(y_true, y_pm, average="macro", zero_division=0)),
    }
    with open(os.path.join(out_dir, "seq_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=float)

    # ---- 预测落盘 ----
    csv_path = os.path.join(out_dir, "seq_predictions.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["seq_id", "true", "pred", "u"] + [f"p{i}" for i in range(K)])
        for sid, yt, yp, u, p in zip(seq_ids, y_true, y_ev, u_ev, p_ev):
            w.writerow([sid, int(yt), int(yp), "%.9g" % u] + ["%.9g" % x for x in p])

    # ---- 终端摘要 ----
    print(f"[S3e] 样本级评估（{len(y_true)} sequences，证据聚合）：")
    print(f"      acc {m['acc']:.4f}  macro-F1 {m['macro_f1']:.4f}  "
          f"weighted-F1 {m['weighted_f1']:.4f}")
    print(f"      macro-P {m['macro_p']:.4f}  macro-R {m['macro_r']:.4f}  "
          f"micro-F1 {m['micro_f1']:.4f}")
    print(f"      证据聚合 acc {m['acc']:.4f}  vs  概率平均 acc {metrics['probmean_acc']:.4f}")
    u = unc
    print(f"      ECE {u['ece']:.4f}  AUROC(u) {u.get('auroc_u', float('nan')):.4f}  "
          f"avg_u {u.get('avg_u', float('nan')):.4f}")
    print(f"[S3e] 完成 -> {out_dir}（seq_predictions.csv + seq_metrics.json）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
