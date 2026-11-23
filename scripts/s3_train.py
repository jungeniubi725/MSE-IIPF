"""S3：训练单次实验。

用法
----

    # 固定划分训练
    python scripts/s3_train.py --config configs/ravdess.yaml --exp-name main

    # 5 折 CV
    python scripts/s3_train.py --config configs/ravdess.yaml --exp-name main_cv --cv

    # 超参覆盖（用于 §5.3 的扫描）
    python scripts/s3_train.py --config configs/ravdess.yaml --exp-name d512 \
        --set model.d=512 train.lr=5e-5

    # 消融（用于 §5.4）
    python scripts/s3_train.py --config configs/ravdess.yaml --exp-name A13 \
        --set model.use_cal=false

    # 只检查数据与模型能否跑通，不真正训练
    python scripts/s3_train.py --config configs/ravdess.yaml --dry-run

产物
----
    runs/<exp-name>/best.pt          最优权重
    runs/<exp-name>/history.json     逐 epoch 训练与验证指标
    runs/<exp-name>/test_metrics.json 测试集完整指标（含逐类、不确定性、α）
    runs/<exp-name>/predictions.npz  原始预测（供 s5 出混淆矩阵等表）
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import MERCDataset, make_collate
from src.engine.evaluator import evaluate, format_metrics
from src.engine.trainer import Trainer
from src.models.mse_iipf import MSEIIPF
from src.utils.config import apply_overrides, load_config
from src.utils.console import setup_console
from src.utils.seed import set_seed
from src.utils.artifacts import export_predictions, save_json

setup_console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S3 训练 MSE-IIPF")
    p.add_argument("--config", required=True)
    p.add_argument("--exp-name", default="default")
    p.add_argument("--out-root", default="runs")
    p.add_argument("--set", nargs="*", default=[], dest="overrides",
                   help="覆盖配置，形如 model.d=512 train.lr=5e-5")
    p.add_argument("--cv", action="store_true", help="跑 k 折说话人独立 CV")
    p.add_argument("--seed", type=int, default=None, help="覆盖配置里的 seed")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dry-run", action="store_true",
                   help="只建数据与模型、跑一个 batch 前向，不训练")
    p.add_argument(
        "--validation-only", action="store_true",
        help="超参数选择模式：训练后只评估 validation，完全不构造或读取 test split",
    )
    return p.parse_args()


def build_model(cfg: dict) -> MSEIIPF:
    m = dict(cfg["model"])
    m["num_classes"] = cfg["data"]["num_classes"]
    # YAML 里的列表转成 tuple（模型签名要求）
    for k in ("itac_weights", "mu", "w_tma", "w_iipf"):
        if k in m and isinstance(m[k], list):
            m[k] = tuple(m[k])
    return MSEIIPF(**m)


def build_loaders(
    cfg: dict, fold: int | None = None, include_test: bool = True,
    evaluation: bool = False, only_split: str | None = None,
) -> dict:
    d = cfg["data"]
    t = cfg["train"]
    kw = dict(feat_dir=d["feat_dir"], max_units=d.get("max_units", 8),
              d_visual=cfg["model"]["d_visual"], d_audio=cfg["model"]["d_audio"],
              d_text=cfg["model"]["d_text"], strict=d.get("strict_feats", False))
    if only_split is not None:
        if only_split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {only_split}")
        if fold is not None and only_split != "val":
            sets = {only_split: MERCDataset(d["manifest"], fold=fold, fold_role=only_split, **kw)}
        else:
            sets = {only_split: MERCDataset(d["manifest"], split=only_split, **kw)}
    elif fold is None:
        splits = ("train", "val", "test") if include_test else ("train", "val")
        sets = {s: MERCDataset(d["manifest"], split=s, **kw) for s in splits}
    else:
        # k 折：该折作 test，其余作 train；val 复用固定 val 划分
        sets = {
            "train": MERCDataset(d["manifest"], fold=fold, fold_role="train", **kw),
            "val": MERCDataset(d["manifest"], split="val", **kw),
        }
        if include_test:
            sets["test"] = MERCDataset(
                d["manifest"], fold=fold, fold_role="test", **kw
            )
    coll = make_collate(cfg["model"]["d_visual"], cfg["model"]["d_audio"],
                        cfg["model"]["d_text"])
    num_workers = t.get("num_workers", 0)
    loaders = {}
    for name, ds in sets.items():
        loaders[name] = DataLoader(
            ds, batch_size=t["batch_size"], shuffle=(name == "train" and not evaluation),
            num_workers=num_workers, collate_fn=coll,
            drop_last=False,
            pin_memory=False,
            persistent_workers=(num_workers > 0),
        )
    # 汇报缺失特征情况，避免静默全零训练
    for name, ds in sets.items():
        miss = {k: v for k, v in ds.missing.items() if v}
        print(f"  {name}: {len(ds)} sequences" + (f"  ⚠ 缺失特征 {miss}" if miss else ""))
    return loaders


def save_predictions(res: dict, path: str) -> None:
    raw = res.get("_raw", {})
    arrays = {k: v for k, v in raw.items() if isinstance(v, np.ndarray)}
    if arrays:
        np.savez_compressed(path, **arrays)


def run_once(cfg: dict, args: argparse.Namespace, out_dir: str,
             fold: int | None = None) -> dict:
    if os.path.exists(os.path.join(out_dir, "best.pt")) and not args.dry_run:
        raise FileExistsError(f"Run already contains a checkpoint: {out_dir}. Choose a new --exp-name.")
    os.makedirs(out_dir, exist_ok=True)
    from src.utils.artifacts import save_json, environment_info
    save_json(os.path.join(out_dir, "config.json"), cfg)
    save_json(os.path.join(out_dir, "environment.json"), environment_info())
    save_json(os.path.join(out_dir, "protocol.json"), {"fold": fold, "validation_only": args.validation_only})
    set_seed(int(cfg["train"].get("seed", 42)))
    loaders = build_loaders(cfg, fold, include_test=not args.validation_only)
    model = build_model(cfg)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  参数量 {n_param/1e6:.2f}M  device={args.device}")

    if args.dry_run:
        batch = next(iter(loaders["train"]))
        model = model.to(args.device)
        h = {k: batch[k].to(args.device) for k in
             ("h_v", "h_a", "h_sv", "h_sa", "mask_v", "mask_a", "mask_sv",
              "mask_sa", "mask_t", "labels", "seq_labels")}
        from src.engine.evaluator import _expand_seq_labels
        out, losses = model(h["h_v"], h["h_a"], h["h_sv"], h["h_sa"], h["labels"],
                            epoch=0, mask_v=h["mask_v"], mask_a=h["mask_a"],
                            mask_sv=h["mask_sv"], mask_sa=h["mask_sa"],
                            mask_t=h["mask_t"],
                            labels_for_itam=_expand_seq_labels(
                                h["seq_labels"], h["labels"].shape[1]),
                            seq_labels=h["seq_labels"])
        print("  dry-run 前向成功。各项损失：")
        for k, v in losses.items():
            if v.dim() == 0:
                print(f"    {k:10s} {float(v.detach()):.4f}")
        print(f"  输出 p_hat {tuple(out['p_hat'].shape)}  "
              f"y_pred {tuple(out['y_pred'].shape)}  "
              f"u {tuple(out['u'].shape) if out.get('u') is not None else None}")
        return {"dry_run": True, "n_param": n_param}

    trainer = Trainer(model, loaders["train"], loaders["val"], cfg,
                      args.device, out_dir)
    best = trainer.fit()
    print(f"  最优 epoch {best['epoch']}（{best['metric']}={best['score']:.4f}）")

    # 载入最优权重后评测 test
    ckpt = os.path.join(out_dir, "best.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=args.device, weights_only=True)["model"])

    validation = evaluate(model.to(args.device), loaders["val"], args.device,
                          cfg["data"]["num_classes"], cfg["data"].get("class_names"),
                          epoch=int(best["epoch"]), amp=cfg["train"].get("amp", True))
    export_predictions(validation["_raw"], out_dir, "validation")
    save_json(os.path.join(out_dir, "validation_metrics.json"), validation)
    if args.validation_only:
        print("  " + format_metrics(validation, "validation-only "))
        save_predictions(
            validation, os.path.join(out_dir, "validation_predictions.npz")
        )
        metrics = {k: v for k, v in validation.items() if k != "_raw"} | {
            "n_param": n_param,
            "selection_split": "validation",
            "best_epoch": int(best["epoch"]),
            "best_metric": str(best["metric"]),
            "best_score": float(best["score"]),
        }
        save_json(os.path.join(out_dir, "validation_metrics.json"), metrics)
        return metrics

    test = evaluate(model.to(args.device), loaders["test"], args.device,
                    cfg["data"]["num_classes"], cfg["data"].get("class_names"),
                    epoch=int(best["epoch"]), amp=cfg["train"].get("amp", True))
    print("  " + format_metrics(test, "test "))

    save_predictions(test, os.path.join(out_dir, "predictions.npz"))
    export_predictions(test["_raw"], out_dir, "test")
    metrics = {k: v for k, v in test.items() if k != "_raw"} | {"n_param": n_param}
    save_json(os.path.join(out_dir, "test_metrics.json"), metrics)
    return metrics


def main() -> int:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed

    d = cfg["data"]
    if not os.path.exists(d["manifest"]):
        print(f"[S3] manifest 不存在：{d['manifest']}，先跑 s0_scan.py", file=sys.stderr)
        return 1
    if not os.path.isdir(d["feat_dir"]) and not args.dry_run:
        print(f"[S3] 特征目录不存在：{d['feat_dir']}，先跑 s2_extract_feats.py",
              file=sys.stderr)
        return 1

    root = os.path.join(args.out_root, args.exp_name)
    from pathlib import Path
    if not args.dry_run and Path(root).exists() and any(Path(root).rglob("best.pt")):
        raise FileExistsError(f"Existing checkpoint under {root}; use a new --exp-name")
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    if not args.cv:
        print(f"[S3] 单次训练 exp={args.exp_name}")
        run_once(cfg, args, root)
        return 0

    n_folds = int(cfg.get("eval", {}).get("cv_folds", 5))
    print(f"[S3] {n_folds} 折说话人独立 CV exp={args.exp_name}")
    results = []
    for fold in range(n_folds):
        print(f"\n--- fold {fold}/{n_folds - 1} ---")
        r = run_once(cfg, args, os.path.join(root, f"fold{fold}"), fold)
        results.append(r)

    # 汇总均值 ± 标准差
    keys = [
        "acc", "macro_f1", "macro_p", "macro_r", "weighted_f1", "micro_f1",
        "seq_acc", "seq_macro_f1", "seq_weighted_f1",
    ]
    summary = {}
    for k in keys:
        vals = [float(r[k]) for r in results if k in r]
        if vals:
            summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                          "folds": vals}
    print("\n[S3] CV 汇总（mean ± std）:")
    for k, v in summary.items():
        print(f"  {k:12s} {v['mean']:.4f} ± {v['std']:.4f}")
    summary_name = (
        "validation_cv_summary.json" if args.validation_only else "cv_summary.json"
    )
    payload = {
        "protocol": "validation_only" if args.validation_only else "test_evaluation",
        "test_split_accessed": not args.validation_only,
        "summary": summary,
        "folds": results,
    }
    with open(os.path.join(root, summary_name), "w", encoding="utf-8") as f:
        json.dump(payload, f,
                  ensure_ascii=False, indent=2, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
