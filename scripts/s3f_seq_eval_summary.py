"""S3f：对 CV 主结果逐折做 sequence 级评估，并汇总成表。

背景
----
论文主表 T1 是 5 折 CV 口径（unit 级聚合）。用户要求补「一个样本对应一个预测」
的 sequence 级口径：一个视频（sequence）由 T 个片段（unit）组成，把 T 个证据
仅累加 α_t−1 的证据并只加入一次单位先验，得到
α_seq=1+Σ_t(α_t−1)，再取 p_seq=α_seq/Σα_seq 作样本级预测。

s3e_seq_eval.py 已支持 --seed/--fold 逐折评估；本脚本循环调用它，收集每个折的
seq_metrics.json，汇总 mean±std 写成表（CSV + LaTeX）。

用法：
    python scripts/s3f_seq_eval_summary.py --exp-name main --seeds 42 43 44 --folds 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 汇总的 sequence 级指标（seq_metrics.json 顶层或 uncertainty 子字段）
TOP_KEYS = ["acc", "macro_p", "macro_r", "macro_f1", "micro_f1", "weighted_f1",
            "probmean_acc", "probmean_macro_f1"]
UNC_KEYS = ["ece", "mce", "avg_conf", "avg_acc", "auroc_u", "avg_u"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S3f 逐折 sequence 级评估并汇总")
    p.add_argument("--config", default="configs/ravdess.yaml")
    p.add_argument("--exp-name", required=True)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--split", default="test")
    p.add_argument("--out-dir", default=None)
    p.add_argument(
        "--fold-output-dir-name", default="seq_eval",
        help="每个 seed/fold 下的预测目录名，如 seq_eval_val 或 seq_eval_test",
    )
    return p.parse_args()


def run_fold(
    exp: str, seed: int, fold: int, cfg: str, split: str,
    fold_output_dir_name: str = "seq_eval",
) -> str | None:
    """跑单折 s3e，返回 seq_metrics.json 路径；失败返回 None。"""
    cmd = [sys.executable, os.path.join("scripts", "s3e_seq_eval.py"),
           "--config", cfg, "--exp-name", exp, "--split", split,
           "--seed", str(seed), "--fold", str(fold),
           "--out-dir", os.path.join(
               "runs", exp, f"seed{seed}", f"fold{fold}",
               fold_output_dir_name,
           )]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    mpath = os.path.join(ROOT, "runs", exp, f"seed{seed}", f"fold{fold}",
                         fold_output_dir_name, "seq_metrics.json")
    if proc.returncode != 0 or not os.path.exists(mpath):
        print(f"  ✗ seed{seed}/fold{fold} rc={proc.returncode}\n"
              f"    {(proc.stderr or proc.stdout)[-400:]}")
        return None
    return mpath


def collect(metric: dict) -> dict:
    unc = metric.get("uncertainty", {})
    out = {k: metric.get(k, float("nan")) for k in TOP_KEYS}
    out.update({k: unc.get(k, float("nan")) for k in UNC_KEYS})
    out["n_sequences"] = metric.get("n_sequences", 0)
    return out


def main() -> int:
    args = parse_args()
    exp = args.exp_name
    cfg_path = args.config if os.path.isabs(args.config) \
        else os.path.join(ROOT, args.config)
    if not os.path.exists(cfg_path):
        print(f"[S3f] 配置不存在：{cfg_path}", file=sys.stderr)
        return 1

    if args.seeds is None:
        from src.utils.config import load_config
        args.seeds = load_config(cfg_path).get("eval", {}).get("seeds", [42])

    per_fold: list[dict] = []
    for seed in args.seeds:
        for fold in range(args.folds):
            print(f"[S3f] seed{seed}/fold{fold} ...")
            mpath = run_fold(
                exp, seed, fold, args.config, args.split,
                args.fold_output_dir_name,
            )
            if mpath is None:
                continue
            with open(mpath, encoding="utf-8") as f:
                m = json.load(f)
            row = {"seed": seed, "fold": fold, **collect(m)}
            per_fold.append(row)

    if not per_fold:
        print("[S3f] 无有效折结果，先跑 s4 --groups main 或 s3_train --cv")
        return 1

    # ---- 汇总（先按 seed 内 fold 平均，再跨 seed 平均，与 T1 口径一致）----
    num_keys = TOP_KEYS + UNC_KEYS

    def mean_std(vals: list[float]) -> tuple[float, float]:
        vals = [float(v) for v in vals if v == v]  # 去掉 nan
        if not vals:
            return float("nan"), float("nan")
        import statistics
        return statistics.fmean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)

    # 每个 seed：该 seed 下所有 fold 的 mean±std
    seed_summary: dict[int, dict] = {}
    for seed in args.seeds:
        rows = [r for r in per_fold if r["seed"] == seed]
        sm = {"n_folds": len(rows), "n_sequences": rows[0]["n_sequences"] if rows else 0}
        for k in num_keys:
            vals = [r[k] for r in rows]
            mn, sd = mean_std(vals)
            sm[k] = mn
            sm[f"{k}_std"] = sd
        seed_summary[seed] = sm

    # 总体：先每 seed 平均，再跨 seed 平均（报告 mean±std）
    overall = {"n_folds_per_seed": args.folds}
    for k in num_keys:
        vals = [seed_summary[s][k] for s in args.seeds]
        mn, sd = mean_std(vals)
        overall[k] = mn
        overall[f"{k}_std"] = sd

    # ---- 落盘 ----
    out_dir = args.out_dir or os.path.join(ROOT, "runs", exp, "seq_eval_cv")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "seq_cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"per_fold": per_fold, "per_seed": seed_summary,
                   "overall": overall}, f, ensure_ascii=False, indent=2, default=float)

    # 样本数：各折 test 集大小不同（416/372），报总数而非 mean±std
    n_seq_total = sum(int(r["n_sequences"]) for r in per_fold)
    overall["n_sequences"] = float(n_seq_total)
    overall["n_sequences_std"] = float("nan")

    # 主表：overall 的 mean ± std（sequence 级口径）
    main_rows = [("n_sequences", "样本数(总)"),
                 ("acc", "Accuracy"),
                 ("macro_f1", "Macro-F1"),
                 ("weighted_f1", "Weighted-F1"),
                 ("ece", "ECE"),
                 ("avg_conf", "Avg Conf"),
                 ("avg_u", "Avg u"),
                 ("auroc_u", "AUROC(u)"),
                 ("probmean_acc", "ProbMean Acc")]

    def _fmt(key: str) -> str:
        if key == "n_sequences":
            return f"{int(overall[key])}"
        sd = overall.get(key + "_std", float("nan"))
        if sd != sd:  # nan
            return f"{overall[key]:.4f}"
        return f"{overall[key]:.4f} ± {sd:.4f}"

    csv_path = os.path.join(out_dir, "seq_cv_table.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "label", "mean ± std"])
        for key, label in main_rows:
            w.writerow([key, label, _fmt(key)])
    # LaTeX
    tex_path = os.path.join(out_dir, "seq_cv_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% sequence 级口径（证据聚合 α_seq=1+Σ_t(α_t−1)），CV mean±std\n")
        f.write("\\begin{tabular}{lcc}\n\\toprule\n指标 & 均值 & 标准差 \\\\\n\\midrule\n")
        for key, label in main_rows:
            if key == "n_sequences":
                f.write(f"{label} & {int(overall[key])} & — \\\\\n")
                continue
            f.write(f"{label} & {overall[key]:.4f} & "
                    f"{overall.get(key + '_std', float('nan')):.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    print("\n[S3f] sequence 级口径（CV mean±std，证据聚合）：")
    for key, label in main_rows:
        print(f"      {label:14s} {_fmt(key)}")
    print(f"[S3f] 完成 -> {out_dir}（seq_cv_summary.json / seq_cv_table.csv / .tex）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
