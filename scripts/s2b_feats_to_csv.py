#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 S2 提取的 .npy 特征导出为 CSV。

特征本身是变长序列 (n, 768)：视觉 n≈8 个片段、音频 n≈41 帧、文本 n≈10-180 token。
CSV 需要固定列数，故按时间维做均值池化，每个单元出一行 768 维向量。

  visual/audio : unit 级，9808 行 × (id + 768)
  text_sv/sa   : sequence 级，2452 行 × (id + 768)

注意：池化后丢失时间维，仅供查看与外部分析。训练仍读原始 .npy —— IIPF
的时序建模需要完整序列，不能用这里的 CSV 回灌。

用法：
    python scripts/s2b_feats_to_csv.py --feats-dir cache/ravdess/feats \
        --out-dir cache/ravdess/feats_csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

# 各模态在 feats-dir 下的子目录名
BRANCHES = ("visual", "audio", "text_sv", "text_sa")


def unsafe_name(fname: str) -> str:
    """safe_name 的逆运算，还原原始 id。"""
    return fname.replace(".npy", "").replace("__", "/").replace("-at-", "#")


def resolve_branch_dir(feats_dir: str, br: str) -> str | None:
    """定位分支的 .npy 所在目录。s2 各模态的落盘层级不一致：
    visual/audio 是 <feats>/<m>/<m>/，文本两支是 <feats>/text/<branch>/，
    另外兼容直接就是 <feats>/<branch>/ 的扁平布局。取第一个含 .npy 的候选。"""
    cands = [
        os.path.join(feats_dir, br, br),          # visual/visual, audio/audio
        os.path.join(feats_dir, "text", br),      # text/text_sv, text/text_sa
        os.path.join(feats_dir, br),              # 扁平
    ]
    for c in cands:
        if os.path.isdir(c) and any(f.endswith(".npy") for f in os.listdir(c)):
            return c
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--feats-dir", default="cache/ravdess/feats")
    p.add_argument("--out-dir", default="cache/ravdess/feats_csv")
    p.add_argument("--branches", nargs="*", default=list(BRANCHES),
                   help=f"要导出的分支，默认全部：{' '.join(BRANCHES)}")
    p.add_argument("--pool", default="mean", choices=["mean", "first", "max"],
                   help="时间维池化方式，默认 mean")
    p.add_argument("--float-fmt", default="%.6g",
                   help="浮点格式，默认 %%.6g（约省一半体积且精度足够）")
    return p.parse_args()


def pool_one(a: np.ndarray, how: str) -> np.ndarray:
    """(n, d) -> (d,)。空数组返回全零。"""
    if a.ndim == 1:
        return a
    if a.shape[0] == 0:
        return np.zeros(a.shape[-1], dtype=np.float32)
    if how == "mean":
        return a.mean(axis=0)
    if how == "max":
        return a.max(axis=0)
    return a[0]


def export_branch(src: str, dst_csv: str, how: str, fmt: str) -> dict:
    files = sorted(f for f in os.listdir(src) if f.endswith(".npy"))
    if not files:
        return {"branch": os.path.basename(src), "n": 0, "note": "无 .npy，跳过"}

    dim = pool_one(np.load(os.path.join(src, files[0])), how).shape[0]
    n_zero, lens = 0, []
    with open(dst_csv, "w", encoding="utf-8-sig", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["id"] + [f"f{i}" for i in range(dim)])
        for i, fn in enumerate(files, 1):
            a = np.load(os.path.join(src, fn))
            lens.append(a.shape[0] if a.ndim > 1 else 1)
            v = pool_one(a, how)
            if not np.any(v):
                n_zero += 1
            w.writerow([unsafe_name(fn)] + [fmt % x for x in v])
            if i % 2000 == 0:
                print(f"      {i}/{len(files)}", flush=True)
    return {
        "branch": os.path.basename(src),
        "n": len(files),
        "dim": dim,
        "seq_len_min": int(min(lens)),
        "seq_len_max": int(max(lens)),
        "seq_len_mean": round(float(np.mean(lens)), 1),
        "all_zero_rows": n_zero,
        "csv": dst_csv,
        "size_mb": round(os.path.getsize(dst_csv) / 2**20, 1),
    }


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[S2b] 池化方式={args.pool}  源={args.feats_dir}")

    reports = []
    for br in args.branches:
        src = resolve_branch_dir(args.feats_dir, br)
        if src is None:
            print(f"[S2b] {br:9} 未找到 .npy，跳过（查过 {br}/{br}、text/{br}、{br}）")
            continue
        dst = os.path.join(args.out_dir, f"{br}.csv")
        print(f"[S2b] {br:9} 导出中 -> {dst}")
        rep = export_branch(src, dst, args.pool, args.float_fmt)
        reports.append(rep)
        if rep["n"] == 0:
            print(f"      {rep['note']}")
        else:
            print(f"      {rep['n']} 行 × {rep['dim']} 维  "
                  f"序列长度 {rep['seq_len_min']}–{rep['seq_len_max']}"
                  f"（均 {rep['seq_len_mean']}）  "
                  f"全零行 {rep['all_zero_rows']}  {rep['size_mb']} MB")

    if not reports:
        print("[S2b] 没有任何分支可导出，先跑 s2_extract_feats.py", file=sys.stderr)
        return 1

    # 汇总表，便于核对
    idx = os.path.join(args.out_dir, "_index.csv")
    keys = ["branch", "n", "dim", "seq_len_min", "seq_len_max", "seq_len_mean",
            "all_zero_rows", "size_mb", "csv"]
    with open(idx, "w", encoding="utf-8-sig", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in reports:
            if r["n"]:
                w.writerow(r)
    print(f"[S2b] 完成，{len(reports)} 个分支 -> {args.out_dir}（汇总 {idx}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
