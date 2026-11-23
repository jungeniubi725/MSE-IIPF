"""S0：扫描原始数据集，生成 manifest.csv。

用法：
    python scripts/s0_scan.py --dataset ravdess --root data/ravdess \
        --out cache/ravdess/manifest.csv --num-segments 4 --probe-duration


产物 manifest.csv 一行一个 unit（时间单元），列见 schema.Sequence.to_rows()。
后续 S1/S2/S3 全部以该文件为唯一入口，不再直接遍历数据目录。
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import ravdess  # noqa: F401  仅为触发 @register
from src.data.schema import available, get_adapter
from src.utils.console import setup_console
from src.utils.media import probe_duration

setup_console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="扫描数据集生成 manifest")
    p.add_argument("--dataset", required=True, choices=available())
    p.add_argument("--root", required=True, help="数据集根目录")
    p.add_argument("--out", required=True, help="输出 manifest.csv 路径")
    p.add_argument("--num-segments", type=int, default=4,
                   help="RAVDESS 单段视频切分的片段数 K（论文 E33 的 t）")
    p.add_argument("--vocal-channel", default="both", choices=["both", "speech", "song"],
                   help="RAVDESS 声道筛选")
    p.add_argument("--label-mode", default="regression",
                   choices=["regression", "neg2", "neg3", "seven"],
                   help="MOSI/MOSEI 标签离散化协议")
    p.add_argument("--probe-duration", action="store_true",
                   help="用 ffprobe 探测时长（RAVDESS 切片段必需，2452 个文件约 2-4 分钟）")
    p.add_argument("--limit", type=int, default=0, help="只处理前 N 个样本，调试用")
    return p.parse_args()


def build_ravdess(args: argparse.Namespace):
    """RAVDESS 需先探时长才能切片段，故分两趟：先扫文件，再探时长重建。"""
    Adapter = get_adapter("ravdess")
    durations: dict[str, float] = {}
    if args.probe_duration:
        probe = Adapter(args.root, num_segments=1, vocal_channel=args.vocal_channel)
        seqs = probe.build_sequences()
        if args.limit:
            seqs = seqs[: args.limit]
        print(f"[S0] 探测 {len(seqs)} 个文件时长 …", flush=True)
        for i, s in enumerate(seqs, 1):
            d = probe_duration(s.units[0].video_path)
            if d:
                durations[s.seq_id] = d
            if i % 200 == 0:
                print(f"      {i}/{len(seqs)}", flush=True)
        ok = len(durations)
        if ok != len(seqs):
            raise RuntimeError("Duration probing failed for some inputs; check FFmpeg/ffprobe before creating temporal units")
        print(f"[S0] 时长探测成功 {ok}/{len(seqs)}"
              + ("" if ok == len(seqs) else "  ⚠ 失败者退化为整段单 unit"))
    else:
        print("[S0] ⚠ 未开 --probe-duration，每个样本只会有 1 个 unit（整段），"
              "IIPF 的时序建模 (L_TPC) 将退化。")

    adapter = Adapter(
        args.root,
        num_segments=args.num_segments,
        vocal_channel=args.vocal_channel,
        durations=durations,
    )
    seqs = adapter.build()
    if args.limit:
        seqs = seqs[: args.limit]
    return adapter, seqs


def main() -> int:
    args = parse_args()
    if not os.path.isdir(args.root):
        print(f"[S0] 数据根目录不存在：{args.root}", file=sys.stderr)
        return 1

    if args.dataset == "ravdess":
        adapter, seqs = build_ravdess(args)
    else:
        Adapter = get_adapter(args.dataset)
        adapter = Adapter(args.root, label_mode=args.label_mode)
        seqs = adapter.build()
        if args.limit:
            seqs = seqs[: args.limit]

    if not seqs:
        print("[S0] 未扫到任何样本，检查 --root", file=sys.stderr)
        return 1

    rows = [r for s in seqs for r in s.to_rows()]
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")

    # ---- 摘要 ----
    print(f"\n[S0] {adapter.name}: {len(seqs)} sequences / {len(df)} units -> {args.out}")
    print(f"     task={adapter.task}  num_classes={adapter.num_classes}")
    print("\n     split 分布（sequence 级）:")
    sp = pd.Series([s.split for s in seqs]).value_counts()
    for k, v in sp.items():
        print(f"       {k:6s} {v}")
    if adapter.task == "classification" and adapter.class_names:
        print("\n     类别分布（sequence 级）:")
        lab = pd.Series([s.label for s in seqs]).value_counts().sort_index()
        for k, v in lab.items():
            name = adapter.class_names[int(k)] if int(k) < len(adapter.class_names) else str(k)
            print(f"       {int(k)} {name:10s} {v}")
    nu = pd.Series([s.num_units for s in seqs])
    print(f"\n     每 sequence 的 unit 数: min={nu.min()} max={nu.max()} mean={nu.mean():.2f}")
    miss_a = int(df["audio_path"].isna().sum())
    if miss_a:
        print(f"     ⚠ 缺失音频的 unit: {miss_a}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
