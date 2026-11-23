"""S3c：可视化 TMA 对齐效果。

读 S3b 提取的对齐后特征（runs/<exp>/tma_feats/aligned*.npz）与原始 .npy，
产出三张图，回答「TMA 是否把三模态对齐到统一情感语义空间」：

  1. alignment_consistency.png   对齐前(原始768d) vs 对齐后(g,256d) 的样本内
                                三模态余弦一致性（柱状图）。
  2. umap_modalities.png         对齐后三模态在 UMAP 2D 的分布，按模态着色。
  3. umap_by_class.png           对齐后 sequence 级融合表示按 8 类情感分面。

配色遵循 dataviz 默认调色板：三模态用 categorical slot 1-3（blue/orange/aqua，
散点 all-pairs 通过 CVD 校验）；8 类用「单色高亮 + 灰背景」分面而非 8 色，
避免散点图 >3 类时色盲区分度不足。

用法：
    python scripts/s3c_viz_tma.py --config configs/ravdess.yaml --exp-name main
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # 无显示环境，直接写文件
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import resolve_branch_dir, safe_name  # noqa: E402

# ---- dataviz 默认调色板（light）----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
C_MOD = {"v": "#2a78d6", "a": "#eb6834", "l": "#1baf7a"}   # slot 1-3
C_BEFORE = "#898781"      # 对齐前（raw）—— 灰
C_AFTER = "#2a78d6"       # 对齐后 —— 蓝

MOD_NAMES = {"v": "Visual", "a": "Audio", "l": "Language"}
PAIRS = [("v", "a"), ("v", "l"), ("a", "l")]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S3c 可视化 TMA 对齐效果")
    p.add_argument("--config", required=True)
    p.add_argument("--exp-name", required=True)
    p.add_argument("--feats-dir", default="cache/ravdess/feats")
    p.add_argument("--manifest", default="cache/ravdess/manifest.csv")
    p.add_argument("--out-dir", default=None, help="默认 runs/<exp-name>/tma_feats")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def load_aligned(feat_dir: str) -> dict:
    u = np.load(os.path.join(feat_dir, "aligned.npz"))
    s = np.load(os.path.join(feat_dir, "aligned_seq.npz"))
    return {"g_v": u["g_v"], "g_a": u["g_a"], "g_l": u["g_l"],
            "label": u["label"], "seq_idx": u["seq_idx"],
            "sv": s["g_v"], "sa": s["g_a"], "sl": s["g_l"],
            "seq_label": s["seq_label"]}


def raw_unit_features(args: argparse.Namespace) -> dict:
    """对齐前的样本内三模态余弦一致性（unit 级）。

    文本分两支：text_sv（视觉描述 D_v）对齐视觉，text_sa（音频转录 D_a）
    对齐音频 —— 语义上一一对应。全部 mean-pool 到 768d。
    """
    df = pd.read_csv(args.manifest)
    df = df[df["split"] == "test"]
    bd = {b: resolve_branch_dir(args.feats_dir, b)
          for b in ("visual", "audio", "text_sv", "text_sa")}

    def _pool(path, dim=768):
        a = np.load(path).astype(np.float32)
        return a.mean(0) if a.ndim > 1 else a

    out = {p: [] for p in PAIRS}
    for r in df.itertuples():
        uid, sid = r.unit_id, r.seq_id
        try:
            v = _pool(os.path.join(bd["visual"], safe_name(uid) + ".npy"))
            a = _pool(os.path.join(bd["audio"], safe_name(uid) + ".npy"))
            sv = _pool(os.path.join(bd["text_sv"], safe_name(sid) + ".npy"))
            sa = _pool(os.path.join(bd["text_sa"], safe_name(sid) + ".npy"))
        except FileNotFoundError:
            continue
        vecs = {"v": v, "a": a, "l": sv}   # v-l 用 D_v
        out[("v", "a")].append(_cos(v, a))
        out[("v", "l")].append(_cos(v, sv))
        out[("a", "l")].append(_cos(a, sa))
    return {p: float(np.mean(x)) for p, x in out.items()}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def _mean_cos(X: np.ndarray, Y: np.ndarray) -> float:
    """按行归一化后，逐行点积的均值（样本内一致性）。"""
    X = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-8)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True).clip(1e-8)
    return float((X * Y).sum(1).mean())


def _setup_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig1_consistency(raw: dict, aligned: dict, out_dir: str, dpi: int):
    labels = ["Visual–Audio", "Visual–Language", "Audio–Language"]
    before = [raw[p] for p in PAIRS]
    after = [aligned[p] for p in PAIRS]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    x = np.arange(len(labels))
    w = 0.36
    b1 = ax.bar(x - w / 2, before, w, color=C_BEFORE, label="Before alignment (raw 768d)")
    b2 = ax.bar(x + w / 2, after, w, color=C_AFTER, label="After alignment (g, 256d)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=INK, fontsize=10)
    ax.set_ylabel("Mean sample-wise cosine similarity", color=INK2, fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    # 直接标注数值（只有 6 根柱，全标）
    for b in (b1, b2):
        for rect in b:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02,
                    f"{rect.get_height():.3f}", ha="center", va="bottom",
                    color=INK2, fontsize=8)
    _setup_ax(ax)
    ax.set_title("Tri-modal consistency in the aligned affective space",
                 color=INK, fontsize=11, loc="left", pad=12)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(out_dir, f"alignment_consistency.{ext}"),
                    dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return dict(zip(labels, after))


def fig2_umap_modalities(aligned: dict, out_dir: str, dpi: int):
    import umap
    g = {"v": aligned["g_v"], "a": aligned["g_a"], "l": aligned["g_l"]}
    X = np.vstack([g[m] for m in ("v", "a", "l")])
    mods = np.concatenate([[m] * g[m].shape[0] for m in ("v", "a", "l")])
    red = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42,
                    metric="cosine").fit_transform(X)

    fig, ax = plt.subplots(figsize=(6.6, 6.0), facecolor=SURFACE)
    for m in ("v", "a", "l"):
        sel = mods == m
        ax.scatter(red[sel, 0], red[sel, 1], s=8, c=C_MOD[m], label=MOD_NAMES[m],
                   alpha=0.7, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(BASELINE)
    ax.legend(frameon=False, loc="best", fontsize=10,
              markerscale=2.4, handletextpad=0.3)
    ax.set_title("UMAP of TMA-aligned features, colored by modality",
                 color=INK, fontsize=11, loc="left", pad=12)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(out_dir, f"umap_modalities.{ext}"),
                    dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def fig3_umap_by_class(aligned: dict, class_names: list[str], out_dir: str, dpi: int):
    import umap
    # sequence 级：三模态 g 取 mean 作为融合表示
    X = (aligned["sv"] + aligned["sa"] + aligned["sl"]) / 3.0
    y = aligned["seq_label"]
    red = umap.UMAP(n_neighbors=15, min_dist=0.15, random_state=42,
                    metric="cosine").fit_transform(X)

    n = len(class_names)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 3.0),
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    for i, name in enumerate(class_names):
        ax = axes[i]
        ax.set_facecolor(SURFACE)
        others = red[y != i]
        ax.scatter(others[:, 0], others[:, 1], s=6, c="#d8d7d2", alpha=0.5,
                   linewidths=0)
        ax.scatter(red[y == i, 0], red[y == i, 1], s=12, c=C_MOD["v"],
                   linewidths=0, label=name)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(BASELINE)
        ax.set_title(name, color=INK, fontsize=9.5, loc="left", pad=6)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("UMAP of sequence-level aligned features, one panel per emotion",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(out_dir, f"umap_by_class.{ext}"),
                    dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    feat_dir = args.out_dir or os.path.join("runs", args.exp_name, "tma_feats")
    if not os.path.exists(os.path.join(feat_dir, "aligned.npz")):
        print(f"[S3c] 无 aligned.npz：{feat_dir}，先跑 s3b_extract_tma.py", file=sys.stderr)
        return 1
    with open(os.path.join(feat_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    class_names = meta.get("class_names") or [str(i) for i in range(meta["n_class"])]

    aligned = load_aligned(feat_dir)
    # 对齐后样本内一致性（unit 级）
    aligned_cos = {"v": aligned["g_v"], "a": aligned["g_a"], "l": aligned["g_l"]}
    aligned_cons = {p: _mean_cos(aligned_cos[p[0]], aligned_cos[p[1]]) for p in PAIRS}

    print(f"[S3c] 对齐后（unit {aligned['g_v'].shape[0]}）样本内一致性：")
    for p in PAIRS:
        print(f"      {MOD_NAMES[p[0]]}-{MOD_NAMES[p[1]]}: {aligned_cons[p]:+.4f}")

    raw = raw_unit_features(args)
    print(f"[S3c] 对齐前（raw 768d）样本内一致性：")
    for p in PAIRS:
        print(f"      {MOD_NAMES[p[0]]}-{MOD_NAMES[p[1]]}: {raw[p]:+.4f}")

    fig1_consistency(raw, aligned_cons, feat_dir, args.dpi)
    fig2_umap_modalities(aligned, feat_dir, args.dpi)
    fig3_umap_by_class(aligned, class_names, feat_dir, args.dpi)

    print(f"[S3c] 完成 -> {feat_dir}")
    for f in ("alignment_consistency.png", "umap_modalities.png", "umap_by_class.png"):
        p = os.path.join(feat_dir, f)
        print(f"      {f}  {os.path.getsize(p)/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
