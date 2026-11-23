"""特征缓存 -> batch。读 S2 落盘的 .npy，按 Sequence 组装 (B, T, ·)。

关键点
-----
1. 视觉/音频特征是 **unit 级**（每片段一个 .npy），文本是 **sequence 级**
   （见 s2 的粒度设计，依据论文 E18）。
2. 同一 batch 内 T（时间单元数）、n_v、n_a、n_sv、n_sa 都可能不同，
   collate 时统一 padding 并给 mask，供 SG-DSFE 的 MHA 与注意力池化屏蔽。
3. RAVDESS 标签在 sequence 级 -> 广播到所有 unit；MOSI/MOSEI 在 unit 级。
"""

from __future__ import annotations

import os
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def safe_name(uid: str) -> str:
    """与 scripts/s2_extract_feats.py 的命名保持一致。"""
    return uid.replace("/", "__").replace("#", "-at-").replace("\\", "__")


def resolve_branch_dir(feat_dir: str, branch: str) -> str:
    """定位分支 .npy 的真实目录。s2 各模态落盘层级不一致：
    visual/audio 在 <feat>/<b>/<b>/，两支文本在 <feat>/text/<b>/，
    也兼容扁平的 <feat>/<b>/。取第一个含 .npy 的候选；都没有则返回扁平路径，
    由 _load 走 strict/missing 逻辑报出来。"""
    cands = [
        os.path.join(feat_dir, branch, branch),
        os.path.join(feat_dir, "text", branch),
        os.path.join(feat_dir, branch),
    ]
    for c in cands:
        if os.path.isdir(c) and any(f.endswith(".npy") for f in os.listdir(c)):
            return c
    return os.path.join(feat_dir, branch)


class MERCDataset(Dataset):
    """一个样本 = 一个 Sequence（含 T 个 unit）。"""

    def __init__(
        self,
        manifest: str | pd.DataFrame,
        feat_dir: str,
        split: str | None = None,
        fold: int | None = None,
        fold_role: str = "test",  # fold 指定时该折扮演的角色
        max_units: int = 8,
        d_visual: int = 768,
        d_audio: int = 768,
        d_text: int = 768,
        strict: bool = False,  # True 时缺特征直接报错，False 时置零并计数
    ) -> None:
        df = pd.read_csv(manifest) if isinstance(manifest, str) else manifest.copy()
        if fold is not None:
            # k 折 CV：仅 fold 非空的演员（RAVDESS Actor 01-20）参与
            # outer CV。fold 为空的固定 val/test 演员必须排除在 CV train
            # 之外，否则 Actor 21-22 会同时出现在 train 和 validation，
            # 导致 checkpoint 选择及后验校准泄漏。
            in_fold = df["fold"] == fold
            if fold_role in ("test", "val"):
                df = df[in_fold]
            elif fold_role == "train":
                df = df[df["fold"].notna() & ~in_fold]
            else:
                raise ValueError(f"未知 fold_role={fold_role!r}")
        elif split is not None:
            df = df[df["split"] == split]
        if df.empty:
            raise ValueError(f"筛选后无样本：split={split} fold={fold}")

        self.feat_dir = feat_dir
        self.max_units = max_units
        self.dims = {"v": d_visual, "a": d_audio, "t": d_text}
        self.strict = strict
        self.missing = {"visual": 0, "audio": 0, "text_sv": 0, "text_sa": 0}
        # 各分支目录只解析一次，避免每次 __getitem__ 都 listdir
        self.branch_dirs = {b: resolve_branch_dir(feat_dir, b)
                            for b in ("visual", "audio", "text_sv", "text_sa")}

        # 按 seq_id 分组，unit 按 unit_index 排序
        df = df.sort_values(["seq_id", "unit_index"])
        self.groups: list[dict] = []
        for seq_id, grp in df.groupby("seq_id", sort=True):
            g = grp.head(max_units)
            self.groups.append({
                "seq_id": seq_id,
                "unit_ids": g["unit_id"].tolist(),
                "labels": [int(x) for x in g["label"].tolist()],
                "seq_label": int(g["seq_label"].iloc[0])
                if not pd.isna(g["seq_label"].iloc[0]) else int(g["label"].iloc[0]),
                "speaker": g["speaker"].iloc[0],
            })

    def __len__(self) -> int:
        return len(self.groups)

    def _load(self, branch: str, key: str, dim: int) -> np.ndarray:
        p = os.path.join(self.branch_dirs[branch], safe_name(key) + ".npy")
        if not os.path.exists(p):
            if self.strict:
                raise FileNotFoundError(f"缺失特征：{p}\n先跑 s2_extract_feats.py")
            self.missing[branch] = self.missing.get(branch, 0) + 1
            return np.zeros((1, dim), dtype=np.float32)
        values = np.load(p, allow_pickle=False).astype(np.float32)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != dim or not np.isfinite(values).all():
            raise ValueError(f"Invalid feature array {p}: {values.shape}")
        return values

    def __getitem__(self, i: int) -> dict:
        g = self.groups[i]
        # 视觉/音频：逐 unit 读，按 unit 拼成 list
        vis = [self._load("visual", u, self.dims["v"]) for u in g["unit_ids"]]
        aud = [self._load("audio", u, self.dims["a"]) for u in g["unit_ids"]]
        # 文本：sequence 级，一个 sequence 一份
        sv = self._load("text_sv", g["seq_id"], self.dims["t"])
        sa = self._load("text_sa", g["seq_id"], self.dims["t"])
        return {
            "seq_id": g["seq_id"],
            "unit_ids": g["unit_ids"],
            "visual": vis,  # list[T] of (n_v, d)
            "audio": aud,  # list[T] of (n_a, d)
            "text_sv": sv,  # (n_sv, d)
            "text_sa": sa,  # (n_sa, d)
            "labels": np.asarray(g["labels"], dtype=np.int64),  # (T,)
            "seq_label": g["seq_label"],
            "speaker": g["speaker"],
        }


def _pad_stack(arrs: list[np.ndarray], dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """把变长 (n_i, d) 列表 pad 成 (B, max_n, d) + mask (B, max_n)。"""
    n_max = max((a.shape[0] for a in arrs), default=1)
    B = len(arrs)
    out = np.zeros((B, n_max, dim), dtype=np.float32)
    mask = np.zeros((B, n_max), dtype=np.float32)
    for i, a in enumerate(arrs):
        n = a.shape[0]
        out[i, :n] = a
        mask[i, :n] = 1.0
    return torch.from_numpy(out), torch.from_numpy(mask)


def collate_fn(batch: list[dict], d_visual: int = 768, d_audio: int = 768,
               d_text: int = 768) -> dict:
    """把 Sequence 列表拼成 batch。

    视觉/音频先在 unit 维展平做 padding，再 reshape 回 (B, T, N, D)？
    —— 不。模型的 SG-DSFE 按 (B, N, D) 处理，IIPF 才按 (B, T, D)。
    故这里把每个 unit 视为 SG-DSFE 的一个「样本」：
        B_eff = B × T，SG-DSFE/TMA 在 B_eff 上跑，
        IIPF 前再 reshape 回 (B, T, ·)。
    unit 数不足 T_max 的样本用 mask_t 屏蔽。
    """
    B = len(batch)
    T_max = max(len(b["visual"]) for b in batch)

    vis_flat: list[np.ndarray] = []
    aud_flat: list[np.ndarray] = []
    mask_t = np.zeros((B, T_max), dtype=np.float32)
    labels = np.zeros((B, T_max), dtype=np.int64)
    for i, b in enumerate(batch):
        T = len(b["visual"])
        mask_t[i, :T] = 1.0
        labels[i, :T] = b["labels"][:T]
        # 不足 T_max 的位置补零特征（会被 mask_t 屏蔽）
        for t in range(T_max):
            if t < T:
                vis_flat.append(b["visual"][t])
                aud_flat.append(b["audio"][t])
            else:
                vis_flat.append(np.zeros((1, d_visual), dtype=np.float32))
                aud_flat.append(np.zeros((1, d_audio), dtype=np.float32))
                labels[i, t] = b["labels"][T - 1]  # 占位，被 mask 屏蔽

    h_v, mask_v = _pad_stack(vis_flat, d_visual)  # (B*T_max, n_v, d)
    h_a, mask_a = _pad_stack(aud_flat, d_audio)
    # 文本 sequence 级 -> 需广播到 B*T_max 与视觉/音频对齐
    sv_rep = [b["text_sv"] for b in batch for _ in range(T_max)]
    sa_rep = [b["text_sa"] for b in batch for _ in range(T_max)]
    h_sv, mask_sv = _pad_stack(sv_rep, d_text)
    h_sa, mask_sa = _pad_stack(sa_rep, d_text)

    return {
        "h_v": h_v, "h_a": h_a, "h_sv": h_sv, "h_sa": h_sa,
        "mask_v": mask_v, "mask_a": mask_a, "mask_sv": mask_sv, "mask_sa": mask_sa,
        "mask_t": torch.from_numpy(mask_t),
        "labels": torch.from_numpy(labels),  # (B, T_max)
        "seq_labels": torch.tensor([b["seq_label"] for b in batch], dtype=torch.long),
        "seq_ids": [b["seq_id"] for b in batch],
        "unit_ids": [u for b in batch for u in b.get("unit_ids", [f"{b['seq_id']}#{i}" for i in range(len(b["visual"]))])],
        "speakers": [b.get("speaker", "") for b in batch],
        "B": B, "T": T_max,
    }


def make_collate(d_visual: int = 768, d_audio: int = 768, d_text: int = 768):
    """DataLoader 需要可 pickle 的 collate。

    Windows 下 num_workers>0 用 spawn，闭包（`make_collate.<locals>._fn`）无法
    通过模块路径被 worker 重建，会报 ``Can't get local object``。改用
    ``functools.partial`` 绑定模块级 ``collate_fn``，参数（int）可 pickle。
    """
    return partial(collate_fn, d_visual=d_visual, d_audio=d_audio, d_text=d_text)
