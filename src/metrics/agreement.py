"""Krippendorff's α —— 模态贡献分析（experiment.txt 指标 4）。

`krippendorff` 包本机未装，此处**自实现名义尺度（nominal）α**，
并附已知算例的单元测试（见文件末 _selftest）。

定义
----
    α = 1 − D_o / D_e

名义尺度下用**重合矩阵（coincidence matrix）**计算：
    D_o = Σ_{c≠k} o_ck / n_total          观察不一致度
    D_e = Σ_{c≠k} n_c·n_k / (n_total − 1)  期望不一致度（按边缘分布随机配对）

其中 o_ck 是重合矩阵元素，n_c 是类别 c 的边缘计数。
支持缺失值（np.nan 或 -1），只在每个单元内有 ≥2 个有效评分时纳入计算。

本项目用法
---------
把三个单模态辅助头（p_v, p_a, p_l）与融合头的**预测标签**视为 4 个「评分者」，
每个样本是一个「单元」，衡量多模态预测的一致程度：
  - α 高 → 三模态判断趋同，模态间信息冗余
  - α 低 → 模态间存在冲突，此时融合与不确定性建模才有价值
再配合 pairwise α（v-a, v-l, a-l）定位是哪两个模态在打架。
"""

from __future__ import annotations

import itertools

import numpy as np


def _coincidence_matrix(
    ratings: np.ndarray, categories: list[int]
) -> tuple[np.ndarray, float]:
    """构造名义尺度重合矩阵。

    ratings: (n_units, n_raters)，缺失用 np.nan 或 -1 表示。
    返回 (重合矩阵 (K,K), n_total)。
    """
    cat_idx = {c: i for i, c in enumerate(categories)}
    K = len(categories)
    o = np.zeros((K, K), dtype=np.float64)
    n_total = 0.0

    for unit in ratings:
        vals = [v for v in unit if not (v is None or (isinstance(v, float) and np.isnan(v)) or v == -1)]
        m_u = len(vals)
        if m_u < 2:
            continue  # 单个评分无法构成配对，按定义排除
        n_total += m_u
        # 单元内所有有序配对，权重 1/(m_u − 1)
        for a, b in itertools.permutations(vals, 2):
            o[cat_idx[int(a)], cat_idx[int(b)]] += 1.0 / (m_u - 1)
    return o, n_total


def krippendorff_alpha(
    ratings: np.ndarray, categories: list[int] | None = None
) -> float:
    """名义尺度 Krippendorff's α。

    ratings: (n_units, n_raters) 整数标签矩阵，缺失用 -1 或 np.nan。
    返回 α ∈ (−∞, 1]。全一致时 α=1；等于随机水平时 α=0。
    单元不足或退化（只有一个类别）时返回 np.nan。
    """
    arr = np.asarray(ratings, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("ratings 须为 (n_units, n_raters) 二维数组")

    valid = arr[~np.isnan(arr) & (arr != -1)]
    if valid.size == 0:
        return float("nan")
    if categories is None:
        categories = sorted({int(v) for v in valid})
    if len(categories) < 2:
        return float("nan")  # 只有一个类别，α 无定义

    o, n_total = _coincidence_matrix(arr, categories)
    if n_total < 2:
        return float("nan")

    n_c = o.sum(axis=1)  # 边缘计数
    # D_o: 非对角元素之和 / n_total
    d_o = (o.sum() - np.trace(o)) / n_total
    # D_e: 按边缘分布随机配对的期望不一致度
    d_e = (n_c.sum() ** 2 - (n_c ** 2).sum()) / (n_total * (n_total - 1.0))
    if d_e <= 0:
        return float("nan")
    return float(1.0 - d_o / d_e)


def modality_agreement(
    pred_v: np.ndarray,
    pred_a: np.ndarray,
    pred_l: np.ndarray,
    pred_fusion: np.ndarray | None = None,
    num_classes: int | None = None,
) -> dict[str, float]:
    """模态贡献分析：整体 α + pairwise α。

    返回 {"alpha_val": 三模态整体, "alpha_va": ..., "alpha_vl": ...,
          "alpha_al": ..., "alpha_all4": 含融合头（若提供）}
    """
    cats = list(range(num_classes)) if num_classes else None
    v, a, l = (np.asarray(x).ravel() for x in (pred_v, pred_a, pred_l))
    out = {
        "alpha_val": krippendorff_alpha(np.stack([v, a, l], axis=1), cats),
        "alpha_va": krippendorff_alpha(np.stack([v, a], axis=1), cats),
        "alpha_vl": krippendorff_alpha(np.stack([v, l], axis=1), cats),
        "alpha_al": krippendorff_alpha(np.stack([a, l], axis=1), cats),
    }
    if pred_fusion is not None:
        f = np.asarray(pred_fusion).ravel()
        out["alpha_all4"] = krippendorff_alpha(np.stack([v, a, l, f], axis=1), cats)
        out["alpha_vf"] = krippendorff_alpha(np.stack([v, f], axis=1), cats)
        out["alpha_af"] = krippendorff_alpha(np.stack([a, f], axis=1), cats)
        out["alpha_lf"] = krippendorff_alpha(np.stack([l, f], axis=1), cats)
    return out


def _selftest() -> None:
    """对齐已知算例，验证实现正确性。"""
    # 1) 完全一致 -> α = 1
    perfect = np.array([[0, 0], [1, 1], [2, 2], [0, 0]])
    a1 = krippendorff_alpha(perfect)
    assert abs(a1 - 1.0) < 1e-9, f"完全一致应为 1，得 {a1}"

    # 2) 手推算例：2 单元 2 评分者 [(0,0), (0,1)]
    #    重合矩阵 o=[[2,1],[1,0]], n=4；D_o=(4−2)/4=0.5
    #    n_c=[3,1]；D_e=(16−10)/(4·3)=0.5 → α=0
    hand = np.array([[0, 0], [0, 1]])
    a2 = krippendorff_alpha(hand)
    assert abs(a2) < 1e-9, f"手推算例应为 0，得 {a2}"

    # 3) 关键验证：名义尺度、2 评分者、完整数据下 α 应等于 Scott's π
    #    （两者只差 n/(n−1) 小样本校正，n 越大越接近）。
    #    这是可独立推导的恒等式，比记忆中的教材数值可靠。
    def scotts_pi(a, b, K):
        po = (a == b).mean()
        pooled = np.concatenate([a, b])
        pe = sum((pooled == k).mean() ** 2 for k in range(K))
        return (po - pe) / (1 - pe)

    rng = np.random.default_rng(0)
    max_diff = 0.0
    for n, K, agree in [(2000, 4, 0.6), (5000, 8, 0.8)]:
        a = rng.integers(0, K, n)
        b = a.copy()
        flip = rng.random(n) > agree
        b[flip] = rng.integers(0, K, flip.sum())
        al = krippendorff_alpha(np.stack([a, b], axis=1), list(range(K)))
        pi = scotts_pi(a, b, K)
        max_diff = max(max_diff, abs(al - pi))
    assert max_diff < 1e-3, f"α 与 Scott's π 偏差过大: {max_diff}"

    # 4) 完全随机（大样本）-> α ≈ 0
    rand = rng.integers(0, 4, size=(4000, 3))
    a4 = krippendorff_alpha(rand)
    assert abs(a4) < 0.05, f"随机应接近 0，得 {a4:.4f}"

    # 5) 缺失值处理：只有 1 个有效评分的单元应被排除，不应报错
    with_missing = np.array([[0, 0], [1, np.nan], [2, 2], [-1, -1], [1, 1]])
    a5 = krippendorff_alpha(with_missing)
    assert not np.isnan(a5), "含缺失值时不应返回 NaN"

    print(f"agreement 自检通过: perfect={a1:.4f} hand={a2:.4f} "
          f"α-vs-π max_diff={max_diff:.2e} random={a4:.4f} missing={a5:.4f}")


if __name__ == "__main__":
    _selftest()
