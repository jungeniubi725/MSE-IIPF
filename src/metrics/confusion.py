"""混淆矩阵导出为 CSV（experiment.txt 指标 3）。不画图，只出表。"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .classification import confusion_mat


def save_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: list[str],
    out_dir: str,
    tag: str = "T3",
) -> dict[str, str]:
    """输出两个 CSV：计数矩阵与行归一化矩阵。

    行 = 真实类别，列 = 预测类别，与 sklearn 约定一致。
    返回 {"count": path, "norm": path}。
    """
    os.makedirs(out_dir, exist_ok=True)
    names = class_names or [str(i) for i in range(num_classes)]

    cm_count = confusion_mat(y_true, y_pred, num_classes, normalize=None)
    cm_norm = confusion_mat(y_true, y_pred, num_classes, normalize="true")

    paths = {}
    for key, mat, fmt in (("count", cm_count, "%d"), ("norm", cm_norm, "%.4f")):
        df = pd.DataFrame(mat, index=names, columns=names)
        df.index.name = "true\\pred"
        p = os.path.join(out_dir, f"{tag}_confusion_{key}.csv")
        df.to_csv(p, float_format="%.4f" if key == "norm" else None)
        paths[key] = p
    return paths


def confusion_summary(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, class_names: list[str]
) -> pd.DataFrame:
    """混淆矩阵的可读摘要：每类最常被误判为哪一类。

    用于论文中讨论「哪些情感易混淆」，比整张矩阵更便于叙述。
    """
    cm = confusion_mat(y_true, y_pred, num_classes, normalize="true")
    names = class_names or [str(i) for i in range(num_classes)]
    rows = []
    for i in range(num_classes):
        row = cm[i].copy()
        correct = row[i]
        row[i] = -1.0  # 排除对角线找最大误判
        j = int(row.argmax())
        rows.append({
            "class": names[i],
            "recall": round(float(correct), 4),
            "top_confusion": names[j],
            "top_confusion_rate": round(float(row[j]), 4),
        })
    return pd.DataFrame(rows)
