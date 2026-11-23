"""分类指标：Accuracy / Precision / Recall / F1，支持 macro / micro / weighted。

与 sklearn.metrics 对齐，但输出格式统一为字典便于日志与表格汇总。
micro-P = micro-R = micro-F1 = Acc（单标签多分类的恒等性）。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int | None = None,
    class_names: list[str] | None = None,
    prefix: str = "",
) -> dict[str, float]:
    """计算四件套 × 三种平均（macro / micro / weighted）。

    y_true, y_pred: (N,) 或 (B,T) 展平后的标签数组。
    返回 {"acc": ..., "macro_p": ..., "macro_r": ..., "macro_f1": ..., ...}
    prefix 用于给键加前缀（如 "val_" -> "val_acc"）。
    """
    yt = y_true.ravel()
    yp = y_pred.ravel()
    acc = accuracy_score(yt, yp)

    # macro / micro / weighted 的 P/R/F1
    kw = {"zero_division": 0}
    if num_classes is not None:
        kw["labels"] = list(range(num_classes))
    macro_p = precision_score(yt, yp, average="macro", **kw)
    macro_r = recall_score(yt, yp, average="macro", **kw)
    macro_f1 = f1_score(yt, yp, average="macro", **kw)

    micro_p = precision_score(yt, yp, average="micro", **kw)
    micro_r = recall_score(yt, yp, average="micro", **kw)
    micro_f1 = f1_score(yt, yp, average="micro", **kw)

    weighted_p = precision_score(yt, yp, average="weighted", **kw)
    weighted_r = recall_score(yt, yp, average="weighted", **kw)
    weighted_f1 = f1_score(yt, yp, average="weighted", **kw)

    out = {
        f"{prefix}acc": acc,
        f"{prefix}macro_p": macro_p,
        f"{prefix}macro_r": macro_r,
        f"{prefix}macro_f1": macro_f1,
        f"{prefix}micro_p": micro_p,
        f"{prefix}micro_r": micro_r,
        f"{prefix}micro_f1": micro_f1,
        f"{prefix}weighted_p": weighted_p,
        f"{prefix}weighted_r": weighted_r,
        f"{prefix}weighted_f1": weighted_f1,
    }
    return out


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: list[str] | None = None,
) -> dict[str, list[float]]:
    """逐类 P/R/F1/support，返回长度为 K 的列表。"""
    yt = y_true.ravel()
    yp = y_pred.ravel()
    labels = list(range(num_classes))
    p = precision_score(yt, yp, labels=labels, average=None, zero_division=0)
    r = recall_score(yt, yp, labels=labels, average=None, zero_division=0)
    f1 = f1_score(yt, yp, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(yt, yp, labels=labels)
    support = cm.sum(axis=1)
    return {
        "precision": p.tolist(),
        "recall": r.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "class_names": class_names or [str(i) for i in labels],
    }


def confusion_mat(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    normalize: str | None = None,
) -> np.ndarray:
    """混淆矩阵。normalize: None（计数）/ "true"（行归一化）/ "pred"（列归一化）。"""
    yt = y_true.ravel()
    yp = y_pred.ravel()
    cm = confusion_matrix(yt, yp, labels=list(range(num_classes)))
    if normalize == "true":
        row_sum = cm.sum(axis=1, keepdims=True).clip(min=1e-6)
        cm = cm / row_sum
    elif normalize == "pred":
        col_sum = cm.sum(axis=0, keepdims=True).clip(min=1e-6)
        cm = cm / col_sum
    return cm
