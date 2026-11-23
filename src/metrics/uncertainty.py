"""不确定性质量指标 —— 证据式分类头的核心评估（大纲 §5.1 附加项）。

  expected_calibration_error  ECE：置信度分箱后 |acc − conf| 的加权平均
  uncertainty_auroc           AUROC(u_t 区分预测正确/错误)，衡量 u 是否「知道自己错」
  risk_coverage_table         风险-覆盖表：按可靠性 r_t 降序，取前 x% 样本的准确率
  conflict_uncertainty_corr   κ_t 与 u_t 的相关性，验证 L_cal 是否真的起作用

这些函数返回指标；scripts/plot_results.py 负责绘图与源数据导出。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    n_bins: int = 15,
) -> dict[str, float]:
    """ECE 与 MCE。confidence 为预测类别的概率（证据头用 p̂ 的最大值）。

    返回 {"ece": ..., "mce": ..., "avg_conf": ..., "avg_acc": ...}
    """
    yt, yp, conf = (np.asarray(x).ravel() for x in (y_true, y_pred, confidence))
    correct = (yt == yp).astype(np.float64)
    n = len(yt)
    if n == 0:
        return {"ece": float("nan"), "mce": float("nan"),
                "avg_conf": float("nan"), "avg_acc": float("nan")}

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce = 0.0, 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # 最后一箱闭区间，其余左开右闭
        sel = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        m = sel.sum()
        if m == 0:
            continue
        acc_bin = correct[sel].mean()
        conf_bin = conf[sel].mean()
        gap = abs(acc_bin - conf_bin)
        ece += (m / n) * gap
        mce = max(mce, gap)
    return {"ece": float(ece), "mce": float(mce),
            "avg_conf": float(conf.mean()), "avg_acc": float(correct.mean())}


def uncertainty_auroc(
    y_true: np.ndarray, y_pred: np.ndarray, uncertainty: np.ndarray
) -> float:
    """AUROC(u_t 作为「预测错误」的打分器)。

    正类 = 预测错误。u 越大越该错 → AUROC > 0.5 说明 u 有判别力。
    全对或全错时 AUROC 无定义，返回 nan。
    """
    yt, yp, u = (np.asarray(x).ravel() for x in (y_true, y_pred, uncertainty))
    err = (yt != yp).astype(int)
    if err.sum() == 0 or err.sum() == len(err):
        return float("nan")
    return float(roc_auc_score(err, u))


def risk_coverage_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    reliability: np.ndarray,
    coverages: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.5),
) -> list[dict[str, float]]:
    """风险-覆盖表：按可靠性 r_t 降序保留前 x% 样本，报该子集的准确率与风险。

    这是证据式方法最直接的实用价值展示 —— 允许模型「拒答」低可靠样本。
    """
    yt, yp, r = (np.asarray(x).ravel() for x in (y_true, y_pred, reliability))
    order = np.argsort(-r, kind="stable")  # 可靠性降序
    yt, yp = yt[order], yp[order]
    n = len(yt)
    rows = []
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        acc = float((yt[:k] == yp[:k]).mean())
        rows.append({
            "coverage": float(cov),
            "n_kept": k,
            "accuracy": acc,
            "risk": 1.0 - acc,
        })
    return rows


def conflict_uncertainty_corr(
    uncertainty: np.ndarray, kappa: np.ndarray
) -> dict[str, float]:
    """κ_t 与 u_t 的 Pearson / Spearman 相关。

    主模型 L_cal 为正确性硬目标（u_t 回归 1[ŷ_t≠y_t]），不再使用 κ 软目标；
    此相关仅作诊断，检验 u_t 是否仍与模态冲突量 κ_t 保持同向，消融 A13（w/o L_cal）可对照。
    """
    from scipy.stats import pearsonr, spearmanr

    u = np.asarray(uncertainty).ravel()
    k = np.asarray(kappa).ravel()
    stats = {
        "std_u": float(np.std(u)),
        "avg_kappa": float(np.mean(k)),
        "std_kappa": float(np.std(k)),
        "min_kappa": float(np.min(k)),
        "max_kappa": float(np.max(k)),
    }
    if len(u) < 3 or np.std(u) < 1e-8 or np.std(k) < 1e-8:
        return {**stats, "pearson": float("nan"), "spearman": float("nan"),
                "mse_u_kappa": float("nan")}
    return {**stats,
        "pearson": float(pearsonr(u, k)[0]),
        "spearman": float(spearmanr(u, k)[0]),
        "mse_u_kappa": float(((u - k) ** 2).mean()),
    }


def uncertainty_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_hat: np.ndarray,
    uncertainty: np.ndarray | None,
    kappa: np.ndarray | None = None,
) -> dict:
    """汇总所有不确定性指标，供 T7 表格使用。

    p_hat: (N, K) 类别概率；uncertainty: (N,) 或 None（Softmax 头无 u）。
    """
    conf = np.asarray(p_hat).reshape(len(np.asarray(y_true).ravel()), -1).max(axis=1)
    rep: dict = {**expected_calibration_error(y_true, y_pred, conf)}
    if uncertainty is not None:
        u = np.asarray(uncertainty).ravel()
        rep["auroc_u"] = uncertainty_auroc(y_true, y_pred, u)
        rep["avg_u"] = float(u.mean())
        rep["risk_coverage"] = risk_coverage_table(y_true, y_pred, 1.0 - u)
        if kappa is not None:
            rep.update(conflict_uncertainty_corr(u, kappa))
    else:
        # Softmax 头：用 1 − max(p) 作为不确定性代理，便于与证据头对比
        proxy = 1.0 - conf
        rep["auroc_u"] = uncertainty_auroc(y_true, y_pred, proxy)
        rep["avg_u"] = float(proxy.mean())
        rep["risk_coverage"] = risk_coverage_table(y_true, y_pred, conf)
    return rep
