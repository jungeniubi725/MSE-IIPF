"""Plot real exported predictions, with CSV source data for every quantitative panel."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.artifacts import save_json
from src.metrics.classification import classification_metrics, per_class_metrics
from src.metrics.uncertainty import uncertainty_report


def load_predictions(path, level):
    with np.load(path, allow_pickle=False) as archive:
        key = "seq_" if level == "sequence" else ""
        required = [key + "y_true", key + "p_hat", "seq_ids" if key else "unit_ids"]
        if any(name not in archive for name in required):
            raise ValueError(f"Missing {level} labels/probabilities/IDs in {path}")
        y, p, ids = [archive[name] for name in required]
        yp = p.argmax(-1)
        u = archive[key + "u"] if key + "u" in archive else 1 - p.max(-1)
        uncertainty_source = "saved uncertainty" if key + "u" in archive else "1 - max probability"
    if p.ndim != 2 or len(p) != len(y) or len(ids) != len(y) or len(u) != len(y) or not len(y):
        raise ValueError("Misaligned or empty prediction arrays")
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-4):
        raise ValueError("Invalid probabilities")
    if not np.isfinite(u).all() or len(set(ids.tolist())) != len(ids):
        raise ValueError("Invalid uncertainty or duplicate IDs")
    return y.astype(int), yp, p, u, ids, uncertainty_source


def save_figure(fig, stem):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix("." + extension), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--level", choices=["unit", "sequence"], default="sequence")
    parser.add_argument("--class-names", nargs="+", help="In class-index order; defaults to numeric class IDs")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--history", type=Path, help="Optional RAVDESS/CMU training history JSON")
    args = parser.parse_args()
    y, yp, p, u, ids, u_source = load_predictions(args.predictions, args.level)
    k = p.shape[1]
    names = args.class_names or [str(i) for i in range(k)]
    if len(names) != k or np.any((y < 0) | (y >= k)):
        parser.error("Class names/labels do not match the probability columns")
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    matrix = confusion_matrix(y, yp, labels=list(range(k)))
    pd.DataFrame(matrix, index=names, columns=names).to_csv(out / "confusion_counts.csv", index_label="true_class")
    normalized = matrix / np.maximum(1, matrix.sum(1, keepdims=True))
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{matrix[i,j]}\n{normalized[i,j]:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if normalized[i,j] > 0.55 else "black")
    ax.set(xticks=range(k), yticks=range(k), xticklabels=names, yticklabels=names, xlabel="Predicted class", ylabel="True class")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.colorbar(im, ax=ax, label="Fraction within true class")
    save_figure(fig, out / "confusion_matrix")

    confidence, correct = p.max(1), (yp == y)
    bins = np.linspace(0, 1, 16)
    rows = []
    for i in range(15):
        select = (confidence >= bins[i] if i == 0 else confidence > bins[i]) & (confidence <= bins[i+1])
        rows.append({"lower": bins[i], "upper": bins[i+1], "count": int(select.sum()),
                     "confidence": float(confidence[select].mean()) if select.any() else np.nan,
                     "accuracy": float(correct[select].mean()) if select.any() else np.nan})
    reliability = pd.DataFrame(rows)
    reliability.to_csv(out / "reliability_bins.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="0.6", label="Perfect calibration")
    ax.plot(reliability.confidence, reliability.accuracy, "o-", color="#3567A6", label="Observed")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy")
    ax.legend(frameon=False)
    save_figure(fig, out / "reliability")

    order = np.argsort(u, kind="stable")
    coverage = np.arange(1, len(y)+1) / len(y)
    risk = np.cumsum(~correct[order]) / np.arange(1, len(y)+1)
    pd.DataFrame({"sample_id": ids[order], "coverage": coverage, "risk": risk, "uncertainty": u[order]}).to_csv(out / "risk_coverage.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.plot(coverage, risk, color="#3567A6")
    ax.set(xlabel="Coverage", ylabel="Risk (error rate)", xlim=(0, 1), ylim=(0, max(0.05, float(risk.max())*1.05)))
    save_figure(fig, out / "risk_coverage")
    pd.DataFrame({"sample_id": ids, "y_true": y, "y_pred": yp, "confidence": confidence, "uncertainty": u}).to_csv(out / "prediction_source_data.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    edges = np.linspace(float(u.min()), float(u.max()) + 1e-8, 21)
    for mask, label, color in [(correct, "Correct", "#3567A6"), (~correct, "Incorrect", "#C65A36")]:
        ax.hist(u[mask], bins=edges, alpha=0.65, label=label, color=color)
    ax.set(xlabel="Uncertainty", ylabel="Samples")
    ax.legend(frameon=False)
    save_figure(fig, out / "uncertainty")
    if args.history:
        history = json.loads(args.history.read_text(encoding="utf-8"))
        records = history.get("history", []) if isinstance(history, dict) else history
        hist = pd.DataFrame([{"epoch": r["epoch"], "train_loss": r.get("train", {}).get("loss", r.get("train_loss")),
                              "val_macro_f1": r.get("val", {}).get("seq_macro_f1", r.get("val", {}).get("macro_f1", r.get("val_macro_f1")))} for r in records])
        hist.to_csv(out / "training_history.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        axes[0].plot(hist.epoch, hist.train_loss, color="#3567A6")
        axes[0].set(xlabel="Epoch", ylabel="Training loss")
        axes[1].plot(hist.epoch, hist.val_macro_f1, color="#3567A6")
        axes[1].set(xlabel="Epoch", ylabel="Validation macro-F1")
        save_figure(fig, out / "training_curves")
    report = {"level": args.level, "n_samples": len(y), "uncertainty_source": u_source,
              **classification_metrics(y, yp, k), "per_class": per_class_metrics(y, yp, k, names),
              "uncertainty": uncertainty_report(y, yp, p, u), "aurc_discrete_mean": float(risk.mean())}
    # Trapezoid area over the specified partial interval, inserting its exact lower boundary.
    x = np.r_[0.5, coverage[coverage > 0.5]]
    r = np.r_[np.interp(0.5, coverage, risk), risk[coverage > 0.5]]
    report["paurc_0.5_1_trapezoid"] = float(np.sum(np.diff(x) * (r[:-1] + r[1:]) / 2))
    save_json(out / "metrics_recomputed.json", report)
    print(f"Figures and source data -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

