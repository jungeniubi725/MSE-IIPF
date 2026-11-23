"""P2: validation-only calibration of the equal-weight seed ensemble.

Per outer fold:
1. align three seed validation probabilities by seq_id and form p_ens;
2. fit one scalar temperature by validation NLL;
3. apply that same T to the aligned test ensemble;
4. optionally select one scalar logit-adjustment tau by validation macro-F1,
   then fit a temperature after adjustment and apply both to test.

No parameter is selected from test labels. Eight free class biases are not
implemented deliberately; the 208-sequence validation set is too small.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.s8_ensemble_seeds import align_members  # noqa: E402
from src.metrics.calibration import (  # noqa: E402
    apply_logit_adjustment,
    apply_temperature,
    fit_logit_adjustment_tau,
    fit_temperature,
    probability_quality_metrics,
)
from src.metrics.classification import (  # noqa: E402
    classification_metrics,
    confusion_mat,
    per_class_metrics,
)
from src.metrics.ensemble import equal_weight_probability_ensemble  # noqa: E402
from src.utils.config import load_config  # noqa: E402


VARIANTS = ("raw", "temperature", "logit_adjustment_temperature")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="P2: validation-only temperature scaling and scalar logit adjustment"
    )
    p.add_argument("--config", default="configs/ravdess.yaml")
    p.add_argument("--exp-name", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--val-subdir", default="seq_eval_val")
    p.add_argument("--test-subdir", default="seq_eval_test")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--tau-max", type=float, default=2.0)
    p.add_argument("--tau-step", type=float, default=0.05)
    return p.parse_args()


def training_class_prior(
    manifest: Path, fold: int, num_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sequence-label prior from the clean outer-fold training actors only."""
    df = pd.read_csv(manifest)
    seq = df.sort_values(["seq_id", "unit_index"]).drop_duplicates("seq_id")
    train = seq[seq["fold"].notna() & (seq["fold"] != fold)]
    labels = train["seq_label"].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=num_classes).astype(np.int64)
    if counts.size != num_classes or np.any(counts <= 0):
        raise ValueError(f"fold{fold}: class prior contains an empty class: {counts}")
    return counts / counts.sum(), counts


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
    class_names: list[str],
) -> dict:
    pred = probabilities.argmax(axis=1)
    return {
        **classification_metrics(y_true, pred, num_classes, class_names),
        "per_class": per_class_metrics(y_true, pred, num_classes, class_names),
        "confusion_matrix": confusion_mat(y_true, pred, num_classes).tolist(),
        "confusion_matrix_normalized": confusion_mat(
            y_true, pred, num_classes, normalize="true"
        ).tolist(),
        **probability_quality_metrics(y_true, probabilities),
    }


def _member_paths(
    exp_root: Path, seeds: list[int], fold: int, subdir: str
) -> list[Path]:
    return [
        exp_root / f"seed{seed}" / f"fold{fold}" / subdir /
        "seq_predictions.csv"
        for seed in seeds
    ]


def _mean_std(folds: list[dict], variant: str, metric: str) -> dict:
    vals = np.asarray([f["test"][variant][metric] for f in folds], dtype=np.float64)
    return {"mean": float(vals.mean()), "std": float(vals.std()),
            "folds": vals.tolist()}


def _write_matrix(path: Path, class_names: list[str], matrix: list[list[float]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *class_names])
        for label, row in zip(class_names, matrix):
            writer.writerow([label, *row])


def main() -> int:
    args = parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must contain distinct ensemble members")
    if args.tau_max < 0 or args.tau_step <= 0:
        raise ValueError("tau range must be non-negative with a positive step")
    tau_grid = np.arange(0.0, args.tau_max + args.tau_step / 2, args.tau_step)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(str(cfg_path))
    k = int(cfg["data"]["num_classes"])
    class_names = cfg["data"].get("class_names") or [str(i) for i in range(k)]
    manifest = Path(cfg["data"]["manifest"])
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = ROOT / runs_root
    exp_root = runs_root / args.exp_name
    out_dir = Path(args.out_dir) if args.out_dir else exp_root / "calibration_seq"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    folds: list[dict] = []
    pooled: dict[str, list[np.ndarray]] = {
        "y": [], "raw": [], "temperature": [],
        "logit_adjustment_temperature": [],
    }
    prediction_rows = []

    for fold in range(args.folds):
        val_ids, val_y, val_members, val_cols = align_members(
            _member_paths(exp_root, args.seeds, fold, args.val_subdir)
        )
        test_ids, test_y, test_members, test_cols = align_members(
            _member_paths(exp_root, args.seeds, fold, args.test_subdir)
        )
        if val_cols != test_cols or val_cols != [f"p{i}" for i in range(k)]:
            raise ValueError(f"fold{fold}: validation/test probability schema mismatch")
        if set(val_ids) & set(test_ids):
            raise ValueError(f"fold{fold}: validation and test seq_id overlap")

        val_raw = equal_weight_probability_ensemble(val_members)["p_ens"]
        test_raw = equal_weight_probability_ensemble(test_members)["p_ens"]

        # Temperature-only branch: fit on validation NLL, apply unchanged to test.
        t_fit = fit_temperature(val_y, val_raw)
        val_temp = apply_temperature(val_raw, t_fit["temperature"])
        test_temp = apply_temperature(test_raw, t_fit["temperature"])
        if not np.array_equal(test_raw.argmax(1), test_temp.argmax(1)):
            raise AssertionError("temperature scaling changed test argmax")

        # One-dimensional logit adjustment: training prior, tau selected only
        # by validation macro-F1, then temperature fitted on adjusted validation.
        prior, counts = training_class_prior(manifest, fold, k)
        tau_fit = fit_logit_adjustment_tau(val_y, val_raw, prior, tau_grid)
        val_adj = apply_logit_adjustment(val_raw, prior, tau_fit["tau"])
        test_adj = apply_logit_adjustment(test_raw, prior, tau_fit["tau"])
        adj_t_fit = fit_temperature(val_y, val_adj)
        val_adj_temp = apply_temperature(val_adj, adj_t_fit["temperature"])
        test_adj_temp = apply_temperature(test_adj, adj_t_fit["temperature"])
        if not np.array_equal(test_adj.argmax(1), test_adj_temp.argmax(1)):
            raise AssertionError("post-adjustment temperature changed test argmax")

        val_metrics = {
            "raw": evaluate_probabilities(val_y, val_raw, k, class_names),
            "temperature": evaluate_probabilities(val_y, val_temp, k, class_names),
            "logit_adjustment": evaluate_probabilities(
                val_y, val_adj, k, class_names
            ),
            "logit_adjustment_temperature": evaluate_probabilities(
                val_y, val_adj_temp, k, class_names
            ),
        }
        test_metrics = {
            "raw": evaluate_probabilities(test_y, test_raw, k, class_names),
            "temperature": evaluate_probabilities(test_y, test_temp, k, class_names),
            "logit_adjustment": evaluate_probabilities(
                test_y, test_adj, k, class_names
            ),
            "logit_adjustment_temperature": evaluate_probabilities(
                test_y, test_adj_temp, k, class_names
            ),
        }
        record = {
            "fold": fold,
            "n_validation": len(val_ids),
            "n_test": len(test_ids),
            "seeds": args.seeds,
            "selection_data": "validation_only",
            "temperature": t_fit,
            "logit_adjustment": {
                **tau_fit,
                "formula": "z'_c = z_c - tau * log(pi_c)",
                "prior_source": "clean_outer_fold_training_sequences",
                "class_prior": prior.tolist(),
                "class_counts": counts.tolist(),
                "temperature_after_adjustment": adj_t_fit,
            },
            "validation": val_metrics,
            "test": test_metrics,
        }
        folds.append(record)
        with (out_dir / f"fold{fold}_calibration.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(record, f, ensure_ascii=False, indent=2, default=float)

        pooled["y"].append(test_y)
        pooled["raw"].append(test_raw)
        pooled["temperature"].append(test_temp)
        pooled["logit_adjustment_temperature"].append(test_adj_temp)
        for i, sid in enumerate(test_ids):
            prediction_rows.append([
                fold, sid, int(test_y[i]),
                int(test_raw[i].argmax()), int(test_temp[i].argmax()),
                int(test_adj_temp[i].argmax()), t_fit["temperature"],
                tau_fit["tau"], adj_t_fit["temperature"],
                *test_raw[i], *test_temp[i], *test_adj_temp[i],
            ])
        print(
            f"[P2] fold{fold}: T={t_fit['temperature']:.4f} "
            f"tau={tau_fit['tau']:.2f} T_adj={adj_t_fit['temperature']:.4f} | "
            f"test ECE {test_metrics['raw']['ece']:.4f}->"
            f"{test_metrics['temperature']['ece']:.4f}, macro-F1 "
            f"{test_metrics['raw']['macro_f1']:.4f}->"
            f"{test_metrics['logit_adjustment_temperature']['macro_f1']:.4f}"
        )

    metric_keys = ["acc", "macro_f1", "weighted_f1", "ece", "nll", "brier",
                   "aurc", "eaurc"]
    cv = {
        variant: {key: _mean_std(folds, variant, key) for key in metric_keys}
        for variant in VARIANTS
    }
    pooled_metrics = {}
    pooled_y = np.concatenate(pooled["y"])
    for variant in VARIANTS:
        pooled_metrics[variant] = evaluate_probabilities(
            pooled_y, np.concatenate(pooled[variant]), k, class_names
        )
    summary = {
        "experiment": args.exp_name,
        "seeds": args.seeds,
        "n_folds": args.folds,
        "protocol": "fit_on_validation_apply_same_parameters_to_test",
        "temperature_objective": "validation_nll",
        "tau_objective": "validation_macro_f1",
        "tau_grid": tau_grid.tolist(),
        "no_test_parameter_search": True,
        "free_class_biases_fitted": 0,
        "per_fold": folds,
        "cv_mean_std": cv,
        "pooled_unique_sequences": pooled_metrics,
    }
    with (out_dir / "calibration_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    with (out_dir / "calibration_cv_table.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "metric", "fold_mean", "fold_std", "pooled"])
        for variant in VARIANTS:
            for key in metric_keys:
                writer.writerow([
                    variant, key, cv[variant][key]["mean"],
                    cv[variant][key]["std"], pooled_metrics[variant][key],
                ])

    with (out_dir / "calibrated_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold", "seq_id", "true", "pred_raw", "pred_temperature",
            "pred_logit_adjustment_temperature", "temperature", "tau",
            "temperature_after_adjustment",
            *[f"p_raw_{i}" for i in range(k)],
            *[f"p_temperature_{i}" for i in range(k)],
            *[f"p_logit_adjustment_temperature_{i}" for i in range(k)],
        ])
        writer.writerows(prediction_rows)

    # Pooled final variant artifacts; per-fold matrices remain in JSON.
    for variant in VARIANTS:
        m = pooled_metrics[variant]
        _write_matrix(
            out_dir / f"confusion_matrix_{variant}.csv",
            class_names, m["confusion_matrix"],
        )
        with (out_dir / f"per_class_{variant}.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:
            pc = m["per_class"]
            writer = csv.writer(f)
            writer.writerow(["class", "precision", "recall", "f1", "support"])
            for i, name in enumerate(pc["class_names"]):
                writer.writerow([name, pc["precision"][i], pc["recall"][i],
                                 pc["f1"][i], pc["support"][i]])

    print(f"[P2] 完成 -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
