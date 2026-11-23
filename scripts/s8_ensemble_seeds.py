"""P1: equal-weight sequence probability ensemble across random seeds.

For each fold, rows are joined by ``seq_id`` (never by file order), then

    p_ens = mean_seed(p_seed),  y_pred = argmax(p_ens).

The script deliberately has no weight-search option. Averaged probabilities
are not interpreted as a Dirichlet distribution: uncertainty is reported as
predictive entropy and ensemble mutual information, never as averaged EDL u.

Expected input for each seed/fold:
``runs/<exp>/seed<seed>/fold<fold>/seq_eval/seq_predictions.csv``.
Use ``s3f_seq_eval_summary.py`` first when these files do not yet exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.classification import (  # noqa: E402
    classification_metrics,
    confusion_mat,
    per_class_metrics,
)
from src.metrics.ensemble import equal_weight_probability_ensemble  # noqa: E402
from src.metrics.uncertainty import (  # noqa: E402
    expected_calibration_error,
    risk_coverage_table,
    uncertainty_auroc,
)
from src.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="P1: 按 seq_id 对齐的三 seed 等权 sequence 概率集成"
    )
    p.add_argument("--config", default="configs/ravdess.yaml")
    p.add_argument("--exp-name", required=True,
                   help="runs 下的实验路径，如 simplify/S3 或 p0/Lseq10")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--out-dir", default=None,
                   help="默认 runs/<exp-name>/ensemble_seq")
    return p.parse_args()


def _probability_columns(fieldnames: list[str] | None) -> list[str]:
    if not fieldnames:
        raise ValueError("prediction CSV has no header")
    cols = [c for c in fieldnames if c.startswith("p") and c[1:].isdigit()]
    cols.sort(key=lambda x: int(x[1:]))
    if len(cols) < 2 or cols != [f"p{i}" for i in range(len(cols))]:
        raise ValueError(f"invalid probability columns: {cols}")
    return cols


def read_prediction_csv(path: Path) -> tuple[dict[str, tuple[int, np.ndarray]], list[str]]:
    """Read one member file into a unique ``seq_id`` mapping."""
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: dict[str, tuple[int, np.ndarray]] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        pcols = _probability_columns(reader.fieldnames)
        for line_no, row in enumerate(reader, 2):
            sid = (row.get("seq_id") or "").strip()
            if not sid:
                raise ValueError(f"{path}:{line_no}: empty seq_id")
            if sid in rows:
                raise ValueError(f"{path}:{line_no}: duplicate seq_id {sid!r}")
            try:
                label = int(row["true"])
                prob = np.asarray([float(row[c]) for c in pcols], dtype=np.float64)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: invalid label/probability") from exc
            rows[sid] = (label, prob)
    if not rows:
        raise ValueError(f"{path}: no prediction rows")
    return rows, pcols


def align_members(
    paths: list[Path],
) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    """Inner-join members only after requiring exactly identical ID sets."""
    loaded = [read_prediction_csv(p) for p in paths]
    mappings = [x[0] for x in loaded]
    pcols = loaded[0][1]
    for path, (_, cols) in zip(paths[1:], loaded[1:]):
        if cols != pcols:
            raise ValueError(f"probability schema mismatch in {path}: {cols} != {pcols}")
    ref_ids = set(mappings[0])
    for path, mapping in zip(paths[1:], mappings[1:]):
        ids = set(mapping)
        if ids != ref_ids:
            missing = sorted(ref_ids - ids)[:3]
            extra = sorted(ids - ref_ids)[:3]
            raise ValueError(
                f"seq_id set mismatch in {path}: missing={missing}, extra={extra}"
            )

    seq_ids = sorted(ref_ids)
    labels = np.asarray([mappings[0][sid][0] for sid in seq_ids], dtype=np.int64)
    members = []
    for path, mapping in zip(paths, mappings):
        member_labels = np.asarray([mapping[sid][0] for sid in seq_ids], dtype=np.int64)
        if not np.array_equal(member_labels, labels):
            bad = int(np.flatnonzero(member_labels != labels)[0])
            raise ValueError(
                f"label mismatch for seq_id {seq_ids[bad]!r} in {path}: "
                f"{member_labels[bad]} != {labels[bad]}"
            )
        members.append(np.stack([mapping[sid][1] for sid in seq_ids]))
    return seq_ids, labels, np.stack(members), pcols


def evaluate_ensemble(
    y_true: np.ndarray,
    member_probs: np.ndarray,
    num_classes: int,
    class_names: list[str],
) -> tuple[dict, dict[str, np.ndarray]]:
    unc = equal_weight_probability_ensemble(member_probs)
    p_ens = unc["p_ens"]
    y_pred = p_ens.argmax(axis=-1)
    pe = unc["predictive_entropy"]
    pe_norm = unc["predictive_entropy_normalized"]
    mi = unc["mutual_information"]
    metrics = {
        **classification_metrics(y_true, y_pred, num_classes, class_names),
        "per_class": per_class_metrics(y_true, y_pred, num_classes, class_names),
        "confusion_matrix": confusion_mat(y_true, y_pred, num_classes).tolist(),
        "confusion_matrix_normalized": confusion_mat(
            y_true, y_pred, num_classes, normalize="true"
        ).tolist(),
        "probability_calibration": expected_calibration_error(
            y_true, y_pred, p_ens.max(axis=-1)
        ),
        "ensemble_uncertainty": {
            "definition": "predictive_entropy_and_mutual_information",
            "entropy_unit": "nats",
            "mean_predictive_entropy": float(pe.mean()),
            "mean_predictive_entropy_normalized": float(pe_norm.mean()),
            "mean_expected_member_entropy": float(
                unc["expected_member_entropy"].mean()
            ),
            "mean_mutual_information": float(mi.mean()),
            "auroc_error_predictive_entropy": uncertainty_auroc(
                y_true, y_pred, pe
            ),
            "risk_coverage_predictive_entropy": risk_coverage_table(
                y_true, y_pred, 1.0 - pe_norm
            ),
            "dirichlet_u_reported": False,
        },
    }
    arrays = {
        **unc,
        "y_true": y_true,
        "y_pred": y_pred,
    }
    return metrics, arrays


def _mean_std(rows: list[dict], key: str) -> dict:
    vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
    return {"mean": float(vals.mean()), "std": float(vals.std()),
            "folds": vals.tolist()}


def _write_prediction_rows(
    writer: csv.writer,
    fold: int,
    seq_ids: list[str],
    arrays: dict[str, np.ndarray],
) -> None:
    p_ens = arrays["p_ens"]
    member_pred = arrays["member_probabilities"].argmax(axis=-1)
    for i, sid in enumerate(seq_ids):
        writer.writerow([
            fold, sid, int(arrays["y_true"][i]), int(arrays["y_pred"][i]),
            *[int(member_pred[m, i]) for m in range(member_pred.shape[0])],
            f"{arrays['predictive_entropy'][i]:.9g}",
            f"{arrays['predictive_entropy_normalized'][i]:.9g}",
            f"{arrays['mutual_information'][i]:.9g}",
            *[f"{x:.9g}" for x in p_ens[i]],
        ])


def main() -> int:
    args = parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must contain at least two distinct seeds")
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(str(cfg_path))
    k = int(cfg["data"]["num_classes"])
    class_names = cfg["data"].get("class_names") or [str(i) for i in range(k)]
    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = ROOT / runs_root
    exp_root = runs_root / args.exp_name
    out_dir = Path(args.out_dir) if args.out_dir else exp_root / "ensemble_seq"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_metrics: list[dict] = []
    pooled_ids: list[str] = []
    pooled_labels: list[np.ndarray] = []
    pooled_members: list[np.ndarray] = []
    pred_path = out_dir / "ensemble_predictions.csv"
    with pred_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold", "seq_id", "true", "pred",
            *[f"pred_seed{s}" for s in args.seeds],
            "predictive_entropy", "predictive_entropy_normalized",
            "mutual_information", *[f"p{i}" for i in range(k)],
        ])
        for fold in range(args.folds):
            paths = [
                exp_root / f"seed{seed}" / f"fold{fold}" /
                "seq_eval" / "seq_predictions.csv"
                for seed in args.seeds
            ]
            seq_ids, labels, members, pcols = align_members(paths)
            if pcols != [f"p{i}" for i in range(k)]:
                raise ValueError(f"fold{fold}: expected {k} classes, got {pcols}")
            metrics, arrays = evaluate_ensemble(labels, members, k, class_names)
            metrics.update({
                "fold": fold,
                "n_sequences": len(seq_ids),
                "seeds": args.seeds,
                "alignment": "exact_seq_id_set_then_lexicographic_sort",
                "probability_formula": "p_ens = mean_seed(p_seed)",
            })
            fold_metrics.append(metrics)
            _write_prediction_rows(writer, fold, seq_ids, arrays)
            pooled_ids.extend(seq_ids)
            pooled_labels.append(labels)
            pooled_members.append(members)
            with (out_dir / f"fold{fold}_metrics.json").open("w", encoding="utf-8") as mf:
                json.dump(metrics, mf, ensure_ascii=False, indent=2, default=float)
            print(
                f"[P1] fold{fold}: n={len(seq_ids)} acc={metrics['acc']:.4f} "
                f"macro-F1={metrics['macro_f1']:.4f} "
                f"H={metrics['ensemble_uncertainty']['mean_predictive_entropy']:.4f}"
            )

    if len(set(pooled_ids)) != len(pooled_ids):
        raise ValueError("seq_id appears in more than one fold; pooled CV metrics are ambiguous")
    pooled_metrics, _ = evaluate_ensemble(
        np.concatenate(pooled_labels), np.concatenate(pooled_members, axis=1),
        k, class_names,
    )
    pooled_metrics["n_sequences"] = len(pooled_ids)
    summary_keys = [
        "acc", "macro_p", "macro_r", "macro_f1", "weighted_f1",
        "micro_f1",
    ]
    cv = {key: _mean_std(fold_metrics, key) for key in summary_keys}
    for key, path in (
        ("ece", ("probability_calibration", "ece")),
        ("predictive_entropy", ("ensemble_uncertainty", "mean_predictive_entropy")),
        ("mutual_information", ("ensemble_uncertainty", "mean_mutual_information")),
    ):
        vals = [{key: r[path[0]][path[1]]} for r in fold_metrics]
        cv[key] = _mean_std(vals, key)
    summary = {
        "experiment": args.exp_name,
        "seeds": args.seeds,
        "n_folds": args.folds,
        "weighting": "fixed_equal_weight",
        "alignment": "seq_id",
        "probability_formula": "p_ens = (p42 + p43 + p44) / 3"
        if args.seeds == [42, 43, 44] else "p_ens = mean_seed(p_seed)",
        "uncertainty_note": (
            "Ensemble probabilities are not treated as one Dirichlet; "
            "predictive entropy and mutual information are reported instead."
        ),
        "per_fold": fold_metrics,
        "cv_mean_std": cv,
        "pooled_unique_sequences": pooled_metrics,
    }
    with (out_dir / "ensemble_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

    with (out_dir / "ensemble_cv_table.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "fold_mean", "fold_std", "pooled"])
        for key in summary_keys:
            writer.writerow([key, cv[key]["mean"], cv[key]["std"],
                             pooled_metrics.get(key, "")])
        writer.writerow(["ece", cv["ece"]["mean"], cv["ece"]["std"],
                         pooled_metrics["probability_calibration"]["ece"]])
        writer.writerow(["predictive_entropy", cv["predictive_entropy"]["mean"],
                         cv["predictive_entropy"]["std"],
                         pooled_metrics["ensemble_uncertainty"]["mean_predictive_entropy"]])
        writer.writerow(["mutual_information", cv["mutual_information"]["mean"],
                         cv["mutual_information"]["std"],
                         pooled_metrics["ensemble_uncertainty"]["mean_mutual_information"]])

    with (out_dir / "ensemble_per_class.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        pc = pooled_metrics["per_class"]
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for i, name in enumerate(pc["class_names"]):
            writer.writerow([name, pc["precision"][i], pc["recall"][i],
                             pc["f1"][i], pc["support"][i]])
    for name, matrix in (
        ("ensemble_confusion_matrix.csv", pooled_metrics["confusion_matrix"]),
        ("ensemble_confusion_matrix_normalized.csv",
         pooled_metrics["confusion_matrix_normalized"]),
    ):
        with (out_dir / name).open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["true\\pred", *class_names])
            for label, row in zip(class_names, matrix):
                writer.writerow([label, *row])

    print(
        f"[P1] 完成 {args.exp_name}: fold macro-F1 "
        f"{cv['macro_f1']['mean']:.4f} ± {cv['macro_f1']['std']:.4f}; "
        f"pooled {pooled_metrics['macro_f1']:.4f} -> {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
