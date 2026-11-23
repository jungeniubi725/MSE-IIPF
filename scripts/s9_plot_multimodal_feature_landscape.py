"""Plot the RAVDESS multimodal feature-file landscape from saved .npy arrays.

Each point is one complete saved feature file. Variable-length (T, 768)
representations from unrelated encoder bases are mapped to a shared set of
rotation-insensitive structural descriptors before PCA and t-SNE. This makes
the joint projection a view of feature structure, not a causal attribution of
classifier performance.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler


SEED = 42
EXPECTED_DIM = 768
FEATURE_GROUPS = (
    ("VideoMAE visual", "visual", "#3B5CCC"),
    ("emotion2vec audio", "audio", "#D55E00"),
    ("RoBERTa–ASR text", "text_sa", "#009E73"),
    ("RoBERTa–visual text", "text_sv", "#7A5195"),
)
DISPLAY_LABELS = {
    "VideoMAE visual": "VideoMAE visual",
    "emotion2vec audio": "emotion2vec audio",
    "RoBERTa–ASR text": "RoBERTa-ASR text",
    "RoBERTa–visual text": "RoBERTa-visual text",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--qa-tools-dir", default=None, type=Path)
    parser.add_argument("--max-iter", type=int, default=1500)
    parser.add_argument("--reuse-source-data", action="store_true")
    return parser.parse_args()


def distribution_stats(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float64).ravel()
    quantiles = np.quantile(values, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    return [
        float(values.mean()),
        float(values.std()),
        float(np.sqrt(np.mean(np.square(values)))),
        float(np.mean(np.abs(values))),
        *[float(value) for value in quantiles],
        float(values.min()),
        float(values.max()),
    ]


def describe_feature(array: np.ndarray) -> np.ndarray:
    x = np.asarray(array, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != EXPECTED_DIM:
        raise ValueError(f"expected (T, {EXPECTED_DIM}), received {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("feature contains NaN or infinite values")

    scale = math.sqrt(x.shape[1])
    token_norm = np.linalg.norm(x, axis=1) / scale
    dimension_mean = x.mean(axis=0)
    dimension_std = x.std(axis=0)

    if len(x) > 1:
        temporal_delta = np.linalg.norm(np.diff(x, axis=0), axis=1) / scale
        left_norm = np.linalg.norm(x[:-1], axis=1)
        right_norm = np.linalg.norm(x[1:], axis=1)
        denominator = np.clip(left_norm * right_norm, 1e-12, None)
        adjacent_cosine = np.sum(x[:-1] * x[1:], axis=1) / denominator
    else:
        temporal_delta = np.zeros(1, dtype=np.float64)
        adjacent_cosine = np.ones(1, dtype=np.float64)

    descriptors = [
        math.log1p(x.shape[0]),
        float(np.mean(x > 0.0)),
        float(np.mean(np.abs(x) < 1e-6)),
        *distribution_stats(x),
        *distribution_stats(token_norm),
        *distribution_stats(dimension_mean),
        *distribution_stats(dimension_std),
        *distribution_stats(temporal_delta),
        *distribution_stats(adjacent_cosine),
    ]
    return np.asarray(descriptors, dtype=np.float64)


def descriptor_names() -> list[str]:
    suffixes = [
        "mean", "std", "rms", "abs_mean", "q05", "q10", "q25", "median",
        "q75", "q90", "q95", "min", "max",
    ]
    names = ["log1p_steps", "fraction_positive", "fraction_near_zero"]
    for prefix in (
        "activation", "token_norm", "dimension_mean", "dimension_std",
        "temporal_delta", "adjacent_cosine",
    ):
        names.extend(f"{prefix}_{suffix}" for suffix in suffixes)
    return names


def build_source_data(features_root: Path, output_csv: Path, max_iter: int) -> tuple[pd.DataFrame, dict]:
    records: list[dict[str, object]] = []
    descriptor_rows: list[np.ndarray] = []
    names = descriptor_names()
    counts: dict[str, int] = {}

    for label, relative_dir, _ in FEATURE_GROUPS:
        paths = sorted((features_root / relative_dir).glob("*.npy"))
        if not paths:
            raise FileNotFoundError(f"no .npy feature files found for {label}")
        counts[label] = len(paths)
        print(f"[features] {label}: {len(paths):,} files", flush=True)
        for index, path in enumerate(paths, start=1):
            array = np.load(path, mmap_mode="r")
            descriptor_rows.append(describe_feature(array))
            records.append({
                "file_id": path.stem,
                "feature_type": label,
                "n_steps": int(array.shape[0] if array.ndim > 1 else 1),
                "feature_dim": int(array.shape[-1]),
            })
            if index % 2000 == 0 or index == len(paths):
                print(f"  {index:,}/{len(paths):,}", flush=True)

    descriptors = np.vstack(descriptor_rows)
    if descriptors.shape[1] != len(names):
        raise AssertionError("descriptor name/column mismatch")

    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    scaled = scaler.fit_transform(descriptors)
    scaled = np.clip(scaled, -12.0, 12.0)
    pca_components = min(24, scaled.shape[1], scaled.shape[0] - 1)
    pca = PCA(n_components=pca_components, random_state=SEED)
    reduced = pca.fit_transform(scaled)
    print(
        f"[PCA] {pca_components} components; cumulative variance="
        f"{pca.explained_variance_ratio_.sum():.4f}",
        flush=True,
    )

    tsne = TSNE(
        n_components=2,
        perplexity=80.0,
        early_exaggeration=12.0,
        learning_rate="auto",
        max_iter=max_iter,
        init="pca",
        random_state=SEED,
        method="barnes_hut",
        angle=0.5,
        n_jobs=-1,
        verbose=1,
    )
    embedding = tsne.fit_transform(reduced)
    embedding -= embedding.mean(axis=0, keepdims=True)

    frame = pd.DataFrame(records)
    for column, values in zip(names, descriptors.T):
        frame[column] = values
    frame["tsne_1"] = embedding[:, 0]
    frame["tsne_2"] = embedding[:, 1]
    frame.to_csv(output_csv, index=False, encoding="utf-8-sig")

    metadata = {
        "point_definition": "one point per saved .npy feature file",
        "n_points": int(len(frame)),
        "counts": counts,
        "feature_dimension": EXPECTED_DIM,
        "descriptor_count": int(descriptors.shape[1]),
        "descriptor_basis": "rotation-insensitive sequence and activation statistics",
        "scaling": "RobustScaler with 10th–90th percentile range; clipped to [-12, 12]",
        "pca_components": pca_components,
        "pca_cumulative_variance": float(pca.explained_variance_ratio_.sum()),
        "projection": {
            "method": "t-SNE (Barnes–Hut)",
            "perplexity": 80.0,
            "max_iter": max_iter,
            "random_seed": SEED,
        },
        "exclusions": 0,
        "caveat": "The projection visualizes feature-file structure, not causal predictive contribution.",
    }
    return frame, metadata


def plot_landscape(frame: pd.DataFrame, output_dir: Path, qa_tools_dir: Path) -> None:
    if qa_tools_dir is not None:
        sys.path.insert(0, str(qa_tools_dir))
        from audit_panel_alignment import require_matplotlib_panel_alignment

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.linewidth": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(7.2047, 4.8819), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.105, top=0.800)

    palette = {label: color for label, _, color in FEATURE_GROUPS}
    # Deterministic hash order prevents one modality from systematically
    # overpainting another without sampling or fabricating observations.
    order = np.argsort(
        pd.util.hash_pandas_object(frame["file_id"], index=False).to_numpy()
    )
    colors = frame["feature_type"].map(palette).to_numpy()[order]
    ax.scatter(
        frame["tsne_1"].to_numpy()[order],
        frame["tsne_2"].to_numpy()[order],
        c=colors,
        s=2.0,
        alpha=0.52,
        linewidths=0,
        rasterized=True,
    )

    x = frame["tsne_1"].to_numpy()
    y = frame["tsne_2"].to_numpy()
    x_pad = max((x.max() - x.min()) * 0.025, 1e-6)
    y_pad = max((y.max() - y.min()) * 0.025, 1e-6)
    ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
    ax.set_ylim(y.min() - y_pad, y.max() + y_pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE 1", fontsize=7.0, labelpad=3)
    ax.set_ylabel("t-SNE 2", fontsize=7.0, labelpad=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#4D4D4D")
        spine.set_linewidth(0.65)

    fig.text(
        0.055, 0.955, "RAVDESS multimodal feature landscape",
        ha="left", va="top", fontsize=9.0, fontweight="bold", color="#1A1A1A",
    )
    fig.text(
        0.055, 0.912,
        "Each point represents one saved feature file; all 24,520 files are shown",
        ha="left", va="top", fontsize=6.5, color="#555555",
    )

    counts = frame["feature_type"].value_counts().to_dict()
    handles = [
        Patch(
            facecolor=color,
            edgecolor="none",
            label=f"{DISPLAY_LABELS[label]} (n={counts[label]:,})",
        )
        for label, _, color in FEATURE_GROUPS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.885),
        ncol=2,
        columnspacing=1.8,
        handletextpad=0.45,
        borderaxespad=0,
        frameon=False,
        fontsize=6.5,
    )

    base = output_dir / "ravdess_multimodal_feature_landscape"
    fig.canvas.draw()
    if qa_tools_dir is not None:
        require_matplotlib_panel_alignment(
            fig,
            json_out=str(base) + ".alignment-layout.json",
            overlay_svg=str(base) + ".alignment-overlay.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=False,
            strict=True,
        )
    fig.savefig(str(base) + ".svg")
    fig.savefig(str(base) + ".pdf")
    fig.savefig(str(base) + ".png", dpi=300)
    fig.savefig(str(base) + ".tiff", dpi=600, format="tiff")
    plt.close(fig)


def write_qa_notes(output_dir: Path, metadata: dict) -> None:
    counts = metadata["counts"]
    notes = f"""# Figure contract and QA notes

## Figure contract

- Core conclusion: Four saved feature-file types occupy distinguishable but partly overlapping regions of a shared structural feature landscape.
- Results-level question: How do the saved visual, audio, ASR-text and visual-description-text representations differ in sequence and activation structure?
- Archetype: Quantitative single-panel point-cloud landscape.
- Backend: Python / Matplotlib only.
- Final size: 183 × 124 mm.
- Hero evidence: all {metadata['n_points']:,} saved feature files shown as individual points.
- Source data: `ravdess_multimodal_feature_landscape_source_data.csv`.
- Exclusions: none.
- Template adaptation: style-only inheritance (dense points, white field, thin border, restrained categorical colors); no source geometry was copied.

## Data mapping

- VideoMAE visual: {counts['VideoMAE visual']:,} unit-level files.
- emotion2vec audio: {counts['emotion2vec audio']:,} unit-level files.
- RoBERTa–ASR text: {counts['RoBERTa–ASR text']:,} sequence-level files.
- RoBERTa–visual text: {counts['RoBERTa–visual text']:,} sequence-level files.
- One point equals one complete `.npy` file. No point sampling or row filtering was used.

## Transform

Variable-length `(T, 768)` arrays from unrelated encoder coordinate systems were not compared dimension-by-dimension. Each file was represented by {metadata['descriptor_count']} rotation-insensitive descriptors covering sequence length, activation distribution, token norms, across-dimension variation, temporal change and adjacent-token cosine similarity. Descriptors were robustly scaled, clipped only after scaling to limit numerical leverage, reduced to {metadata['pca_components']} PCA components ({metadata['pca_cumulative_variance']:.1%} cumulative variance), and projected with Barnes–Hut t-SNE (perplexity 80, seed 42, {metadata['projection']['max_iter']} iterations).

## Interpretation limits

- t-SNE preserves local neighbourhoods more reliably than global distances; island spacing and area are not quantitative effect sizes.
- Unequal point counts reflect the actual file granularity: visual/audio are unit-level, whereas text is sequence-level.
- This figure visualizes feature-file structure and modality-specific representation occupancy. It does not estimate causal contribution or feature attribution to the classifier.
- Colors redundantly map to explicit legend labels and do not rely on red/green discrimination alone.
"""
    (output_dir / "FIGURE_QA_NOTES.md").write_text(notes, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_csv = args.output_dir / "ravdess_multimodal_feature_landscape_source_data.csv"
    metadata_path = args.output_dir / "ravdess_multimodal_feature_landscape_metadata.json"

    if args.reuse_source_data and source_csv.exists() and metadata_path.exists():
        frame = pd.read_csv(source_csv)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        frame, metadata = build_source_data(args.features_root, source_csv, args.max_iter)
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    expected = sum(len(list((args.features_root / relative).glob("*.npy")))
                   for _, relative, _ in FEATURE_GROUPS)
    if len(frame) != expected:
        raise AssertionError(f"source-data rows {len(frame)} != feature files {expected}")
    plot_landscape(frame, args.output_dir, args.qa_tools_dir)
    write_qa_notes(args.output_dir, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
