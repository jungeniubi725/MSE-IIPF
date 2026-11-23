"""Plot the full RAVDESS IIPF fusion landscape with upstream-source glyphs.

The saved IIPF test table contains one 256-dimensional fused vector per valid
unit. Equivalent train and validation exports from the same checkpoint are
accepted so that all RAVDESS source files can be mapped to the fused manifold.

Visual and audio files are unit-level and therefore inherit the coordinate of
their matching fused unit. The two text files are sequence-level and inherit
the centroid of the four fused units in that sequence. Small fixed source-
specific offsets make coincident markers visible; the offsets do not encode a
scientific distance. Prediction regions are robust circular summaries of the
saved evidential head's final unit-level predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F


SEED = 42
CLASS_NAMES = (
    "Neutral",
    "Calm",
    "Happy",
    "Sad",
    "Angry",
    "Fearful",
    "Disgust",
    "Surprised",
)
SOURCE_GROUPS = (
    ("VideoMAE visual", "#3B5CCC", "unit", (-1.0, 0.0)),
    ("emotion2vec audio", "#D55E00", "unit", (1.0, 0.0)),
    ("RoBERTa-ASR text", "#009E73", "sequence", (0.0, -1.0)),
    ("RoBERTa-visual text", "#7A5195", "sequence", (0.0, 1.0)),
)
CLASS_LINE_COLORS = (
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#17BECF",
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-fusion", required=True, type=Path)
    parser.add_argument("--val-fusion", required=True, type=Path)
    parser.add_argument("--test-fusion", required=True, type=Path)
    parser.add_argument("--evidence-head", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--qa-tools-dir", default=None, type=Path)
    parser.add_argument("--max-iter", type=int, default=1500)
    parser.add_argument("--reuse-embedding", action="store_true")
    return parser.parse_args()


def read_fusion_tables(args: argparse.Namespace) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split, path in (
        ("train", args.train_fusion),
        ("val", args.val_fusion),
        ("test", args.test_fusion),
    ):
        frame = pd.read_csv(path)
        frame.insert(1, "split", split)
        frames.append(frame)

    frame = pd.concat(frames, ignore_index=True)
    feature_columns = [f"f{i}" for i in range(256)]
    required = {"id", "label", *feature_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"fusion tables are missing columns: {sorted(missing)}")
    if len(frame) != 9808:
        raise ValueError(f"expected 9,808 fused units, received {len(frame):,}")
    if frame["id"].duplicated().any():
        duplicated = frame.loc[frame["id"].duplicated(), "id"].iloc[0]
        raise ValueError(f"duplicate unit id: {duplicated}")
    if set(frame["label"].astype(int)) != set(range(8)):
        raise ValueError("all eight RAVDESS labels must be present")
    x = frame[feature_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("fusion features contain NaN or infinite values")
    if np.any(np.all(x == 0.0, axis=1)):
        raise ValueError("fusion features contain an all-zero row")
    frame["seq_id"] = frame["id"].str.rsplit("#", n=1).str[0]
    unit_counts = frame.groupby("seq_id", sort=False).size()
    if len(unit_counts) != 2452 or not np.all(unit_counts.to_numpy() == 4):
        raise ValueError(
            "expected 2,452 sequences with exactly four fused units each"
        )
    return frame


def add_predictions(
    frame: pd.DataFrame, evidence_head: Path
) -> tuple[pd.DataFrame, np.ndarray]:
    feature_columns = [f"f{i}" for i in range(256)]
    x = frame[feature_columns].to_numpy(dtype=np.float32)
    state = torch.load(evidence_head, map_location="cpu", weights_only=True)
    expected_shapes = {
        "0.weight": (256, 256),
        "0.bias": (256,),
        "2.weight": (8, 256),
        "2.bias": (8,),
    }
    for key, shape in expected_shapes.items():
        if key not in state or tuple(state[key].shape) != shape:
            raise ValueError(f"unexpected evidence-head tensor {key}")

    with torch.inference_mode():
        tensor = torch.from_numpy(x)
        hidden = F.relu(F.linear(tensor, state["0.weight"], state["0.bias"]))
        logits = F.linear(hidden, state["2.weight"], state["2.bias"])
        evidence = F.softplus(logits).clamp(max=1e4)
        alpha = evidence + 1.0
        probabilities = alpha / alpha.sum(dim=1, keepdim=True)
    probs = probabilities.numpy()
    frame = frame.copy()
    frame["predicted_label"] = probs.argmax(axis=1).astype(np.int64)
    frame["predicted_emotion"] = [
        CLASS_NAMES[index] for index in frame["predicted_label"]
    ]
    frame["prediction_confidence"] = probs.max(axis=1)
    frame["prediction_correct"] = (
        frame["predicted_label"].to_numpy() == frame["label"].to_numpy()
    )
    return frame, probs


def compute_embedding(
    frame: pd.DataFrame,
    output_csv: Path,
    max_iter: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    feature_columns = [f"f{i}" for i in range(256)]
    x = frame[feature_columns].to_numpy(dtype=np.float32)
    scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_components=50, random_state=SEED)
    reduced = pca.fit_transform(scaled)
    print(
        f"[PCA] 50 components; cumulative variance="
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
    frame = frame.copy()
    frame["tsne_1"] = embedding[:, 0]
    frame["tsne_2"] = embedding[:, 1]

    keep_columns = [
        "id",
        "seq_id",
        "split",
        "label",
        "predicted_label",
        "predicted_emotion",
        "prediction_confidence",
        "prediction_correct",
        "tsne_1",
        "tsne_2",
    ]
    frame[keep_columns].to_csv(output_csv, index=False, encoding="utf-8-sig")
    metadata = {
        "fused_unit_count": int(len(frame)),
        "sequence_count": int(frame["seq_id"].nunique()),
        "fusion_dimension": 256,
        "pca_components": 50,
        "pca_cumulative_variance": float(pca.explained_variance_ratio_.sum()),
        "projection": {
            "method": "t-SNE (Barnes-Hut)",
            "perplexity": 80.0,
            "max_iter": int(max_iter),
            "random_seed": SEED,
        },
    }
    return frame, metadata


def reuse_embedding(frame: pd.DataFrame, output_csv: Path) -> pd.DataFrame:
    saved = pd.read_csv(output_csv)
    required = {
        "id",
        "tsne_1",
        "tsne_2",
        "predicted_label",
        "prediction_confidence",
    }
    if not required.issubset(saved.columns):
        raise ValueError("saved embedding is missing required columns")
    if len(saved) != len(frame) or set(saved["id"]) != set(frame["id"]):
        raise ValueError("saved embedding does not match the fusion tables")
    coordinates = saved.set_index("id")[["tsne_1", "tsne_2"]]
    frame = frame.copy()
    frame["tsne_1"] = frame["id"].map(coordinates["tsne_1"])
    frame["tsne_2"] = frame["id"].map(coordinates["tsne_2"])
    return frame


def build_point_cloud(frame: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    x_range = float(frame["tsne_1"].max() - frame["tsne_1"].min())
    y_range = float(frame["tsne_2"].max() - frame["tsne_2"].min())
    dx = 0.0026 * x_range
    dy = 0.0026 * y_range

    unit_base = frame[
        [
            "id",
            "seq_id",
            "split",
            "label",
            "predicted_label",
            "predicted_emotion",
            "prediction_confidence",
            "prediction_correct",
            "tsne_1",
            "tsne_2",
        ]
    ].copy()
    sequence_base = (
        frame.groupby("seq_id", sort=False)
        .agg(
            split=("split", "first"),
            label=("label", "first"),
            predicted_label=(
                "predicted_label",
                lambda values: int(np.bincount(values, minlength=8).argmax()),
            ),
            prediction_confidence=("prediction_confidence", "mean"),
            prediction_correct=("prediction_correct", "mean"),
            tsne_1=("tsne_1", "mean"),
            tsne_2=("tsne_2", "mean"),
        )
        .reset_index()
    )
    sequence_base["id"] = sequence_base["seq_id"]
    sequence_base["predicted_emotion"] = [
        CLASS_NAMES[index] for index in sequence_base["predicted_label"]
    ]

    clouds: list[pd.DataFrame] = []
    for source_type, _, granularity, offset in SOURCE_GROUPS:
        cloud = unit_base.copy() if granularity == "unit" else sequence_base.copy()
        cloud["source_type"] = source_type
        cloud["source_granularity"] = granularity
        cloud["plot_x"] = cloud["tsne_1"] + offset[0] * dx
        cloud["plot_y"] = cloud["tsne_2"] + offset[1] * dy
        clouds.append(cloud)
    point_cloud = pd.concat(clouds, ignore_index=True)
    expected_counts = {
        "VideoMAE visual": 9808,
        "emotion2vec audio": 9808,
        "RoBERTa-ASR text": 2452,
        "RoBERTa-visual text": 2452,
    }
    observed = point_cloud["source_type"].value_counts().to_dict()
    if observed != expected_counts:
        raise AssertionError(f"source-marker counts {observed} != {expected_counts}")
    if len(point_cloud) != 24520:
        raise AssertionError("point cloud must contain exactly 24,520 markers")
    return point_cloud, dx, dy


def class_circles(frame: pd.DataFrame) -> list[dict[str, object]]:
    circles: list[dict[str, object]] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        subset = frame.loc[frame["predicted_label"] == class_index]
        if subset.empty:
            raise ValueError(f"no prediction for class {class_name}")
        points = subset[["tsne_1", "tsne_2"]].to_numpy(dtype=np.float64)
        center = np.median(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        radius = float(np.quantile(distances, 0.65))
        radius = max(radius, 0.015 * max(np.ptp(frame["tsne_1"]), np.ptp(frame["tsne_2"])))
        circles.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "count": int(len(subset)),
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "radius": radius,
                "coverage": float(np.mean(distances <= radius)),
            }
        )
    return circles


def plot_landscape(
    frame: pd.DataFrame,
    point_cloud: pd.DataFrame,
    circles: list[dict[str, object]],
    output_dir: Path,
    qa_tools_dir: Path,
) -> None:
    if qa_tools_dir is not None:
        sys.path.insert(0, str(qa_tools_dir))
        from audit_panel_alignment import require_matplotlib_panel_alignment

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(7.2047, 4.8819), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.150, top=0.785)

    palette = {name: color for name, color, _, _ in SOURCE_GROUPS}
    order = np.argsort(
        pd.util.hash_pandas_object(
            point_cloud[["id", "source_type"]], index=False
        ).to_numpy()
    )
    ax.scatter(
        point_cloud["plot_x"].to_numpy()[order],
        point_cloud["plot_y"].to_numpy()[order],
        c=point_cloud["source_type"].map(palette).to_numpy()[order],
        s=2.15,
        alpha=0.50,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    for item in sorted(circles, key=lambda value: float(value["radius"]), reverse=True):
        class_index = int(item["class_index"])
        color = CLASS_LINE_COLORS[class_index]
        circle = Circle(
            (float(item["center_x"]), float(item["center_y"])),
            float(item["radius"]),
            fill=False,
            edgecolor=color,
            linewidth=0.85,
            linestyle=(0, (5, 2)),
            alpha=0.92,
            zorder=4,
        )
        ax.add_patch(circle)

    x = point_cloud["plot_x"].to_numpy()
    y = point_cloud["plot_y"].to_numpy()
    x_pad = max((x.max() - x.min()) * 0.025, 1e-6)
    y_pad = max((y.max() - y.min()) * 0.025, 1e-6)
    ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
    ax.set_ylim(y.min() - y_pad, y.max() + y_pad)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#4D4D4D")
        spine.set_linewidth(0.65)

    accuracy = float(frame["prediction_correct"].mean())
    test_accuracy = float(
        frame.loc[frame["split"] == "test", "prediction_correct"].mean()
    )
    fig.text(
        0.055,
        0.955,
        "RAVDESS IIPF fusion landscape",
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color="#1A1A1A",
    )
    fig.text(
        0.055,
        0.914,
        "24,520 source-file markers anchored to 9,808 fused units "
        f"(pooled acc. {accuracy:.1%}; test acc. {test_accuracy:.1%})",
        ha="left",
        va="top",
        fontsize=6.25,
        color="#555555",
    )

    counts = point_cloud["source_type"].value_counts().to_dict()
    modality_handles = [
        Patch(
            facecolor=color,
            edgecolor="none",
            label=f"{name} (n={counts[name]:,})",
        )
        for name, color, _, _ in SOURCE_GROUPS
    ]
    fig.legend(
        handles=modality_handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.878),
        ncol=2,
        columnspacing=1.8,
        handletextpad=0.45,
        borderaxespad=0,
        frameon=False,
        fontsize=6.2,
    )
    emotion_handles = [
        Line2D(
            [0],
            [0],
            color=CLASS_LINE_COLORS[index],
            linewidth=1.15,
            linestyle=(0, (5, 2)),
            label=class_name,
        )
        for index, class_name in enumerate(CLASS_NAMES)
    ]
    fig.legend(
        handles=emotion_handles,
        loc="lower left",
        bbox_to_anchor=(0.055, 0.025),
        ncol=8,
        columnspacing=0.95,
        handlelength=1.8,
        handletextpad=0.35,
        borderaxespad=0,
        frameon=False,
        fontsize=5.5,
    )

    base = output_dir / "ravdess_iipf_fusion_landscape"
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


def write_qa_notes(
    output_dir: Path,
    frame: pd.DataFrame,
    point_cloud: pd.DataFrame,
    circles: list[dict[str, object]],
    metadata: dict[str, object],
    marker_dx: float,
    marker_dy: float,
) -> None:
    counts = point_cloud["source_type"].value_counts().to_dict()
    predicted_counts = (
        frame["predicted_label"].value_counts().sort_index().to_dict()
    )
    metadata.update(
        {
            "source_marker_count": int(len(point_cloud)),
            "source_marker_counts": counts,
            "prediction_accuracy": float(frame["prediction_correct"].mean()),
            "split_prediction_accuracy": {
                split: float(group["prediction_correct"].mean())
                for split, group in frame.groupby("split", sort=False)
            },
            "predicted_class_counts": {
                CLASS_NAMES[index]: int(predicted_counts[index]) for index in range(8)
            },
            "emotion_circle_definition": (
                "circle centered at the coordinate-wise median of unit embeddings "
                "for each predicted class; radius is the 65th percentile distance"
            ),
            "source_marker_offsets": {
                "tsne_1": marker_dx,
                "tsne_2": marker_dy,
                "meaning": "fixed visibility offsets only; not scientific distances",
            },
            "exclusions": 0,
        }
    )
    (output_dir / "ravdess_iipf_fusion_landscape_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    circle_lines = "\n".join(
        f"- {item['class_name']}: predicted n={item['count']:,}; "
        f"circle coverage={item['coverage']:.1%}."
        for item in circles
    )
    notes = f"""# Figure contract and QA notes

## Figure contract

- Core conclusion: The four upstream RAVDESS feature-file types map onto a shared IIPF fusion manifold, while the saved evidential head defines eight partly overlapping predicted-emotion regions.
- Results-level question: How does IIPF reorganize the multimodal feature space, and where do final predicted categories occupy that space?
- Archetype: Quantitative single-panel point-cloud landscape.
- Backend: Python / Matplotlib only.
- Final size: 183 x 124 mm.
- Hero evidence: all {len(point_cloud):,} upstream source-file markers, anchored to all {len(frame):,} fused unit embeddings.
- Source data: `ravdess_iipf_fusion_landscape_fused_units.csv` and `ravdess_iipf_fusion_landscape_source_data.csv`.
- Exclusions: none.
- Template adaptation: style-only inheritance from the provided point-cloud example.

## Data mapping

- VideoMAE visual: {counts['VideoMAE visual']:,} unit-level markers.
- emotion2vec audio: {counts['emotion2vec audio']:,} unit-level markers.
- RoBERTa-ASR text: {counts['RoBERTa-ASR text']:,} sequence-level markers.
- RoBERTa-visual text: {counts['RoBERTa-visual text']:,} sequence-level markers.
- Visual/audio markers inherit their matching fused-unit coordinate. Each text marker inherits the centroid of its sequence's four fused-unit coordinates.
- Fixed display offsets make coincident source markers visible. These offsets do not represent modality distance or feature attribution.

## Transform and prediction

- Every fused vector has 256 dimensions. Features were standardized, reduced to 50 PCA components ({metadata['pca_cumulative_variance']:.1%} cumulative variance), then projected by Barnes-Hut t-SNE (perplexity 80, seed 42, {metadata['projection']['max_iter']} iterations).
- Final predictions were recomputed directly from the saved evidential classification head using `softplus(logits) + 1` Dirichlet parameters and `argmax(alpha / sum(alpha))`.
- Unit-level prediction accuracy against the saved labels is {frame['prediction_correct'].mean():.1%} when train/validation/test are pooled for full-dataset visualization; test-only accuracy is {frame.loc[frame['split'] == 'test', 'prediction_correct'].mean():.1%}.
- Dashed emotion circles are descriptive 65% core regions, not confidence intervals or decision boundaries.
- Circle colors are keyed to the eight emotion categories in the lower-left legend.

## Predicted emotion regions

{circle_lines}

## Interpretation limits

- t-SNE preserves local neighbourhoods more reliably than global distance, orientation, island area or gap size.
- The four colors encode upstream source provenance. They do not claim four independent post-fusion embeddings; IIPF stores one shared fused vector per unit.
- Text sources are sequence-level, while visual/audio sources are unit-level; their counts therefore differ by a factor of four.
- Overlapping class circles indicate that predicted categories are not cleanly separated in this two-dimensional projection.
- The displayed classification accuracy is the unmodified result of the saved classifier and features.
"""
    (output_dir / "FIGURE_QA_NOTES.md").write_text(notes, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fused_csv = args.output_dir / "ravdess_iipf_fusion_landscape_fused_units.csv"
    source_csv = args.output_dir / "ravdess_iipf_fusion_landscape_source_data.csv"

    frame = read_fusion_tables(args)
    frame, _ = add_predictions(frame, args.evidence_head)
    if args.reuse_embedding and fused_csv.exists():
        frame = reuse_embedding(frame, fused_csv)
        metadata_path = args.output_dir / "ravdess_iipf_fusion_landscape_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            raise FileNotFoundError(
                "--reuse-embedding requires the existing metadata JSON"
            )
    else:
        frame, metadata = compute_embedding(frame, fused_csv, args.max_iter)

    point_cloud, marker_dx, marker_dy = build_point_cloud(frame)
    point_cloud.to_csv(source_csv, index=False, encoding="utf-8-sig")
    circles = class_circles(frame)
    plot_landscape(frame, point_cloud, circles, args.output_dir, args.qa_tools_dir)
    write_qa_notes(
        args.output_dir,
        frame,
        point_cloud,
        circles,
        metadata,
        marker_dx,
        marker_dy,
    )
    print(
        json.dumps(
            {
                "fused_units": len(frame),
                "source_markers": len(point_cloud),
                "prediction_accuracy": float(frame["prediction_correct"].mean()),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
