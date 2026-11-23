"""CMU-MOSEI CSV -> ALBEF alignment -> IIPF -> evidential classification.

SG-DSFE is intentionally bypassed. Large frame CSVs are streamed one video at
a time and compacted directly into label-aligned, fixed-length sequences.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


WORKSPACE = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = WORKSPACE / "runs" / "cmu_mosei"
DATA_ROOT = WORKSPACE / "data" / "cmu_mosei"
BASE_SCRIPT = Path(__file__).with_name("run_mosi_bypass_experiment.py")
SDK_SPLIT_FILE = DATA_ROOT / "official_splits.json"

spec = importlib.util.spec_from_file_location("mosi_experiment_base", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import shared experiment implementation: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

TriALBEFAlign = base.TriALBEFAlign
TriCfg = base.TriCfg

CLASS_NAMES = ["Negative", "Neutral", "Positive"]
KEYS = ["video_id", "start_time", "end_time"]

DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "label_csv": str(DATA_ROOT / "CMU_MOSEI_Labels.csv"),
        "audio_csv": str(DATA_ROOT / "CMU_MOSEI_COVAREP.csv"),
        "text_csv": str(DATA_ROOT / "CMU_MOSEI_TimestampedWordVectors.csv"),
        "vision_csv": str(DATA_ROOT / "CMU_MOSEI_VisualFacet42.csv"),
        "official_split_file": str(SDK_SPLIT_FILE),
        "label_threshold": 0.5,
        "max_audio_tokens": 48,
        "max_text_tokens": 32,
        "max_vision_tokens": 24,
        "chunk_size": 200000,
        "cache": str(OUTPUT_ROOT / "cache" / "mosei_albef_inputs.npz"),
    },
    "alignment": {
        "seed": 42,
        "batch_size": 32,
        "epochs": 8,
        "lr": 3.0e-4,
        "weight_decay": 0.05,
        "patience": 3,
        "d_model": 128,
        "embed_dim": 128,
        "depth": 2,
        "nhead": 8,
        "mlp_ratio": 4.0,
        "dropout": 0.2,
        "queue_size": 2048,
        "momentum": 0.995,
        "temp_init": 0.07,
        "alpha": 0.4,
        "itm_weight": 0.3,
        "supervised_weight": 0.2,
        "checkpoint": str(OUTPUT_ROOT / "checkpoints" / "albef_alignment_best.pt"),
        "features": str(OUTPUT_ROOT / "cache" / "albef_aligned_features.npz"),
    },
    "fusion": {
        "epochs": 50,
        "batch_size": 8,
        "lr": 3.0e-4,
        "weight_decay": 1.0e-2,
        "patience": 10,
        "d_p": 128,
        "d_f": 128,
        "tau_r": 1.0,
        "tau_z": 0.07,
        "tau_q": 0.07,
        "lambda_d": 0.5,
        "lambda_cyc": 0.5,
        "anneal_epochs": 10,
        "lambda_cal": 0.5,
        "lambda_ce": 4.0,
        "lambda_uni": 0.5,
        "aux_warmup_epochs": 3,
    },
}


def setup_runtime() -> logging.Logger:
    base.OUTPUT_ROOT = OUTPUT_ROOT
    base.CLASS_NAMES = CLASS_NAMES
    base.NUM_CLASSES = len(CLASS_NAMES)
    logger = base.setup_logging()
    base.LOGGER = logger
    return logger


LOGGER = logging.getLogger(__name__)


def _feature_columns(path: Path) -> list[str]:
    return [c for c in pd.read_csv(path, nrows=0).columns if c.startswith("feature_")]


def _compact_modality(
    path: Path,
    labels_by_video: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_samples: int,
    max_tokens: int,
    chunk_size: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """Stream a video-contiguous CSV and uniformly sample each label interval."""
    feature_cols = _feature_columns(path)
    usecols = KEYS + feature_cols
    dtype: dict[str, Any] = {"video_id": "string", "start_time": np.float32, "end_time": np.float32}
    dtype.update({c: np.float32 for c in feature_cols})
    values = np.zeros((n_samples, max_tokens, len(feature_cols)), dtype=np.float32)
    valid = np.zeros((n_samples, max_tokens), dtype=np.bool_)
    finalized: set[str] = set()
    carry: pd.DataFrame | None = None
    rows_read = 0
    matched_frames = 0
    started = time.time()

    def consume(video_id: str, group: pd.DataFrame) -> None:
        nonlocal matched_frames
        if video_id in finalized:
            raise RuntimeError(f"{name} CSV is not video-contiguous; repeated video {video_id!r}")
        finalized.add(video_id)
        targets = labels_by_video.get(video_id)
        if targets is None:
            return
        indices, label_starts, label_ends = targets
        starts = group["start_time"].to_numpy(np.float32)
        ends = group["end_time"].to_numpy(np.float32)
        feats = group[feature_cols].to_numpy(np.float32, copy=False)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        for index, start, end in zip(indices, label_starts, label_ends):
            overlap = (starts < float(end) + 1e-6) & (ends > float(start) - 1e-6)
            selected = feats[overlap]
            if len(selected) > max_tokens:
                take = np.linspace(0, len(selected) - 1, max_tokens, dtype=np.int64)
                selected = selected[take]
            count = min(len(selected), max_tokens)
            if count:
                values[int(index), :count] = selected[:count]
                valid[int(index), :count] = True
                matched_frames += count

    reader = pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype,
        chunksize=chunk_size,
        low_memory=False,
        memory_map=True,
    )
    for chunk_no, chunk in enumerate(reader, start=1):
        rows_read += len(chunk)
        chunk["video_id"] = chunk["video_id"].astype(str)
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last_video = str(chunk["video_id"].iloc[-1])
        is_last = chunk["video_id"] == last_video
        complete = chunk.loc[~is_last]
        carry = chunk.loc[is_last].copy()
        for video_id, group in complete.groupby("video_id", sort=False):
            consume(str(video_id), group)
        if chunk_no == 1 or chunk_no % 20 == 0:
            LOGGER.info(
                "%s streaming chunk=%d rows=%s videos=%d elapsed=%.1f min",
                name, chunk_no, f"{rows_read:,}", len(finalized), (time.time() - started) / 60.0,
            )
    if carry is not None and not carry.empty:
        consume(str(carry["video_id"].iloc[0]), carry)
    empty = int((~valid.any(axis=1)).sum())
    LOGGER.info(
        "%s compacted shape=%s rows=%s matched_tokens=%s empty=%d time=%.1f min",
        name, values.shape, f"{rows_read:,}", f"{matched_frames:,}", empty, (time.time() - started) / 60.0,
    )
    return values, valid, feature_cols, rows_read


def _standardize_tokens(
    values: np.ndarray, valid: np.ndarray, train_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = values[valid & train_mask[:, None]]
    if not len(selected):
        raise RuntimeError("No valid training tokens found")
    mean = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = selected.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    del selected
    for first in range(0, len(values), 1024):
        last = min(first + 1024, len(values))
        block_valid = valid[first:last]
        block = values[first:last]
        block[block_valid] = np.clip((block[block_valid] - mean) / std, -10.0, 10.0)
    return values, mean, std


def prepare_cache(cfg: dict[str, Any], force: bool = False) -> Path:
    dc = cfg["data"]
    cache_path = Path(dc["cache"])
    if cache_path.exists() and not force:
        LOGGER.info("Using existing compact input cache: %s", cache_path)
        return cache_path
    started = time.time()
    labels = pd.read_csv(dc["label_csv"], low_memory=False)
    labels = labels[KEYS + ["feature_0"]].rename(columns={"feature_0": "opinion"})
    labels["video_id"] = labels["video_id"].astype(str)
    labels = labels.reset_index(drop=True)
    if labels.duplicated(KEYS).any():
        raise RuntimeError("MOSEI labels contain duplicate video/time windows")

    official = base.load_official_splits(Path(dc["official_split_file"]))
    official_ids = set().union(*official.values())
    source_utterances = len(labels)
    source_videos = int(labels["video_id"].nunique())
    outside_official = ~labels["video_id"].isin(official_ids)
    excluded_utterances = int(outside_official.sum())
    excluded_videos = int(labels.loc[outside_official, "video_id"].nunique())
    if excluded_utterances:
        LOGGER.warning(
            "Excluding %d utterances from %d videos absent from the SDK official folds",
            excluded_utterances, excluded_videos,
        )
        labels = labels.loc[~outside_official].reset_index(drop=True)
    labels["sample_id"] = [f"mosei-{i:05d}" for i in range(len(labels))]
    split = np.empty(len(labels), dtype="U5")
    for i, video_id in enumerate(labels["video_id"]):
        membership = [key for key, ids in official.items() if video_id in ids]
        if len(membership) != 1:
            raise RuntimeError(f"Invalid official split membership for {video_id!r}: {membership}")
        split[i] = membership[0]
    train_mask = split == "train"
    opinions = labels["opinion"].to_numpy(np.float32)
    y = base.label_three_class(opinions, float(dc["label_threshold"]))
    labels_by_video: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for video_id, group in labels.groupby("video_id", sort=False):
        labels_by_video[str(video_id)] = (
            group.index.to_numpy(np.int64),
            group["start_time"].to_numpy(np.float32),
            group["end_time"].to_numpy(np.float32),
        )

    arrays: dict[str, np.ndarray] = {}
    dimensions: dict[str, int] = {}
    empty_counts: dict[str, int] = {}
    source_rows: dict[str, int] = {}
    stats: dict[str, np.ndarray] = {}
    for name in ("audio", "text", "vision"):
        values, valid, columns, rows = _compact_modality(
            Path(dc[f"{name}_csv"]), labels_by_video, len(labels),
            int(dc[f"max_{name}_tokens"]), int(dc["chunk_size"]), name,
        )
        values, mean, std = _standardize_tokens(values, valid, train_mask)
        arrays[name] = values
        arrays[f"{name}_valid"] = valid
        stats[f"{name}_mean"] = mean
        stats[f"{name}_std"] = std
        dimensions[name] = len(columns)
        empty_counts[name] = int((~valid.any(axis=1)).sum())
        source_rows[name] = rows

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        sample_id=labels["sample_id"].to_numpy("U16"),
        video_id=labels["video_id"].to_numpy("U32"),
        start_time=labels["start_time"].to_numpy(np.float32),
        end_time=labels["end_time"].to_numpy(np.float32),
        opinion=opinions,
        labels=y,
        split=split,
        **arrays,
        **stats,
    )
    audit = {
        "n_utterances": len(labels),
        "n_videos": int(labels["video_id"].nunique()),
        "source_n_utterances": source_utterances,
        "source_n_videos": source_videos,
        "excluded_outside_official_folds": {
            "utterances": excluded_utterances,
            "videos": excluded_videos,
            "reason": "video IDs absent from the local CMU Multimodal SDK official fold constants",
        },
        "split_utterances": {s: int((split == s).sum()) for s in ("train", "val", "test")},
        "split_videos": {s: int(labels.loc[split == s, "video_id"].nunique()) for s in ("train", "val", "test")},
        "class_names": CLASS_NAMES,
        "class_threshold": float(dc["label_threshold"]),
        "class_counts": {name: int((y == i).sum()) for i, name in enumerate(CLASS_NAMES)},
        "split_class_counts": {
            s: {name: int(((split == s) & (y == i)).sum()) for i, name in enumerate(CLASS_NAMES)}
            for s in ("train", "val", "test")
        },
        "feature_dimensions": dimensions,
        "max_tokens": {name: int(dc[f"max_{name}_tokens"]) for name in ("audio", "text", "vision")},
        "empty_segments": empty_counts,
        "source_rows": source_rows,
        "official_split_source": str(dc["official_split_file"]),
        "standardization": "training-split valid tokens only; clipped to [-10, 10]",
        "seconds": time.time() - started,
    }
    base.write_json(OUTPUT_ROOT / "data_audit.json", audit)
    LOGGER.info("Prepared MOSEI compact cache in %.1f min: %s", (time.time() - started) / 60.0, cache_path)
    return cache_path


class AlignmentDataset(Dataset):
    def __init__(self, data: dict[str, np.ndarray], indices: np.ndarray):
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        i = int(self.indices[item])
        return {
            "global_id": torch.tensor(i, dtype=torch.long),
            "audio": torch.from_numpy(self.data["audio"][i]),
            "text": torch.from_numpy(self.data["text"][i]),
            "vision": torch.from_numpy(self.data["vision"][i]),
            "audio_pad": torch.from_numpy(~self.data["audio_valid"][i]),
            "text_pad": torch.from_numpy(~self.data["text_valid"][i]),
            "vision_pad": torch.from_numpy(~self.data["vision_valid"][i]),
            "label": torch.tensor(int(self.data["labels"][i]), dtype=torch.long),
            "opinion": torch.tensor(float(self.data["opinion"][i]), dtype=torch.float32),
        }


def build_alignment_model(data: dict[str, np.ndarray], cfg: dict[str, Any]) -> TriALBEFAlign:
    ac = cfg["alignment"]
    dc = cfg["data"]
    tri_cfg = TriCfg(
        embed_dim=int(ac["embed_dim"]), temp_init=float(ac["temp_init"]),
        queue_size=int(ac["queue_size"]), momentum=float(ac["momentum"]), distill=True,
        d_model=int(ac["d_model"]), depth=int(ac["depth"]), nhead=int(ac["nhead"]),
        mlp_ratio=float(ac["mlp_ratio"]), dropout=float(ac["dropout"]),
        max_tokens=max(int(dc["max_audio_tokens"]), int(dc["max_text_tokens"]), int(dc["max_vision_tokens"])),
        num_tokens_a=int(dc["max_audio_tokens"]), num_tokens_b=int(dc["max_text_tokens"]),
        num_tokens_c=int(dc["max_vision_tokens"]), input_type_a="sequence",
        input_type_b="sequence", input_type_c="sequence", label_dim=1,
    )
    return TriALBEFAlign(data["audio"].shape[-1], data["text"].shape[-1], data["vision"].shape[-1], tri_cfg)


def alignment_forward(model, batch, device, cfg):
    b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    same_class = b["label"].view(-1, 1).eq(b["label"].view(1, -1))
    out = model(
        b["audio"], b["text"], b["vision"], alpha=float(cfg["alignment"]["alpha"]),
        idx=b["global_id"], mask_a=b["audio_pad"], mask_b=b["text_pad"],
        mask_c=b["vision_pad"], targets=b["opinion"], soft_pos_mask=same_class,
    )
    loss = out["loss_itc"] + float(cfg["alignment"]["itm_weight"]) * out["loss_itm"]
    loss = loss + float(cfg["alignment"]["supervised_weight"]) * out["loss_sup"]
    return loss, out


@torch.no_grad()
def extract_aligned_features(cfg, data, device, force=False) -> Path:
    output = Path(cfg["alignment"]["features"])
    if output.exists() and not force:
        LOGGER.info("Using existing aligned features: %s", output)
        return output
    model = build_alignment_model(data, cfg).to(device)
    state = torch.load(cfg["alignment"]["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    loader = DataLoader(
        AlignmentDataset(data, np.arange(len(data["labels"]))), batch_size=64,
        shuffle=False, num_workers=0, pin_memory=device.type == "cuda",
    )
    outputs = {"g_audio": [], "g_text": [], "g_vision": []}
    for batch in loader:
        audio, text, vision = (batch[k].to(device) for k in ("audio", "text", "vision"))
        am, tm, vm = (batch[k].to(device) for k in ("audio_pad", "text_pad", "vision_pad"))
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            at, _ = model.enc_a(audio, key_padding_mask=am)
            tt, _ = model.enc_b(text, key_padding_mask=tm)
            vt, _ = model.enc_c(vision, key_padding_mask=vm)
            outputs["g_audio"].append(F.normalize(model.proj_a(at[:, 0]), dim=-1).float().cpu().numpy())
            outputs["g_text"].append(F.normalize(model.proj_b(tt[:, 0]), dim=-1).float().cpu().numpy())
            outputs["g_vision"].append(F.normalize(model.proj_c(vt[:, 0]), dim=-1).float().cpu().numpy())
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **{key: np.concatenate(value) for key, value in outputs.items()},
        **{key: data[key] for key in ("sample_id", "video_id", "start_time", "end_time", "opinion", "labels", "split")},
    )
    LOGGER.info("Extracted aligned MOSEI features: %s", output)
    return output


def _md_table(df: pd.DataFrame, columns: list[str], names: list[str]) -> str:
    lines = ["| " + " | ".join(names) + " |", "|" + "---|" * len(names)]
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                cells.append("N/A" if pd.isna(value) else f"{float(value):.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
