"""CMU-MOSI: CSV -> ALBEF alignment -> IIPF -> evidential classifier.

This experiment intentionally bypasses SG-DSFE.  It consumes the local CSV
features directly, uses the ALBEF-main tri-modal alignment implementation,
then trains the repository IIPF and hard-correctness evidential head.

All artifacts are written below CMU_MOSI_OUTPUT.  The script is self-contained
apart from importing the two in-repository model implementations.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import math
import os
import platform
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset


WORKSPACE = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = WORKSPACE / "runs" / "cmu_mosi"
DATA_ROOT = WORKSPACE / "data" / "cmu_mosi"
SDK_SPLIT_FILE = DATA_ROOT / "official_splits.json"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from src.models.tri_albef_align import TriALBEFAlign, TriCfg
from src.metrics.agreement import modality_agreement  # noqa: E402
from src.metrics.classification import classification_metrics, per_class_metrics  # noqa: E402
from src.metrics.uncertainty import uncertainty_report  # noqa: E402
from src.models.euc_head import EvidentialHead  # noqa: E402
from src.models.iipf import IIPF  # noqa: E402


CLASS_NAMES = ["Negative", "Neutral", "Positive"]
NUM_CLASSES = len(CLASS_NAMES)
KEYS = ["video_id", "start_time", "end_time"]
SEEDS = [42, 43, 44]


DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "label_csv": str(DATA_ROOT / "CMU_MOSI_Opinion_Labels.csv"),
        "audio_csv": str(DATA_ROOT / "CMU_MOSI_OpenSmile_EB10.csv"),
        "text_csv": str(DATA_ROOT / "CMU_MOSI_TimestampedWordVectors.csv"),
        "vision_csv": str(DATA_ROOT / "CMU_MOSI_Visual_Facet_42.csv"),
        "official_split_file": str(SDK_SPLIT_FILE),
        "label_threshold": 0.5,
        "max_text_tokens": 32,
        "max_vision_tokens": 32,
        "cache": str(OUTPUT_ROOT / "cache" / "mosi_albef_inputs.npz"),
    },
    "alignment": {
        "seed": 42,
        "batch_size": 32,
        "epochs": 12,
        "lr": 3.0e-4,
        "weight_decay": 0.05,
        "patience": 4,
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


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items() if k != "_raw"}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")


def setup_logging() -> logging.Logger:
    (OUTPUT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mosi_bypass")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(OUTPUT_ROOT / "logs" / "experiment.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_official_splits(path: Path) -> dict[str, set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    splits = {key: set(data[key]) for key in ("train", "val", "test")}
    if any(splits[a] & splits[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Official split membership overlaps")
    return splits


def label_three_class(values: np.ndarray, threshold: float) -> np.ndarray:
    y = np.full(len(values), 1, dtype=np.int64)
    y[values <= -threshold] = 0
    y[values >= threshold] = 2
    return y


def _with_occurrence(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_occurrence"] = out.groupby(KEYS, sort=False).cumcount()
    return out


def _select_interval_tokens(
    frame_df: pd.DataFrame,
    labels: pd.DataFrame,
    max_tokens: int,
    modality: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_cols = [c for c in frame_df.columns if c not in KEYS]
    feat_dim = len(feature_cols)
    grouped: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for video_id, group in frame_df.groupby("video_id", sort=False):
        grouped[str(video_id)] = (
            group["start_time"].to_numpy(np.float32),
            group["end_time"].to_numpy(np.float32),
            group[feature_cols].to_numpy(np.float32),
        )
    del frame_df

    values = np.zeros((len(labels), max_tokens, feat_dim), dtype=np.float32)
    valid = np.zeros((len(labels), max_tokens), dtype=np.bool_)
    empty_ids: list[str] = []
    for i, row in enumerate(labels.itertuples(index=False)):
        video_id = str(row.video_id)
        starts, ends, feats = grouped.get(
            video_id,
            (np.empty(0, np.float32), np.empty(0, np.float32), np.empty((0, feat_dim), np.float32)),
        )
        overlap = (starts < float(row.end_time) + 1e-6) & (ends > float(row.start_time) - 1e-6)
        selected = feats[overlap]
        if len(selected) == 0:
            empty_ids.append(f"{video_id}:{row.start_time:.6f}-{row.end_time:.6f}")
            continue
        if len(selected) > max_tokens:
            indices = np.linspace(0, len(selected) - 1, max_tokens, dtype=np.int64)
            selected = selected[indices]
        n = len(selected)
        values[i, :n] = selected
        valid[i, :n] = True
    LOGGER.info("%s tokens: shape=%s empty=%d", modality, values.shape, len(empty_ids))
    return values, valid, feature_cols


def _standardize_vector(x: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mean = x[train].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x[train].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    out = np.clip((x - mean) / std, -10.0, 10.0).astype(np.float32)
    return out, mean, std


def _standardize_tokens(
    x: np.ndarray, valid: np.ndarray, train: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    train_valid = valid & train[:, None]
    flat = x[train_valid]
    mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = flat.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    out = np.zeros_like(x, dtype=np.float32)
    out[valid] = np.clip((x[valid] - mean) / std, -10.0, 10.0)
    return out, mean, std


def prepare_cache(cfg: dict[str, Any], force: bool = False) -> Path:
    data_cfg = cfg["data"]
    cache_path = Path(data_cfg["cache"])
    if cache_path.exists() and not force:
        LOGGER.info("Using existing input cache: %s", cache_path)
        return cache_path

    start = time.time()
    labels = pd.read_csv(data_cfg["label_csv"])
    labels = labels.rename(columns={"feature_0": "opinion"})
    labels = _with_occurrence(labels[KEYS + ["opinion"]])
    labels["sample_id"] = [f"mosi-{i:04d}" for i in range(len(labels))]

    audio = pd.read_csv(data_cfg["audio_csv"])
    audio = _with_occurrence(audio)
    audio_cols = [c for c in audio.columns if c.startswith("feature_")]
    audio = labels[KEYS + ["_occurrence"]].merge(
        audio[KEYS + ["_occurrence"] + audio_cols],
        on=KEYS + ["_occurrence"],
        how="left",
        validate="one_to_one",
    )
    if audio[audio_cols].isna().all(axis=1).any():
        raise RuntimeError("Some opinion segments have no EB10 feature row")
    audio_values = audio[audio_cols].to_numpy(np.float32)
    del audio

    text_df = pd.read_csv(data_cfg["text_csv"])
    text_values, text_valid, text_cols = _select_interval_tokens(
        text_df, labels, int(data_cfg["max_text_tokens"]), "text"
    )
    del text_df
    vision_df = pd.read_csv(data_cfg["vision_csv"])
    vision_values, vision_valid, vision_cols = _select_interval_tokens(
        vision_df, labels, int(data_cfg["max_vision_tokens"]), "vision"
    )
    del vision_df

    official = load_official_splits(Path(data_cfg["official_split_file"]))
    split = np.empty(len(labels), dtype="U5")
    for i, video_id in enumerate(labels["video_id"].astype(str)):
        hits = [name for name, ids in official.items() if video_id in ids]
        if len(hits) != 1:
            raise RuntimeError(f"Video {video_id!r} has invalid official split membership: {hits}")
        split[i] = hits[0]
    train_mask = split == "train"

    audio_values, audio_mean, audio_std = _standardize_vector(audio_values, train_mask)
    text_values, text_mean, text_std = _standardize_tokens(text_values, text_valid, train_mask)
    vision_values, vision_mean, vision_std = _standardize_tokens(vision_values, vision_valid, train_mask)
    opinions = labels["opinion"].to_numpy(np.float32)
    y = label_three_class(opinions, float(data_cfg["label_threshold"]))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        sample_id=labels["sample_id"].to_numpy("U16"),
        video_id=labels["video_id"].astype(str).to_numpy("U32"),
        start_time=labels["start_time"].to_numpy(np.float32),
        end_time=labels["end_time"].to_numpy(np.float32),
        opinion=opinions,
        labels=y,
        split=split,
        audio=audio_values,
        text=text_values,
        vision=vision_values,
        text_valid=text_valid,
        vision_valid=vision_valid,
        audio_mean=audio_mean,
        audio_std=audio_std,
        text_mean=text_mean,
        text_std=text_std,
        vision_mean=vision_mean,
        vision_std=vision_std,
    )

    audit = {
        "n_utterances": len(labels),
        "n_videos": int(labels["video_id"].nunique()),
        "duplicate_time_windows_retained_by_occurrence": int(labels.duplicated(KEYS, keep=False).sum()),
        "split_utterances": {name: int((split == name).sum()) for name in ("train", "val", "test")},
        "split_videos": {
            name: int(labels.loc[split == name, "video_id"].nunique()) for name in ("train", "val", "test")
        },
        "class_names": CLASS_NAMES,
        "class_threshold": float(data_cfg["label_threshold"]),
        "class_counts": {
            name: int((y == i).sum()) for i, name in enumerate(CLASS_NAMES)
        },
        "feature_dimensions": {
            "audio_openSMILE_EB10": len(audio_cols),
            "text_timestamped_word_vectors": len(text_cols),
            "vision_FACET_4_2": len(vision_cols),
        },
        "max_tokens": {
            "text": int(data_cfg["max_text_tokens"]),
            "vision": int(data_cfg["max_vision_tokens"]),
        },
        "empty_text_segments": int((~text_valid.any(axis=1)).sum()),
        "empty_vision_segments": int((~vision_valid.any(axis=1)).sum()),
        "official_split_source": str(data_cfg["official_split_file"]),
        "seconds": time.time() - start,
    }
    write_json(OUTPUT_ROOT / "data_audit.json", audit)
    LOGGER.info("Prepared %d utterances in %.1fs: %s", len(labels), time.time() - start, cache_path)
    return cache_path


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


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
            "text_pad": torch.from_numpy(~self.data["text_valid"][i]),
            "vision_pad": torch.from_numpy(~self.data["vision_valid"][i]),
            "label": torch.tensor(int(self.data["labels"][i]), dtype=torch.long),
            "opinion": torch.tensor(float(self.data["opinion"][i]), dtype=torch.float32),
        }


def build_alignment_model(data: dict[str, np.ndarray], cfg: dict[str, Any]) -> TriALBEFAlign:
    acfg = cfg["alignment"]
    model_cfg = TriCfg(
        embed_dim=int(acfg["embed_dim"]),
        temp_init=float(acfg["temp_init"]),
        queue_size=int(acfg["queue_size"]),
        momentum=float(acfg["momentum"]),
        distill=True,
        d_model=int(acfg["d_model"]),
        depth=int(acfg["depth"]),
        nhead=int(acfg["nhead"]),
        mlp_ratio=float(acfg["mlp_ratio"]),
        dropout=float(acfg["dropout"]),
        max_tokens=max(int(cfg["data"]["max_text_tokens"]), int(cfg["data"]["max_vision_tokens"]), 16),
        num_tokens_a=16,
        num_tokens_b=int(cfg["data"]["max_text_tokens"]),
        num_tokens_c=int(cfg["data"]["max_vision_tokens"]),
        input_type_a="vector",
        input_type_b="sequence",
        input_type_c="sequence",
        label_dim=1,
    )
    return TriALBEFAlign(
        dim_a=int(data["audio"].shape[-1]),
        dim_b=int(data["text"].shape[-1]),
        dim_c=int(data["vision"].shape[-1]),
        cfg=model_cfg,
    )


def _alignment_forward(
    model: TriALBEFAlign,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    same_class = b["label"].view(-1, 1).eq(b["label"].view(1, -1))
    out = model(
        b["audio"],
        b["text"],
        b["vision"],
        alpha=float(cfg["alignment"]["alpha"]),
        idx=b["global_id"],
        mask_b=b["text_pad"],
        mask_c=b["vision_pad"],
        targets=b["opinion"],
        soft_pos_mask=same_class,
    )
    loss = (
        out["loss_itc"]
        + float(cfg["alignment"]["itm_weight"]) * out["loss_itm"]
        + float(cfg["alignment"]["supervised_weight"]) * out["loss_sup"]
    )
    return loss, out


def train_alignment(
    cfg: dict[str, Any],
    data: dict[str, np.ndarray],
    device: torch.device,
    force: bool = False,
) -> Path:
    acfg = cfg["alignment"]
    checkpoint = Path(acfg["checkpoint"])
    if checkpoint.exists() and not force:
        LOGGER.info("Using existing alignment checkpoint: %s", checkpoint)
        return checkpoint
    seed_everything(int(acfg["seed"]))
    train_idx = np.where(data["split"] == "train")[0]
    val_idx = np.where(data["split"] == "val")[0]
    batch_size = int(acfg["batch_size"])
    if int(acfg["queue_size"]) % batch_size != 0:
        raise ValueError("alignment.queue_size must be divisible by alignment.batch_size")
    generator = torch.Generator().manual_seed(int(acfg["seed"]))
    train_loader = DataLoader(
        AlignmentDataset(data, train_idx),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        AlignmentDataset(data, val_idx),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = build_alignment_model(data, cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(acfg["lr"]), weight_decay=float(acfg["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(acfg["epochs"])), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, float]] = []
    best_val = float("inf")
    bad_epochs = 0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(acfg["epochs"]) + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss, _ = _alignment_forward(model, batch, device, cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach()))
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    loss, _ = _alignment_forward(model, batch, device, cfg)
                val_losses.append(float(loss.detach()))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_losses)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "temperature": float(model.temp.detach().cpu()),
        }
        history.append(row)
        LOGGER.info(
            "ALBEF epoch %02d train=%.4f val=%.4f temp=%.4f",
            epoch,
            row["train_loss"],
            row["val_loss"],
            row["temperature"],
        )
        if row["val_loss"] < best_val - 1e-5:
            best_val = row["val_loss"]
            bad_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "tri_cfg": asdict(model.cfg),
                    "epoch": epoch,
                    "val_loss": best_val,
                    "config": cfg,
                },
                checkpoint,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= int(acfg["patience"]):
                LOGGER.info("ALBEF early stopping at epoch %d", epoch)
                break
    write_json(OUTPUT_ROOT / "logs" / "alignment_history.json", history)
    return checkpoint


@torch.no_grad()
def extract_aligned_features(
    cfg: dict[str, Any],
    data: dict[str, np.ndarray],
    device: torch.device,
    force: bool = False,
) -> Path:
    output = Path(cfg["alignment"]["features"])
    if output.exists() and not force:
        LOGGER.info("Using existing aligned features: %s", output)
        return output
    checkpoint = Path(cfg["alignment"]["checkpoint"])
    model = build_alignment_model(data, cfg).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    loader = DataLoader(
        AlignmentDataset(data, np.arange(len(data["labels"]))),
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    audio_out, text_out, vision_out = [], [], []
    for batch in loader:
        audio = batch["audio"].to(device)
        text = batch["text"].to(device)
        vision = batch["vision"].to(device)
        text_pad = batch["text_pad"].to(device)
        vision_pad = batch["vision_pad"].to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            a_tok, _ = model.enc_a(audio)
            t_tok, _ = model.enc_b(text, key_padding_mask=text_pad)
            v_tok, _ = model.enc_c(vision, key_padding_mask=vision_pad)
            ga = F.normalize(model.proj_a(a_tok[:, 0]), dim=-1)
            gt = F.normalize(model.proj_b(t_tok[:, 0]), dim=-1)
            gv = F.normalize(model.proj_c(v_tok[:, 0]), dim=-1)
        audio_out.append(ga.float().cpu().numpy())
        text_out.append(gt.float().cpu().numpy())
        vision_out.append(gv.float().cpu().numpy())
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        g_audio=np.concatenate(audio_out),
        g_text=np.concatenate(text_out),
        g_vision=np.concatenate(vision_out),
        sample_id=data["sample_id"],
        video_id=data["video_id"],
        start_time=data["start_time"],
        end_time=data["end_time"],
        opinion=data["opinion"],
        labels=data["labels"],
        split=data["split"],
    )
    LOGGER.info("Extracted aligned features: %s", output)
    return output


class VideoSequenceDataset(Dataset):
    def __init__(self, data: dict[str, np.ndarray], split: str):
        self.data = data
        indices = np.where(data["split"] == split)[0]
        groups: dict[str, list[int]] = defaultdict(list)
        for i in indices:
            groups[str(data["video_id"][i])].append(int(i))
        self.groups = []
        for video_id in sorted(groups):
            ordered = sorted(groups[video_id], key=lambda i: (float(data["start_time"][i]), i))
            self.groups.append((video_id, np.asarray(ordered, dtype=np.int64)))

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, item: int) -> dict[str, Any]:
        video_id, idx = self.groups[item]
        return {
            "video_id": video_id,
            "indices": idx,
            "g_v": torch.from_numpy(self.data["g_vision"][idx]),
            "g_a": torch.from_numpy(self.data["g_audio"][idx]),
            "g_l": torch.from_numpy(self.data["g_text"][idx]),
            "labels": torch.from_numpy(self.data["labels"][idx]),
            "opinion": torch.from_numpy(self.data["opinion"][idx]),
        }


def collate_sequences(batch: list[dict[str, Any]]) -> dict[str, Any]:
    bsz = len(batch)
    max_t = max(len(x["indices"]) for x in batch)
    d = int(batch[0]["g_v"].shape[-1])
    out = {
        "g_v": torch.zeros((bsz, max_t, d), dtype=torch.float32),
        "g_a": torch.zeros((bsz, max_t, d), dtype=torch.float32),
        "g_l": torch.zeros((bsz, max_t, d), dtype=torch.float32),
        "labels": torch.zeros((bsz, max_t), dtype=torch.long),
        "opinion": torch.zeros((bsz, max_t), dtype=torch.float32),
        "mask_t": torch.zeros((bsz, max_t), dtype=torch.bool),
        "indices": torch.full((bsz, max_t), -1, dtype=torch.long),
        "video_id": [],
    }
    for i, sample in enumerate(batch):
        n = len(sample["indices"])
        for key in ("g_v", "g_a", "g_l", "labels", "opinion"):
            out[key][i, :n] = sample[key]
        out["mask_t"][i, :n] = True
        out["indices"][i, :n] = torch.from_numpy(sample["indices"])
        out["video_id"].append(sample["video_id"])
    return out


class BypassIIPFClassifier(nn.Module):
    def __init__(
        self,
        d: int,
        cfg: dict[str, Any],
        *,
        d_p: int | None = None,
        tau_r: float | None = None,
        use_iipf: bool = True,
        use_cmc: bool = True,
        use_tpc: bool = True,
        use_prr: bool = True,
        use_proxy_weight: bool = True,
        use_evidential: bool = True,
        use_cal: bool = True,
    ) -> None:
        super().__init__()
        fcfg = cfg["fusion"]
        self.use_iipf = use_iipf
        self.use_evidential = use_evidential
        self.d_f = int(fcfg["d_f"])
        if use_iipf:
            self.iipf = IIPF(
                d=d,
                d_p=int(d_p if d_p is not None else fcfg["d_p"]),
                d_f=self.d_f,
                num_classes=NUM_CLASSES,
                tau_r=float(tau_r if tau_r is not None else fcfg["tau_r"]),
                tau_z=float(fcfg["tau_z"]),
                tau_q=float(fcfg["tau_q"]),
                lambda_d=float(fcfg["lambda_d"]),
                lambda_cyc=float(fcfg["lambda_cyc"]),
                use_cmc=use_cmc,
                use_tpc=use_tpc,
                use_prr=use_prr,
                use_proxy_weight=use_proxy_weight,
            )
        else:
            self.direct_gate = nn.Sequential(nn.Linear(d, 1))
            self.direct_proj = nn.Identity() if d == self.d_f else nn.Linear(d, self.d_f)
        if use_evidential:
            self.head = EvidentialHead(
                d_f=self.d_f,
                d_modal=d,
                num_classes=NUM_CLASSES,
                anneal_epochs=int(fcfg["anneal_epochs"]),
                lambda_cal=float(fcfg["lambda_cal"]) if use_cal else 0.0,
                lambda_ce=float(fcfg["lambda_ce"]),
                lambda_uni=float(fcfg["lambda_uni"]),
                lambda_seq=0.0,
            )
        else:
            self.softmax_head = nn.Linear(self.d_f, NUM_CLASSES)
            self.modal_v = nn.Linear(d, NUM_CLASSES)
            self.modal_a = nn.Linear(d, NUM_CLASSES)
            self.modal_l = nn.Linear(d, NUM_CLASSES)

    def forward(
        self,
        g_v: torch.Tensor,
        g_a: torch.Tensor,
        g_l: torch.Tensor,
        y: torch.Tensor,
        mask_t: torch.Tensor,
        epoch: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if self.use_iipf:
            fused, aux = self.iipf(g_v, g_a, g_l, mask_t)
            f = fused["f"]
        else:
            stack = torch.stack((g_v, g_a, g_l), dim=2)
            weights = self.direct_gate(stack).softmax(dim=2)
            f = self.direct_proj((weights * stack).sum(dim=2))
            zero = f.new_zeros(())
            fused, aux = {"f": f}, {"L_CMC": zero, "L_TPC": zero, "L_PRR": zero}
        if self.use_evidential:
            out, head_losses = self.head(f, g_v, g_a, g_l, y, epoch, mask_t, None)
        else:
            logits = self.softmax_head(f)
            p = logits.softmax(dim=-1)
            p_v = self.modal_v(g_v).softmax(dim=-1)
            p_a = self.modal_a(g_a).softmax(dim=-1)
            p_l = self.modal_l(g_l).softmax(dim=-1)
            keep = mask_t
            ce = F.cross_entropy(logits[keep], y[keep])
            flat_y = y[keep].unsqueeze(1)
            uni = torch.stack(
                [
                    -pm[keep]
                    .clamp_min(1e-8)
                    .log()
                    .gather(1, flat_y)
                    .mean()
                    for pm in (p_v, p_a, p_l)
                ]
            ).mean()
            out = {
                "p_hat": p,
                "y_pred": p.argmax(dim=-1),
                "u": 1.0 - p.max(dim=-1).values,
                "r": p.max(dim=-1).values,
                "p_v": p_v,
                "p_a": p_a,
                "p_l": p_l,
            }
            head_losses = {"L_head": ce + 0.5 * uni, "L_CE": ce, "L_uni": uni}
        return {**fused, **out}, {**aux, **head_losses}


class BaselineClassifier(nn.Module):
    def __init__(self, d: int, mode: str):
        super().__init__()
        self.mode = mode
        self.modal_v = nn.Linear(d, NUM_CLASSES)
        self.modal_a = nn.Linear(d, NUM_CLASSES)
        self.modal_l = nn.Linear(d, NUM_CLASSES)
        if mode == "early_concat":
            self.fusion = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Dropout(0.2), nn.Linear(d, NUM_CLASSES))
        elif mode == "attention":
            self.attn = nn.Linear(d, 1)
            self.fusion = nn.Linear(d, NUM_CLASSES)
        elif mode in {"text_only", "mean_fusion", "late_fusion"}:
            self.fusion = nn.Linear(d, NUM_CLASSES)
        else:
            raise ValueError(mode)

    def forward(
        self,
        g_v: torch.Tensor,
        g_a: torch.Tensor,
        g_l: torch.Tensor,
        y: torch.Tensor,
        mask_t: torch.Tensor,
        epoch: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        logits_v, logits_a, logits_l = self.modal_v(g_v), self.modal_a(g_a), self.modal_l(g_l)
        p_v, p_a, p_l = logits_v.softmax(-1), logits_a.softmax(-1), logits_l.softmax(-1)
        if self.mode == "text_only":
            logits = self.fusion(g_l)
            p = logits.softmax(-1)
        elif self.mode == "early_concat":
            logits = self.fusion(torch.cat((g_v, g_a, g_l), dim=-1))
            p = logits.softmax(-1)
        elif self.mode == "mean_fusion":
            logits = self.fusion((g_v + g_a + g_l) / 3.0)
            p = logits.softmax(-1)
        elif self.mode == "attention":
            stack = torch.stack((g_v, g_a, g_l), dim=2)
            weights = self.attn(stack).softmax(dim=2)
            logits = self.fusion((weights * stack).sum(dim=2))
            p = logits.softmax(-1)
        else:
            p = (p_v + p_a + p_l) / 3.0
            logits = p.clamp_min(1e-8).log()
        keep = mask_t
        loss = F.nll_loss(p[keep].clamp_min(1e-8).log(), y[keep])
        uni = torch.stack(
            [F.cross_entropy(x[keep], y[keep]) for x in (logits_v, logits_a, logits_l)]
        ).mean()
        confidence = p.max(dim=-1).values
        return {
            "p_hat": p,
            "y_pred": p.argmax(dim=-1),
            "u": 1.0 - confidence,
            "r": confidence,
            "p_v": p_v,
            "p_a": p_a,
            "p_l": p_l,
        }, {"L_head": loss + 0.5 * uni, "L_CE": loss, "L_uni": uni}


def make_sequence_loaders(
    data: dict[str, np.ndarray], cfg: dict[str, Any], seed: int, include_test: bool = True
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": int(cfg["fusion"]["batch_size"]),
        "num_workers": 0,
        "collate_fn": collate_sequences,
    }
    train = DataLoader(VideoSequenceDataset(data, "train"), shuffle=True, generator=generator, **common)
    val = DataLoader(VideoSequenceDataset(data, "val"), shuffle=False, **common)
    test = DataLoader(VideoSequenceDataset(data, "test"), shuffle=False, **common) if include_test else None
    return train, val, test


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate_fusion(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    collect_raw: bool = True,
) -> dict[str, Any]:
    model.eval()
    buffers: dict[str, list[np.ndarray]] = defaultdict(list)
    losses = []
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out, loss_dict = model(
                batch["g_v"], batch["g_a"], batch["g_l"], batch["labels"], batch["mask_t"], epoch
            )
        keep = batch["mask_t"].bool()
        losses.append(float(loss_dict["L_head"].detach()))
        buffers["y"].append(batch["labels"][keep].cpu().numpy())
        buffers["opinion"].append(batch["opinion"][keep].cpu().numpy())
        buffers["index"].append(batch["indices"][keep].cpu().numpy())
        buffers["yp"].append(out["y_pred"][keep].cpu().numpy())
        buffers["p"].append(out["p_hat"][keep].float().cpu().numpy())
        buffers["u"].append(out["u"][keep].float().cpu().numpy())
        for name, key in (("pv", "p_v"), ("pa", "p_a"), ("pl", "p_l")):
            buffers[name].append(out[key][keep].argmax(dim=-1).cpu().numpy())
    cat = {k: np.concatenate(v) for k, v in buffers.items()}
    y, yp, p = cat["y"], cat["yp"], cat["p"]
    result: dict[str, Any] = {
        "loss": float(np.mean(losses)),
        "n_units": int(len(y)),
        **classification_metrics(y, yp, NUM_CLASSES),
        "per_class": per_class_metrics(y, yp, NUM_CLASSES, CLASS_NAMES),
    }
    result["uncertainty"] = uncertainty_report(y, yp, p, cat["u"], None)
    result["agreement"] = modality_agreement(cat["pv"], cat["pa"], cat["pl"], yp, NUM_CLASSES)
    non_neutral = y != 1
    binary_pred = (p[non_neutral, 2] >= p[non_neutral, 0]).astype(np.int64)
    binary_true = (y[non_neutral] == 2).astype(np.int64)
    result["binary_non_neutral"] = classification_metrics(binary_true, binary_pred, 2)
    result["confusion_count"] = confusion_matrix(y, yp, labels=list(range(NUM_CLASSES)))
    cm = result["confusion_count"].astype(np.float64)
    result["confusion_normalized"] = cm / np.clip(cm.sum(axis=1, keepdims=True), 1.0, None)
    if collect_raw:
        result["_raw"] = cat
    return result


def train_fusion_run(
    name: str,
    model_factory,
    data: dict[str, np.ndarray],
    cfg: dict[str, Any],
    seed: int,
    device: torch.device,
    force: bool = False,
    evaluate_test: bool = True,
) -> dict[str, Any]:
    run_dir = OUTPUT_ROOT / name
    metrics_path = run_dir / ("test_metrics.json" if evaluate_test else "validation_metrics.json")
    if metrics_path.exists() and not force:
        LOGGER.info("Using existing run: %s seed=%d", name, seed)
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
        return saved
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    train_loader, val_loader, test_loader = make_sequence_loaders(data, cfg, seed, include_test=evaluate_test)
    model = model_factory().to(device)
    fcfg = cfg["fusion"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(fcfg["lr"]), weight_decay=float(fcfg["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(fcfg["epochs"])), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_val, best_epoch, bad_epochs = -float("inf"), 0, 0
    history: list[dict[str, Any]] = []
    best_path = run_dir / "best.pt"
    for epoch in range(1, int(fcfg["epochs"]) + 1):
        model.train()
        epoch_losses = []
        for raw_batch in train_loader:
            batch = _to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                _, losses = model(
                    batch["g_v"], batch["g_a"], batch["g_l"], batch["labels"], batch["mask_t"], epoch
                )
                loss = losses["L_head"]
                if "L_CMC" in losses:
                    warm = min(1.0, epoch / max(1, int(fcfg["aux_warmup_epochs"])))
                    loss = loss + warm * (losses["L_CMC"] + losses["L_TPC"] + losses["L_PRR"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach()))
        scheduler.step()
        val = evaluate_fusion(model, val_loader, device, epoch, collect_raw=False)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_macro_f1": val["macro_f1"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        score = float(val["macro_f1"])
        if score > best_val + 1e-5:
            best_val, best_epoch, bad_epochs = score, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_macro_f1": score, "cfg": cfg, "variant": name, "seed": seed}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= int(fcfg["patience"]):
                break
    state = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    test = evaluate_fusion(model, test_loader if evaluate_test else val_loader, device, best_epoch, collect_raw=True)
    raw = test.pop("_raw")
    test.update({"run": name, "seed": seed, "best_epoch": best_epoch, "best_val_macro_f1": best_val})
    write_json(run_dir / "history.json", history)
    write_json(metrics_path, test)
    predictions = pd.DataFrame(
        {
            "global_index": raw["index"],
            "y_true": raw["y"],
            "y_pred": raw["yp"],
            "opinion": raw["opinion"],
            "p_negative": raw["p"][:, 0],
            "p_neutral": raw["p"][:, 1],
            "p_positive": raw["p"][:, 2],
            "uncertainty": raw["u"],
            "pred_visual": raw["pv"],
            "pred_audio": raw["pa"],
            "pred_text": raw["pl"],
        }
    ).sort_values("global_index")
    predictions.to_csv(run_dir / ("predictions.csv" if evaluate_test else "validation_predictions.csv"), index=False, encoding="utf-8-sig")
    LOGGER.info(
        "%s seed=%d best_epoch=%d test_acc=%.4f test_macro_f1=%.4f",
        name,
        seed,
        best_epoch,
        test["acc"],
        test["macro_f1"],
    )
    return test
