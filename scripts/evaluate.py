"""Evaluate a saved RAVDESS checkpoint without retraining or checkpoint selection."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.s3_train import build_loaders, build_model
from src.engine.evaluator import evaluate, format_metrics
from src.utils.artifacts import export_predictions, save_json
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--fold", type=int, choices=range(5), help="Use the same outer fold as training")
    parser.add_argument("--manifest", help="Relocate the manifest without changing the model config")
    parser.add_argument("--feat-dir", help="Relocate cached features")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = state["cfg"]
    if args.manifest:
        cfg["data"]["manifest"] = args.manifest
    if args.feat_dir:
        cfg["data"]["feat_dir"] = args.feat_dir
    # Fold is recorded beside every packaged training checkpoint.
    protocol_path = args.checkpoint.parent / "protocol.json"
    import json
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else {}
    recorded_fold = protocol.get("fold")
    if args.fold is not None and recorded_fold is not None and args.fold != recorded_fold:
        parser.error("--fold differs from the checkpoint's recorded training fold")
    fold = recorded_fold if args.fold is None else args.fold
    set_seed(int(cfg["train"].get("seed", 42)))
    loader = build_loaders(cfg, fold=fold, evaluation=True, only_split=args.split)[args.split]
    model = build_model(cfg).to(args.device)
    model.load_state_dict(state["model"])
    result = evaluate(model, loader, args.device, cfg["data"]["num_classes"],
                      cfg["data"].get("class_names"), int(state["epoch"]), cfg["train"].get("amp", True))
    destination = args.out_dir or args.checkpoint.parent / "evaluation" / args.split
    prefix = "validation" if args.split == "val" else args.split
    export_predictions(result["_raw"], destination, prefix)
    save_json(destination / f"{prefix}_metrics.json", result)
    save_json(destination / "evaluation.json", {"checkpoint": str(args.checkpoint), "split": args.split,
              "fold": fold, "epoch": state["epoch"], "config": cfg})
    print(format_metrics(result, args.split + " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

