"""Run the actual CMU cached-feature ALBEF -> IIPF experiment, one seed at a time."""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.artifacts import environment_info, export_predictions, save_json
from src.utils.config import apply_overrides

ABLATIONS = {"main": {}, "no_iipf": {"use_iipf": False}, "no_cmc": {"use_cmc": False},
             "no_tpc": {"use_tpc": False}, "no_prr": {"use_prr": False},
             "no_proxy_weight": {"use_proxy_weight": False}, "softmax": {"use_evidential": False},
             "no_cal": {"use_cal": False}}
BASELINES = ["text_only", "early_concat", "mean_fusion", "late_fusion", "attention"]


def model_factory(base, cfg, dimension, variant):
    if variant in BASELINES:
        return lambda: base.BaselineClassifier(dimension, variant)
    return lambda: base.BypassIIPFClassifier(dimension, cfg, **ABLATIONS[variant])


def export_split(base, model, data, cfg, device, epoch, split, run_dir):
    loader = DataLoader(base.VideoSequenceDataset(data, split), shuffle=False,
                        batch_size=int(cfg["fusion"]["batch_size"]), num_workers=0, collate_fn=base.collate_sequences)
    result = base.evaluate_fusion(model, loader, device, epoch, collect_raw=True)
    raw = result.pop("_raw")
    index = raw["index"].astype(int)
    arrays = {"y_true": raw["y"], "y_pred": raw["yp"], "p_hat": raw["p"], "u": raw["u"],
              "unit_ids": data["sample_id"][index].astype(str), "video_id": data["video_id"][index].astype(str),
              "start_time": data["start_time"][index], "end_time": data["end_time"][index],
              "opinion": raw["opinion"], "pred_v": raw["pv"], "pred_a": raw["pa"], "pred_l": raw["pl"]}
    prefix = "validation" if split == "val" else split
    export_predictions(arrays, run_dir, prefix)
    save_json(run_dir / f"{prefix}_metrics.json", {**result, "sample_unit": "utterance", "split": split,
              "class_names": base.CLASS_NAMES, "architecture": "cached_features_ALBEF_IIPF",
              "uncertainty_kind": "1-max_probability" if cfg["variant"] in BASELINES + ["softmax"] else "K/S"})
    print(f"{prefix}: n={len(index)} accuracy={result['acc']:.4f} macro-F1={result['macro_f1']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mosi", "mosei"], required=True)
    parser.add_argument("--stage", choices=["prepare", "align", "train", "evaluate", "all"], default="all")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", choices=list(ABLATIONS) + BASELINES, default="main")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides", help="For example alignment.epochs=12 fusion.epochs=50")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.stage == "evaluate" and args.overrides:
        parser.error("Evaluation uses the stored training configuration; --set is not allowed")
    module = importlib.import_module(f"experiments.cmu.run_{args.dataset}_bypass_experiment")
    base = module if args.dataset == "mosi" else module.base
    data_dir = (args.data_dir or Path("data") / ("cmu_" + args.dataset)).resolve()
    root = (args.out_dir or Path("runs") / ("cmu_" + args.dataset)).resolve() / f"seed{args.seed}"
    root.mkdir(parents=True, exist_ok=True)
    base.OUTPUT_ROOT = root
    base.LOGGER = base.setup_logging()
    if args.dataset == "mosei":
        module.OUTPUT_ROOT = root
        module.LOGGER = base.LOGGER
        base.AlignmentDataset = module.AlignmentDataset
        base.build_alignment_model = module.build_alignment_model
        base._alignment_forward = module.alignment_forward
    cfg = copy.deepcopy(module.DEFAULT_CONFIG)
    for key in ("label_csv", "audio_csv", "text_csv", "vision_csv"):
        cfg["data"][key] = str(data_dir / Path(cfg["data"][key]).name)
    cfg["data"]["official_split_file"] = str(data_dir / "official_splits.json")
    cfg["data"]["cache"] = str(root / "cache" / "inputs.npz")
    cfg["alignment"].update(seed=args.seed, checkpoint=str(root / "alignment" / "best.pt"),
                              features=str(root / "cache" / "aligned_features.npz"))
    cfg = apply_overrides(cfg, args.overrides)
    cfg["variant"], cfg["seed"], cfg["dataset"] = args.variant, args.seed, args.dataset
    run_dir = root / args.variant
    checkpoint = run_dir / "best.pt"
    if args.stage == "evaluate":
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        cfg = state["cfg"]
        # Aligned features can be relocated together with the complete output directory.
        cfg["alignment"]["features"] = str(root / "cache" / "aligned_features.npz")
    else:
        alignment_record = root / "alignment_config.json"
        alignment_cfg = {"data": cfg["data"], "alignment": cfg["alignment"]}
        if alignment_record.exists() and json.loads(alignment_record.read_text(encoding="utf-8")) != alignment_cfg:
            raise ValueError("This seed directory already belongs to a different alignment/data configuration. Use a new --out-dir.")
        save_json(alignment_record, alignment_cfg)
        save_json(root / "environment.json", environment_info())
        base.seed_everything(args.seed)
        if args.stage in ("prepare", "all", "align"):
            path = module.prepare_cache(cfg, force=False)
            if args.stage == "prepare":
                return 0
        if args.stage in ("align", "all"):
            data = base.load_npz(Path(cfg["data"]["cache"]))
            base.train_alignment(cfg, data, torch.device(args.device), force=False)
            module.extract_aligned_features(cfg, data, torch.device(args.device), force=False)
            if args.stage == "align":
                return 0
    features_path = Path(cfg["alignment"]["features"])
    if not features_path.exists():
        raise FileNotFoundError(f"Run --stage align first: {features_path}")
    data = base.load_npz(features_path)
    factory = model_factory(base, cfg, int(data["g_audio"].shape[-1]), args.variant)
    device = torch.device(args.device)
    if args.stage in ("train", "all"):
        if checkpoint.exists():
            raise FileExistsError(f"Existing trained checkpoint: {checkpoint}; use --stage evaluate or a new output directory")
        save_json(run_dir / "config.json", cfg)
        # Checkpoint selection and hyperparameter runs read validation, never a test loader.
        base.train_fusion_run(args.variant, factory, data, cfg, args.seed, device, evaluate_test=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = factory().to(device)
    model.load_state_dict(state["model"])
    base.seed_everything(args.seed)
    export_split(base, model, data, cfg, device, state["epoch"], "val", run_dir)
    if args.stage in ("all", "evaluate"):
        export_split(base, model, data, cfg, device, state["epoch"], "test", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

