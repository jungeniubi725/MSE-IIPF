"""Exercise train -> saved checkpoint -> independent evaluation -> plots on tiny fixtures.

The generated random fixtures and their metrics are software checks, not research results.
No external data, pretrained weights or network are used.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import safe_name
from src.utils.config import load_config
from src.utils.artifacts import save_json


def make_fixture(root):
    root = Path(root).resolve()
    features = root / "features"
    for branch in ("visual", "audio", "text_sv", "text_sa"):
        (features / branch).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(31)
    rows = []
    for split, n, actor in [("train", 6, 1), ("val", 3, 21), ("test", 3, 23)]:
        for i in range(n):
            sid = f"ravdess/Actor_{actor:02d}/fixture_{i}"
            for branch in ("text_sv", "text_sa"):
                np.save(features / branch / (safe_name(sid)+".npy"), rng.standard_normal((3, 8)).astype("float32"))
            for unit in range(2):
                uid = f"{sid}#{unit}"
                for branch in ("visual", "audio"):
                    np.save(features / branch / (safe_name(uid)+".npy"), rng.standard_normal((3, 8)).astype("float32"))
                rows.append({"seq_id": sid, "unit_id": uid, "unit_index": unit, "dataset": "ravdess",
                             "split": split, "fold": 0 if split == "train" else None, "speaker": f"Actor_{actor:02d}",
                             "label": i % 3, "seq_label": i % 3})
    manifest = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs/ravdess.yaml"))
    cfg["data"].update(manifest=str(manifest), feat_dir=str(features), num_classes=3, class_names=["a", "b", "c"], max_units=2)
    cfg["model"].update(d_visual=8, d_audio=8, d_text=8, d=16, d_p=8, d_f=16, num_heads=2)
    cfg["train"].update(epochs=1, batch_size=2, num_workers=0, amp=False, scheduler="none")
    config_path = root / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    save_json(root / "FIXTURE_ONLY.json", {"purpose": "software smoke test", "research_results": False, "random_seed": 31})
    return config_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("runs/smoke"))
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    out = args.out_dir.resolve()
    if (out / "training" / "best.pt").exists():
        p.error("Smoke output already contains a checkpoint; choose a new --out-dir")
    cfg = make_fixture(out)
    project = Path(__file__).resolve().parents[1]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", MPLBACKEND="Agg")
    commands = [
        ["scripts/s3_train.py", "--config", str(cfg), "--out-root", str(out), "--exp-name", "training", "--validation-only", "--device", args.device],
        ["scripts/evaluate.py", "--checkpoint", str(out / "training/best.pt"), "--out-dir", str(out / "evaluation"), "--device", args.device],
        ["scripts/plot_results.py", "--predictions", str(out / "evaluation/test_predictions.npz"), "--out-dir", str(out / "figures"), "--history", str(out / "training/history.json")],
    ]
    for command in commands:
        subprocess.run([sys.executable, *command], cwd=project, env=env, check=True)
    with np.load(out / "evaluation/test_predictions.npz", allow_pickle=False) as predictions:
        assert len(predictions["unit_ids"]) == 6
        assert len(predictions["seq_ids"]) == 3
        assert np.isfinite(predictions["p_hat"]).all()
    save_json(out / "SMOKE_TEST_PASSED.json", {"passed": True, "research_results": False, "commands": commands})
    print("Smoke test passed. These fixture metrics are not experimental results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
