"""Small end-to-end contracts for both distinct CMU preprocessing paths."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize("dataset", ["mosi", "mosei"])
def test_cmu_prepare_align_train_and_independent_evaluate(tmp_path, dataset):
    data_dir = tmp_path / dataset
    data_dir.mkdir()
    rng = np.random.default_rng(9)
    splits = {"train": ["v0", "v1"], "val": ["v2", "v3"], "test": ["v4", "v5"]}
    labels, audio, text, vision = [], [], [], []
    for video in sum(splits.values(), []):
        for i in range(3):
            key = {"video_id": video, "start_time": float(i), "end_time": float(i + 1)}
            labels.append({**key, "feature_0": float(i - 1)})
            for destination, dim in [(audio, 16), (text, 8), (vision, 8)]:
                destination.append({**key, **{f"feature_{j}": value for j, value in enumerate(rng.standard_normal(dim))}})
    prefix = "CMU_" + dataset.upper()
    names = (["Opinion_Labels", "OpenSmile_EB10", "TimestampedWordVectors", "Visual_Facet_42"]
             if dataset == "mosi" else ["Labels", "COVAREP", "TimestampedWordVectors", "VisualFacet42"])
    for name, rows in zip(names, [labels, audio, text, vision]):
        pd.DataFrame(rows).to_csv(data_dir / f"{prefix}_{name}.csv", index=False)
    (data_dir / "official_splits.json").write_text(json.dumps(splits), encoding="utf-8")
    project = Path(__file__).resolve().parents[1]
    root = tmp_path / "output"
    common = [sys.executable, "scripts/run_cmu.py", "--dataset", dataset, "--data-dir", str(data_dir),
              "--out-dir", str(root), "--device", "cpu"]
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    changes = ["alignment.epochs=1", "alignment.batch_size=2", "alignment.queue_size=8",
               "alignment.d_model=16", "alignment.embed_dim=16", "alignment.depth=1", "alignment.nhead=2",
               "fusion.epochs=1", "fusion.batch_size=2", "fusion.d_p=8", "fusion.d_f=16"]
    result = subprocess.run(common + ["--stage", "align", "--set", *changes], cwd=project,
                            env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run(common + ["--stage", "train", "--set", *changes], cwd=project,
                            env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    run = root / "seed42/main"
    assert (run / "validation_predictions.npz").exists()
    assert not (run / "test_predictions.npz").exists()
    result = subprocess.run(common + ["--stage", "evaluate"], cwd=project,
                            env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    with np.load(run / "test_predictions.npz", allow_pickle=False) as saved:
        assert len(saved["unit_ids"]) == 6
        assert set(saved["video_id"]) == {"v4", "v5"}
        assert saved["p_hat"].shape == (6, 3)
        assert np.isfinite(saved["p_hat"]).all()
        np.testing.assert_allclose(saved["p_hat"].sum(1), 1, atol=1e-5)
