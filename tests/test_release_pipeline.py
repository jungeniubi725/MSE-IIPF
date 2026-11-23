from pathlib import Path
import io
import json
import zipfile

import h5py
import numpy as np
import pandas as pd
import pytest

from scripts.prepare_cmu import convert_csd
from src.utils.artifacts import export_predictions
from src.utils.downloads import extract_ravdess, parse_sdk_splits
from scripts.package_release import build_archive
from scripts.s2_extract_feats import _read_jsonl


def test_downloaded_split_code_is_not_executed():
    source = "standard_train_fold=['a']\nstandard_valid_fold=['b']\nstandard_test_fold=['c']\nraise RuntimeError('must not execute')"
    assert parse_sdk_splits(source) == {"train": ["a"], "val": ["b"], "test": ["c"]}
    with pytest.raises(ValueError, match="Overlapping"):
        parse_sdk_splits(source.replace("['c']", "['a']"))


def test_ravdess_extraction_filters_and_rejects_escape(tmp_path):
    path = tmp_path / "media.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("Actor_01/01-01-01-01-01-01-01.mp4", b"fixture")
        z.writestr("Actor_01/02-01-01-01-01-01-01.mp4", b"unused")
        z.writestr("Actor_01/03-01-01-01-01-01-01.wav", b"fixture")
    assert extract_ravdess(path, tmp_path / "data") == 2
    assert len(list((tmp_path / "data").rglob("*.*"))) == 2
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("../escape.py", b"x")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_ravdess(path, tmp_path / "data")


def test_csd_conversion_preserves_timestamps_and_duplicates(tmp_path):
    source, target = tmp_path / "fixture.csd", tmp_path / "fixture.csv"
    with h5py.File(source, "w") as f:
        data = f.create_group("features/data/video-a")
        data.create_dataset("intervals", data=[[0, 1], [0, 1], [1, 2]])
        data.create_dataset("features", data=[[1, 2], [3, 4], [5, 6]])
    assert convert_csd(source, target) == 3
    data = pd.read_csv(target)
    assert data.video_id.tolist() == ["video-a"] * 3
    assert data.start_time.tolist() == [0, 0, 1]
    assert data.feature_1.tolist() == [2, 4, 6]


def test_prediction_export_retains_ids_and_rejects_duplicates(tmp_path):
    raw = {"unit_ids": np.array(["video#0", "video#1"]), "y_true": np.array([1, 0]),
           "p_hat": np.array([[0.2, 0.8], [0.7, 0.3]]), "u": np.array([0.3, 0.5])}
    export_predictions(raw, tmp_path)
    data = pd.read_csv(tmp_path / "test_unit_predictions.csv")
    assert data.unit_id.tolist() == ["video#0", "video#1"]
    assert data.y_pred.tolist() == [1, 0]
    raw["unit_ids"] = np.array(["duplicate", "duplicate"])
    with pytest.raises(ValueError, match="duplicate"):
        export_predictions(raw, tmp_path / "invalid")


def test_source_archive_excludes_weights_results_and_cache(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "README.md").write_text("# Test project", encoding="utf-8")
    for relative in ["src/__init__.py", "src/data/__init__.py", "src/data/dataset.py",
                     ".github/workflows/tests.yml", "weights/model.pt", "data/manifest.json",
                     "runs/test_metrics.json", "src/__pycache__/code.pyc"]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    archive = tmp_path / "release.zip"
    build_archive(root, archive)
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {"MERC-GitHub/README.md", "MERC-GitHub/src/__init__.py",
                                    "MERC-GitHub/src/data/__init__.py", "MERC-GitHub/src/data/dataset.py",
                                    "MERC-GitHub/.github/workflows/tests.yml", "MERC-GitHub/SHA256SUMS.txt"}


def test_text_retry_uses_latest_success_and_rejects_dummy(tmp_path):
    path = tmp_path / "captions.jsonl"
    records = [{"id": "a", "error": "decode failure", "text": "fallback"},
               {"id": "a", "backend": "llava_next", "text": "real caption"}]
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    assert _read_jsonl(str(path)) == {"a": "real caption"}
    path.write_text(json.dumps({"id": "a", "backend": "dummy", "text": "placeholder"}), encoding="utf-8")
    with pytest.raises(ValueError, match="dummy"):
        _read_jsonl(str(path))


def test_feature_export_loader_keeps_incomplete_final_batch(tmp_path):
    from scripts.smoke_test import make_fixture
    from scripts.s3_train import build_loaders
    from src.utils.config import load_config
    cfg = load_config(str(make_fixture(tmp_path)))
    cfg["train"]["batch_size"] = 4
    loader = build_loaders(cfg, evaluation=True, only_split="train")["train"]
    ids = [sid for batch in loader for sid in batch["seq_ids"]]
    assert len(ids) == 6
    assert len(set(ids)) == 6

