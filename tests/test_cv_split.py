import numpy as np
import pandas as pd

from src.data.dataset import MERCDataset


def _row(seq_id, speaker, split, fold, label):
    return {
        "seq_id": seq_id,
        "unit_id": seq_id + "#0",
        "unit_index": 0,
        "label": label,
        "seq_label": label,
        "speaker": speaker,
        "split": split,
        "fold": fold,
    }


def test_cv_train_excludes_outer_fold_and_fixed_val_test(tmp_path):
    df = pd.DataFrame([
        _row("outer", "Actor_01", "train", 0.0, 0),
        _row("inner", "Actor_05", "train", 1.0, 1),
        _row("fixed_val", "Actor_21", "val", np.nan, 2),
        _row("fixed_test", "Actor_23", "test", np.nan, 3),
    ])
    kwargs = {"feat_dir": str(tmp_path), "strict": False}
    train = MERCDataset(df, fold=0, fold_role="train", **kwargs)
    test = MERCDataset(df, fold=0, fold_role="test", **kwargs)
    val = MERCDataset(df, split="val", **kwargs)

    assert [x["seq_id"] for x in train.groups] == ["inner"]
    assert [x["seq_id"] for x in test.groups] == ["outer"]
    assert [x["seq_id"] for x in val.groups] == ["fixed_val"]
