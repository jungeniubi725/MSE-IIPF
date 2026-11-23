from types import SimpleNamespace

from scripts import s3_train


def test_validation_only_loader_does_not_construct_test(monkeypatch):
    calls = []

    class FakeDataset:
        def __init__(self, manifest, **kwargs):
            calls.append(kwargs)
            self.missing = {}

        def __len__(self):
            return 8

    def fake_loader(dataset, **kwargs):
        return SimpleNamespace(dataset=dataset)

    monkeypatch.setattr(s3_train, "MERCDataset", FakeDataset)
    monkeypatch.setattr(s3_train, "DataLoader", fake_loader)
    monkeypatch.setattr(s3_train, "make_collate", lambda *args: None)

    cfg = {
        "data": {"manifest": "manifest.csv", "feat_dir": "feats"},
        "model": {"d_visual": 2, "d_audio": 2, "d_text": 2},
        "train": {"batch_size": 2, "num_workers": 0},
    }
    loaders = s3_train.build_loaders(cfg, fold=0, include_test=False)

    assert set(loaders) == {"train", "val"}
    assert not any(call.get("fold_role") == "test" for call in calls)
    assert not any(call.get("split") == "test" for call in calls)

