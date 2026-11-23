import json

import torch

from scripts.s3e_seq_eval import load_effective_config
from src.utils.evidential import aggregate_dirichlet_evidence


def test_sequence_aggregation_counts_unit_prior_once():
    # Three prior-only time steps still represent one prior-only sequence.
    alpha = torch.ones(3, 4)
    got = aggregate_dirichlet_evidence(alpha, dim=0)
    assert torch.equal(got, torch.ones(4))
    assert torch.isclose(4.0 / got.sum(), torch.tensor(1.0))


def test_sequence_aggregation_adds_only_evidence():
    alpha = torch.tensor(
        [[2.0, 1.0, 4.0], [1.5, 3.0, 1.0], [1.0, 1.0, 2.0]]
    )
    got = aggregate_dirichlet_evidence(alpha, dim=0)
    expected = torch.tensor([2.5, 3.0, 5.0])
    assert torch.allclose(got, expected)

    # Removing the repeated uniform prior does not change argmax.
    assert got.argmax() == alpha.sum(0).argmax()


def test_sequence_aggregation_preserves_batch_axis():
    alpha = torch.ones(2, 3, 5)
    alpha[0, :, 2] += torch.tensor([1.0, 2.0, 3.0])
    got = aggregate_dirichlet_evidence(alpha, dim=1)
    assert got.shape == (2, 5)
    assert torch.equal(got[0], torch.tensor([1.0, 1.0, 7.0, 1.0, 1.0]))
    assert torch.equal(got[1], torch.ones(5))


def test_sequence_aggregation_ignores_masked_padding_evidence():
    alpha = torch.tensor([
        [[2.0, 1.0], [1.0, 4.0], [99.0, 99.0]],
        [[1.0, 3.0], [88.0, 88.0], [77.0, 77.0]],
    ])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)
    got = aggregate_dirichlet_evidence(alpha, dim=1, mask=mask)
    expected = torch.tensor([[2.0, 4.0], [1.0, 3.0]])
    assert torch.equal(got, expected)


def test_sequence_eval_prefers_saved_run_config(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("model:\n  use_semantic_stream: true\n", encoding="utf-8")
    seed_dir = tmp_path / "runs" / "simplify" / "S3" / "seed42"
    fold_dir = seed_dir / "fold0"
    fold_dir.mkdir(parents=True)
    saved = {"model": {"use_semantic_stream": False}}
    (seed_dir / "config.json").write_text(json.dumps(saved), encoding="utf-8")

    cfg, source = load_effective_config(str(base), str(fold_dir / "best.pt"))
    assert cfg["model"]["use_semantic_stream"] is False
    assert source == str(seed_dir / "config.json")
