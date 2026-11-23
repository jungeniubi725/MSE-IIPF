import csv

import numpy as np
import pytest

from scripts.s8_ensemble_seeds import align_members
from src.metrics.ensemble import equal_weight_probability_ensemble


def _write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq_id", "true", "pred", "u", "p0", "p1"])
        writer.writerows(rows)


def test_alignment_uses_seq_id_not_row_order(tmp_path):
    a, b, c = (tmp_path / f"s{x}.csv" for x in range(3))
    _write(a, [["id_b", 1, 1, 0.2, 0.1, 0.9],
               ["id_a", 0, 0, 0.2, 0.8, 0.2]])
    _write(b, [["id_a", 0, 0, 0.3, 0.7, 0.3],
               ["id_b", 1, 1, 0.3, 0.2, 0.8]])
    _write(c, [["id_b", 1, 1, 0.4, 0.3, 0.7],
               ["id_a", 0, 0, 0.4, 0.6, 0.4]])

    ids, labels, members, cols = align_members([a, b, c])
    assert ids == ["id_a", "id_b"]
    assert labels.tolist() == [0, 1]
    assert cols == ["p0", "p1"]
    assert np.allclose(members[:, 0, 0], [0.8, 0.7, 0.6])


def test_alignment_rejects_missing_id(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _write(a, [["id_a", 0, 0, 0.2, 0.8, 0.2],
               ["id_b", 1, 1, 0.2, 0.1, 0.9]])
    _write(b, [["id_a", 0, 0, 0.2, 0.7, 0.3]])
    with pytest.raises(ValueError, match="seq_id set mismatch"):
        align_members([a, b])


def test_equal_weight_average_and_uncertainty_decomposition():
    members = np.asarray([
        [[0.9, 0.1], [0.5, 0.5]],
        [[0.6, 0.4], [0.5, 0.5]],
        [[0.3, 0.7], [0.5, 0.5]],
    ])
    out = equal_weight_probability_ensemble(members)
    assert np.allclose(out["p_ens"], [[0.6, 0.4], [0.5, 0.5]])
    assert out["mutual_information"][0] > 0
    assert out["mutual_information"][1] == pytest.approx(0.0, abs=1e-12)
    assert np.all(out["predictive_entropy_normalized"] <= 1.0 + 1e-12)


def test_probability_ensemble_rejects_non_probabilities():
    bad = np.asarray([[[0.8, 0.8]], [[0.5, 0.5]]])
    with pytest.raises(ValueError, match="do not sum to 1"):
        equal_weight_probability_ensemble(bad)
