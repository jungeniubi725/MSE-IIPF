import numpy as np
import pytest

from src.metrics.calibration import (
    apply_logit_adjustment,
    apply_temperature,
    aurc_eaurc,
    fit_temperature,
    fit_logit_adjustment_tau,
    multiclass_brier,
    multiclass_nll,
)


def test_temperature_preserves_argmax_and_can_reduce_nll():
    y = np.array([0, 1, 0, 1])
    p = np.array([[0.99, 0.01], [0.01, 0.99], [0.1, 0.9], [0.9, 0.1]])
    fit = fit_temperature(y, p)
    scaled = apply_temperature(p, fit["temperature"])
    assert np.array_equal(p.argmax(1), scaled.argmax(1))
    assert multiclass_nll(y, scaled) <= multiclass_nll(y, p) + 1e-10


def test_multiclass_brier_known_value():
    y = np.array([0, 1])
    p = np.array([[1.0, 0.0], [0.25, 0.75]])
    assert multiclass_brier(y, p) == pytest.approx(0.0625)


def test_aurc_is_optimal_when_all_errors_rank_last():
    y = np.array([0, 0, 0, 0])
    pred = np.array([0, 0, 1, 1])
    good = aurc_eaurc(y, pred, np.array([0.9, 0.8, 0.2, 0.1]))
    bad = aurc_eaurc(y, pred, np.array([0.2, 0.1, 0.9, 0.8]))
    assert good["eaurc"] == pytest.approx(0.0)
    assert bad["eaurc"] > good["eaurc"]


def test_logit_adjustment_is_identity_at_zero():
    p = np.array([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]])
    prior = np.array([0.6, 0.3, 0.1])
    assert np.allclose(apply_logit_adjustment(p, prior, 0.0), p)


def test_tau_search_uses_one_scalar_and_prefers_zero_on_tie():
    y = np.array([0, 1, 2])
    p = np.eye(3) * 0.8 + (1.0 - np.eye(3)) * 0.1
    fit = fit_logit_adjustment_tau(
        y, p, np.array([0.6, 0.3, 0.1]), np.array([0.0, 0.5, 1.0])
    )
    assert fit["tau"] == 0.0
    assert len(fit["grid"]) == 3
