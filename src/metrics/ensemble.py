"""Probability-ensemble utilities for independently trained classifiers."""

from __future__ import annotations

import numpy as np


def categorical_entropy(probabilities: np.ndarray, axis: int = -1) -> np.ndarray:
    """Categorical entropy in nats, with ``0 log 0`` treated as zero."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim == 0:
        raise ValueError("probabilities must have at least one dimension")
    if not np.all(np.isfinite(p)):
        raise ValueError("probabilities contain non-finite values")
    if np.any(p < -1e-8) or np.any(p > 1.0 + 1e-8):
        raise ValueError("probabilities must lie in [0, 1]")
    p = np.clip(p, 0.0, 1.0)
    return -(np.where(p > 0.0, p * np.log(np.clip(p, 1e-12, 1.0)), 0.0)).sum(axis=axis)


def equal_weight_probability_ensemble(
    member_probabilities: np.ndarray,
    row_sum_atol: float = 1e-3,
) -> dict[str, np.ndarray]:
    """Average member probabilities and decompose predictive uncertainty.

    Parameters
    ----------
    member_probabilities:
        Array shaped ``(n_members, n_samples, n_classes)``. No member weights
        are accepted deliberately: the formal P1 protocol fixes equal weights
        and must not tune them on the test set.

    Returns
    -------
    ``p_ens`` is the arithmetic mean. Predictive entropy is
    ``H[p_ens]``; mutual information is ``H[p_ens] - mean_m H[p_m]``.
    These are ensemble uncertainty measures and do not reinterpret averaged
    probabilities as a single Dirichlet distribution.
    """
    probs = np.asarray(member_probabilities, dtype=np.float64)
    if probs.ndim != 3:
        raise ValueError(
            "member_probabilities must have shape (members, samples, classes)"
        )
    if probs.shape[0] < 2 or probs.shape[1] == 0 or probs.shape[2] < 2:
        raise ValueError("need at least 2 members, 1 sample and 2 classes")
    if not np.all(np.isfinite(probs)):
        raise ValueError("member probabilities contain non-finite values")
    if np.any(probs < -1e-8) or np.any(probs > 1.0 + 1e-8):
        raise ValueError("member probabilities must lie in [0, 1]")
    sums = probs.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=row_sum_atol, rtol=0.0):
        worst = float(np.max(np.abs(sums - 1.0)))
        raise ValueError(f"probability rows do not sum to 1 (max error={worst:.6g})")

    # CSV formatting can introduce tiny row-sum errors; normalise only after
    # validating that the rows are already probabilities within tolerance.
    probs = np.clip(probs, 0.0, 1.0)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    p_ens = probs.mean(axis=0)
    predictive_entropy = categorical_entropy(p_ens)
    member_entropy = categorical_entropy(probs)
    expected_entropy = member_entropy.mean(axis=0)
    mutual_information = np.maximum(predictive_entropy - expected_entropy, 0.0)
    normalizer = np.log(probs.shape[-1])
    return {
        "p_ens": p_ens,
        "predictive_entropy": predictive_entropy,
        "predictive_entropy_normalized": predictive_entropy / normalizer,
        "expected_member_entropy": expected_entropy,
        "mutual_information": mutual_information,
        "member_probabilities": probs,
    }
