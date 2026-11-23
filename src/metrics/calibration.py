"""Post-hoc multiclass calibration and selective-risk metrics."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

from .uncertainty import expected_calibration_error


def _validate_probabilities(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.int64).ravel()
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != y.size or p.shape[1] < 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes)")
    if y.size == 0 or np.any(y < 0) or np.any(y >= p.shape[1]):
        raise ValueError("invalid or empty labels")
    if not np.all(np.isfinite(p)) or np.any(p < -1e-8):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-3, rtol=0.0):
        raise ValueError("probability rows do not sum to 1")
    p = np.clip(p, 0.0, None)
    p /= p.sum(axis=1, keepdims=True)
    return y, p


def multiclass_nll(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean negative log likelihood of the true class."""
    y, p = _validate_probabilities(y_true, probabilities)
    return float(-np.log(np.clip(p[np.arange(y.size), y], 1e-12, 1.0)).mean())


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Multiclass Brier score ``mean_i sum_c (p_ic-y_ic)^2``."""
    y, p = _validate_probabilities(y_true, probabilities)
    target = np.eye(p.shape[1], dtype=np.float64)[y]
    return float(np.square(p - target).sum(axis=1).mean())


def aurc_eaurc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
) -> dict[str, float]:
    """Finite-sample AURC and excess AURC under confidence ranking.

    AURC is the mean cumulative error rate while coverage grows from ``1/n``
    to one. The finite-sample optimum places all correct predictions before
    all errors; E-AURC subtracts that attainable optimum for the same error
    count.
    """
    y = np.asarray(y_true).ravel()
    pred = np.asarray(y_pred).ravel()
    conf = np.asarray(confidence, dtype=np.float64).ravel()
    if y.size == 0 or pred.size != y.size or conf.size != y.size:
        raise ValueError("y_true, y_pred and confidence must have equal nonzero length")
    if not np.all(np.isfinite(conf)):
        raise ValueError("confidence contains non-finite values")
    error = (pred != y).astype(np.float64)
    # Stable sort makes tied-confidence behavior deterministic.
    order = np.argsort(-conf, kind="mergesort")
    risk = np.cumsum(error[order]) / np.arange(1, y.size + 1)
    n_error = int(error.sum())
    optimal_error = np.r_[np.zeros(y.size - n_error), np.ones(n_error)]
    optimal_risk = np.cumsum(optimal_error) / np.arange(1, y.size + 1)
    aurc = float(risk.mean())
    optimal = float(optimal_risk.mean())
    return {
        "aurc": aurc,
        "aurc_optimal": optimal,
        "eaurc": max(0.0, aurc - optimal),
    }


def probability_quality_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 15
) -> dict[str, float]:
    """ECE, NLL, Brier, AURC and E-AURC for one probability matrix."""
    y, p = _validate_probabilities(y_true, probabilities)
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    return {
        **expected_calibration_error(y, pred, conf, n_bins=n_bins),
        "nll": multiclass_nll(y, p),
        "brier": multiclass_brier(y, p),
        **aurc_eaurc(y, pred, conf),
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=-1, keepdims=True)


def probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("probabilities must be two-dimensional")
    return np.log(np.clip(p, 1e-12, 1.0))


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply scalar temperature to log probabilities; argmax is invariant."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    return _softmax(probabilities_to_logits(probabilities) / float(temperature))


def fit_temperature(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bounds: tuple[float, float] = (0.05, 20.0),
) -> dict[str, float]:
    """Fit one positive temperature by validation NLL only."""
    y, p = _validate_probabilities(y_true, probabilities)
    lo, hi = bounds
    if not (0 < lo < hi):
        raise ValueError("temperature bounds must satisfy 0 < low < high")

    def objective(log_t: float) -> float:
        return multiclass_nll(y, apply_temperature(p, float(np.exp(log_t))))

    result = minimize_scalar(
        objective, bounds=(float(np.log(lo)), float(np.log(hi))),
        method="bounded", options={"xatol": 1e-7},
    )
    candidates = [(1.0, objective(0.0)), (float(np.exp(result.x)), float(result.fun))]
    temperature, nll = min(candidates, key=lambda x: x[1])
    return {
        "temperature": float(temperature),
        "validation_nll": float(nll),
        "optimizer_success": bool(result.success),
    }


def apply_logit_adjustment(
    probabilities: np.ndarray,
    class_prior: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Apply ``z'_c = z_c - tau * log(pi_c)`` to log probabilities."""
    p = np.asarray(probabilities, dtype=np.float64)
    prior = np.asarray(class_prior, dtype=np.float64).ravel()
    if p.ndim != 2 or prior.size != p.shape[1]:
        raise ValueError("class_prior length must match probability classes")
    if not np.all(np.isfinite(prior)) or np.any(prior <= 0):
        raise ValueError("class_prior must be finite and strictly positive")
    prior = prior / prior.sum()
    return _softmax(probabilities_to_logits(p) - float(tau) * np.log(prior)[None, :])


def fit_logit_adjustment_tau(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_prior: np.ndarray,
    tau_grid: np.ndarray | None = None,
) -> dict:
    """Select one non-negative ``tau`` by validation macro-F1.

    Ties prefer the smaller tau, then lower validation NLL. The returned
    curve makes the complete one-dimensional validation search auditable.
    """
    y, p = _validate_probabilities(y_true, probabilities)
    grid = np.asarray(
        np.linspace(0.0, 2.0, 41) if tau_grid is None else tau_grid,
        dtype=np.float64,
    ).ravel()
    if grid.size == 0 or not np.all(np.isfinite(grid)) or np.any(grid < 0):
        raise ValueError("tau_grid must contain finite non-negative values")
    grid = np.unique(grid)
    curve = []
    for tau in grid:
        adjusted = apply_logit_adjustment(p, class_prior, float(tau))
        macro_f1 = float(f1_score(
            y, adjusted.argmax(axis=1), average="macro", zero_division=0,
            labels=list(range(p.shape[1])),
        ))
        curve.append({
            "tau": float(tau),
            "macro_f1": macro_f1,
            "nll": multiclass_nll(y, adjusted),
        })
    # Round only for tie detection; exact floating noise must not select a
    # larger correction when predictions and macro-F1 are identical.
    best = min(curve, key=lambda r: (-round(r["macro_f1"], 12), r["tau"], r["nll"]))
    return {
        "tau": best["tau"],
        "validation_macro_f1": best["macro_f1"],
        "validation_nll_before_temperature": best["nll"],
        "grid": curve,
        "selected_at_upper_boundary": bool(best["tau"] == float(grid.max())),
    }
