"""E2 — Shared metric functions.

Pure numpy. No sklearn/scipy dependency, so this module runs anywhere.

Class order is fixed as ("H", "D", "A") throughout the E2 experiment.
"""
from __future__ import annotations

import numpy as np

CLASS_ORDER = ("H", "D", "A")
EPS = 1e-15


def to_onehot(y: np.ndarray) -> np.ndarray:
    """Convert array of 'H'/'D'/'A' labels to (n, 3) one-hot."""
    out = np.zeros((len(y), 3), dtype=float)
    for i, cls in enumerate(CLASS_ORDER):
        out[:, i] = (y == cls).astype(float)
    return out


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Multiclass log loss. p is (n, 3) in CLASS_ORDER."""
    oh = to_onehot(y)
    p_clip = np.clip(p, EPS, 1.0)
    # renormalize after clipping
    p_clip = p_clip / p_clip.sum(axis=1, keepdims=True)
    return float(-np.mean(np.sum(oh * np.log(p_clip), axis=1)))


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    """Multiclass Brier score (mean squared error over all 3 classes)."""
    oh = to_onehot(y)
    return float(np.mean(np.sum((p - oh) ** 2, axis=1)))


def rps(y: np.ndarray, p: np.ndarray) -> float:
    """Ranked Probability Score for ordered outcomes H < D < A.

    RPS = (1/(r-1)) * sum_{i=1}^{r-1} (cumsum(p)_i - cumsum(o)_i)^2
    with r = 3 categories.
    """
    oh = to_onehot(y)
    cp = np.cumsum(p, axis=1)
    co = np.cumsum(oh, axis=1)
    # Only first r-1 = 2 cumulative terms contribute
    return float(np.mean(np.sum((cp[:, :2] - co[:, :2]) ** 2, axis=1) / 2.0))


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    """Argmax accuracy."""
    pred = np.array([CLASS_ORDER[i] for i in np.argmax(p, axis=1)])
    return float(np.mean(pred == y))


def draw_recall(y: np.ndarray, p: np.ndarray) -> float:
    """Fraction of actual draws that were predicted as draws."""
    pred = np.array([CLASS_ORDER[i] for i in np.argmax(p, axis=1)])
    actual_draws = (y == "D")
    if actual_draws.sum() == 0:
        return float("nan")
    return float(np.mean(pred[actual_draws] == "D"))


def draw_predicted_rate(p: np.ndarray) -> float:
    """Fraction of fixtures where D is the argmax."""
    pred_idx = np.argmax(p, axis=1)
    return float(np.mean(pred_idx == 1))


def calibration_bins(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10,
) -> list[dict]:
    """Reliability curve, pooled across all 3 classes.

    Each (fixture, class) pair contributes one point:
    predicted probability vs binary outcome.
    """
    oh = to_onehot(y)
    p_flat = p.ravel()
    o_flat = oh.ravel()

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p_flat >= lo) & (p_flat <= hi)
        else:
            mask = (p_flat >= lo) & (p_flat < hi)
        n = int(mask.sum())
        if n == 0:
            bins.append({
                "bin_lo": round(lo, 3), "bin_hi": round(hi, 3),
                "n": 0, "mean_predicted": None, "observed_freq": None,
                "gap": None,
            })
            continue
        mean_pred = float(p_flat[mask].mean())
        obs = float(o_flat[mask].mean())
        bins.append({
            "bin_lo": round(lo, 3), "bin_hi": round(hi, 3),
            "n": n,
            "mean_predicted": round(mean_pred, 6),
            "observed_freq": round(obs, 6),
            "gap": round(obs - mean_pred, 6),
        })
    return bins


def expected_calibration_error(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10,
) -> float:
    """ECE: weighted mean absolute gap between predicted and observed."""
    bins = calibration_bins(y, p, n_bins)
    total = sum(b["n"] for b in bins)
    if total == 0:
        return float("nan")
    ece = 0.0
    for b in bins:
        if b["n"] == 0:
            continue
        ece += (b["n"] / total) * abs(b["gap"])
    return float(ece)


def all_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Compute the full E2 metric suite."""
    return {
        "n": int(len(y)),
        "log_loss": round(log_loss(y, p), 6),
        "brier": round(brier_score(y, p), 6),
        "rps": round(rps(y, p), 6),
        "accuracy": round(accuracy(y, p), 6),
        "draw_recall": round(draw_recall(y, p), 6),
        "draw_predicted_rate": round(draw_predicted_rate(p), 6),
        "ece": round(expected_calibration_error(y, p), 6),
    }


def validate_probs(p: np.ndarray, name: str = "probs") -> None:
    """Assert that a probability matrix is well-formed."""
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"{name}: expected (n, 3), got {p.shape}")
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name}: contains non-finite values")
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError(f"{name}: values outside [0, 1]")
    sums = p.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-6):
        bad = np.where(np.abs(sums - 1.0) > 1e-6)[0]
        raise ValueError(
            f"{name}: {len(bad)} rows do not sum to 1 "
            f"(first bad sum={sums[bad[0]]:.10f})"
        )
