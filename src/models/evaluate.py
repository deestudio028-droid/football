"""Evaluation metrics for the V1 H/D/A classifier.

Implemented directly in NumPy rather than via `sklearn.metrics`,
deliberately: this module has no dependency on scikit-learn being
installed at all, which matters in an environment (like the sandbox
this was authored in) where scikit-learn cannot be installed but the
metric functions still need to be written, tested, and verified with
hand-computed examples. This is not a stand-in for a model (nothing
here trains anything or replaces LogisticRegression/HistGradientBoosting
-- see train.py for those, which DO require and use real scikit-learn);
it's a small set of well-defined mathematical functions that don't need
scikit-learn's much heavier dependency for correctness, and every
function here is independently, deterministically testable without it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baselines import CLASS_ORDER


class MetricInputError(ValueError):
    pass


def validate_probabilities(P: np.ndarray, class_order: list[str] = CLASS_ORDER) -> None:
    """Raises MetricInputError if `P` is not a valid (n, len(class_order))
    probability matrix: finite, non-negative, each row summing to ~1.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] != len(class_order):
        raise MetricInputError(f"P must have shape (n, {len(class_order)}), got {P.shape}")
    if not np.all(np.isfinite(P)):
        raise MetricInputError("P contains NaN or infinite values.")
    if np.any(P < 0):
        raise MetricInputError("P contains negative probabilities.")
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        bad = np.where(~np.isclose(row_sums, 1.0, atol=1e-6))[0]
        raise MetricInputError(f"{len(bad)} row(s) of P do not sum to ~1 (e.g. row {bad[0]}: sum={row_sums[bad[0]]})")


def encode_labels(y, class_order: list[str] = CLASS_ORDER) -> np.ndarray:
    y = np.asarray(y)
    unexpected = set(np.unique(y)) - set(class_order)
    if unexpected:
        raise MetricInputError(f"Labels contain values outside {class_order}: {sorted(unexpected)}")
    index_of = {c: i for i, c in enumerate(class_order)}
    return np.array([index_of[label] for label in y], dtype=int)


def log_loss(y, P, class_order: list[str] = CLASS_ORDER, eps: float = 1e-15) -> float:
    validate_probabilities(P, class_order)
    y_idx = encode_labels(y, class_order)
    P = np.asarray(P, dtype=float)
    true_class_probs = P[np.arange(len(y_idx)), y_idx]
    return float(-np.mean(np.log(np.clip(true_class_probs, eps, 1.0))))


def brier_score(y, P, class_order: list[str] = CLASS_ORDER) -> float:
    validate_probabilities(P, class_order)
    y_idx = encode_labels(y, class_order)
    P = np.asarray(P, dtype=float)
    Y = np.eye(len(class_order))[y_idx]
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def accuracy(y, P, class_order: list[str] = CLASS_ORDER) -> float:
    validate_probabilities(P, class_order)
    y_idx = encode_labels(y, class_order)
    preds = np.asarray(P).argmax(axis=1)
    return float(np.mean(preds == y_idx))


def confusion_matrix(y, P, class_order: list[str] = CLASS_ORDER) -> np.ndarray:
    """Rows = true class, columns = predicted class, in class_order."""
    validate_probabilities(P, class_order)
    y_idx = encode_labels(y, class_order)
    preds = np.asarray(P).argmax(axis=1)
    n = len(class_order)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_idx, preds):
        cm[t, p] += 1
    return cm


def macro_f1(y, P, class_order: list[str] = CLASS_ORDER) -> float:
    cm = confusion_matrix(y, P, class_order)
    n = len(class_order)
    f1s = []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def balanced_accuracy(y, P, class_order: list[str] = CLASS_ORDER) -> float:
    """Macro-average of per-class recall, matching sklearn's
    balanced_accuracy_score definition for the multiclass case.
    """
    cm = confusion_matrix(y, P, class_order)
    n = len(class_order)
    recalls = []
    for i in range(n):
        tp = cm[i, i]
        support = cm[i, :].sum()
        recalls.append(tp / support if support > 0 else 0.0)
    return float(np.mean(recalls))


@dataclass
class EvaluationResult:
    log_loss: float
    brier: float
    macro_f1: float
    balanced_accuracy: float
    accuracy: float
    confusion_matrix: np.ndarray
    n: int

    def as_dict(self) -> dict:
        return {
            "log_loss": self.log_loss, "brier": self.brier, "macro_f1": self.macro_f1,
            "balanced_accuracy": self.balanced_accuracy, "accuracy": self.accuracy,
            "confusion_matrix": self.confusion_matrix.tolist(), "n": self.n,
        }


def evaluate(y, P, class_order: list[str] = CLASS_ORDER) -> EvaluationResult:
    """Computes every metric in one pass, all against the same
    already-validated (y, P) pair."""
    validate_probabilities(P, class_order)
    return EvaluationResult(
        log_loss=log_loss(y, P, class_order),
        brier=brier_score(y, P, class_order),
        macro_f1=macro_f1(y, P, class_order),
        balanced_accuracy=balanced_accuracy(y, P, class_order),
        accuracy=accuracy(y, P, class_order),
        confusion_matrix=confusion_matrix(y, P, class_order),
        n=len(np.asarray(y)),
    )
