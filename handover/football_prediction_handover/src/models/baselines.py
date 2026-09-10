"""Majority-class and frequency baselines (design report §8).

Both baselines are fit exclusively from a training partition's own
labels -- never validation or test labels -- and both always emit
valid 3-class (H, D, A) probability distributions (each row sums to 1,
every entry is a finite, non-negative float) so they can be scored with
exactly the same metric functions (`evaluate.py`) as any real model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ALLOWED_LABELS

CLASS_ORDER = list(ALLOWED_LABELS)  # ("H", "D", "A") -- fixed column order for every probability array


class BaselineError(RuntimeError):
    pass


def _validate_training_labels(y_train: pd.Series) -> None:
    unexpected = set(y_train.unique()) - set(ALLOWED_LABELS)
    if unexpected:
        raise BaselineError(f"Training labels contain unexpected values: {sorted(unexpected)}")
    if len(y_train) == 0:
        raise BaselineError("Cannot fit a baseline on zero training rows.")


def class_frequencies(y_train: pd.Series) -> np.ndarray:
    """Training-period H/D/A frequencies, in CLASS_ORDER. Computed ONLY
    from `y_train` -- callers must never pass validation/test labels here.
    """
    _validate_training_labels(y_train)
    counts = y_train.value_counts()
    freqs = np.array([counts.get(c, 0) for c in CLASS_ORDER], dtype=float)
    return freqs / freqs.sum()


class FrequencyBaseline:
    """Predicts the training period's H/D/A proportions as a constant
    probability triple for every row, regardless of that row's features.
    """

    def __init__(self) -> None:
        self.class_order = CLASS_ORDER
        self._freqs: np.ndarray | None = None

    def fit(self, y_train: pd.Series) -> "FrequencyBaseline":
        self._freqs = class_frequencies(y_train)
        return self

    def predict_proba(self, n_rows: int) -> np.ndarray:
        if self._freqs is None:
            raise BaselineError("FrequencyBaseline.fit() must be called before predict_proba().")
        return np.tile(self._freqs, (n_rows, 1))


class MajorityClassBaseline:
    """Predicts the training period's single most frequent class with
    probability 1.0 (and 0.0 for the other two) for every row.
    """

    def __init__(self) -> None:
        self.class_order = CLASS_ORDER
        self._majority_idx: int | None = None

    def fit(self, y_train: pd.Series) -> "MajorityClassBaseline":
        freqs = class_frequencies(y_train)
        self._majority_idx = int(np.argmax(freqs))
        return self

    def predict_proba(self, n_rows: int) -> np.ndarray:
        if self._majority_idx is None:
            raise BaselineError("MajorityClassBaseline.fit() must be called before predict_proba().")
        P = np.zeros((n_rows, len(self.class_order)))
        P[:, self._majority_idx] = 1.0
        return P
