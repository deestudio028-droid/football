"""Phase 42 — Experiment E17: Candidate Status & Reliability Estimation Models.

Implements:
1. GradientBoostingStatusRegressor: Tree ensemble predicting continuous expected match RPS.
2. AnalyticalStatusScorer: Closed-form geometric score for fast interpretable status mapping.
3. IsotonicStatusCalibrator: Non-parametric monotonic calibration engine.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class GradientBoostingStatusRegressor:
    """Predicts continuous expected match RPS loss using HistGradientBoosting."""

    def __init__(self, max_iter: int = 100, max_depth: int = 4, learning_rate: float = 0.05):
        self.reg = HistGradientBoostingRegressor(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y_continuous_rps: np.ndarray) -> GradientBoostingStatusRegressor:
        self.reg.fit(X, y_continuous_rps)
        return self

    def predict_expected_rps(self, X: np.ndarray) -> np.ndarray:
        return self.reg.predict(X)

    def predict_status_score(self, X: np.ndarray) -> np.ndarray:
        exp_rps = self.predict_expected_rps(X)
        return np.clip(1.0 - (exp_rps / 0.50), 0.0, 1.0)


class AnalyticalStatusScorer:
    """Closed-form transparent geometric status scoring engine."""

    def compute_score(self, norm_entropy: np.ndarray, max_prob: np.ndarray, margin: np.ndarray) -> np.ndarray:
        raw_score = (1.0 - norm_entropy) + 0.5 * margin + 0.5 * (max_prob - 0.333)
        return np.clip(raw_score / 1.5, 0.0, 1.0)


class IsotonicStatusCalibrator:
    """Applies non-parametric isotonic regression to calibrate continuous status scores."""

    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, raw_scores: np.ndarray, realized_reliability: np.ndarray) -> IsotonicStatusCalibrator:
        self.iso.fit(raw_scores, realized_reliability)
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        return self.iso.transform(raw_scores)
