"""Phase 41 — Experiment E16: Candidate Reliability & Confidence Models.

Implements:
1. LogisticReliabilityClassifier: Calibrated Logistic model for top-1 correctness.
2. GradientBoostingReliabilityRegressor: Tree ensemble predicting expected continuous loss.
3. ExtraTreesReliabilityRegressor: ExtraTrees ensemble for variance-reduced expected loss estimation.
4. AnalyticalConfidenceEngine: Closed-form geometric confidence scoring function.
5. IsotonicCalibratedReliabilityEngine: Non-parametric isotonic reliability calibration.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class LogisticReliabilityClassifier:
    """Predicts probability of top-1 forecast correctness using regularized logistic regression."""

    def __init__(self, c_reg: float = 1.0):
        self.c_reg = c_reg
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.clf = LogisticRegression(C=c_reg, max_iter=1000)

    def fit(self, X: np.ndarray, y_top1_correct: np.ndarray) -> LogisticReliabilityClassifier:
        X_proc = self.scaler.fit_transform(self.imputer.fit_transform(X))
        self.clf.fit(X_proc, y_top1_correct)
        return self

    def predict_reliability(self, X: np.ndarray) -> np.ndarray:
        X_proc = self.scaler.transform(self.imputer.transform(X))
        probs = self.clf.predict_proba(X_proc)
        if probs.shape[1] == 1:
            return np.ones(len(X), dtype=np.float64) * 0.5
        return probs[:, 1]


class GradientBoostingReliabilityRegressor:
    """Predicts continuous expected match RPS loss using non-linear tree boosting."""

    def __init__(self, max_iter: int = 100, max_depth: int = 4, learning_rate: float = 0.05):
        self.reg = HistGradientBoostingRegressor(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y_continuous_rps: np.ndarray) -> GradientBoostingReliabilityRegressor:
        self.reg.fit(X, y_continuous_rps)
        return self

    def predict_expected_rps(self, X: np.ndarray) -> np.ndarray:
        return self.reg.predict(X)

    def predict_reliability_score(self, X: np.ndarray) -> np.ndarray:
        exp_rps = self.predict_expected_rps(X)
        # Higher expected RPS -> lower reliability score in [0, 1]
        return np.clip(1.0 - (exp_rps / 0.50), 0.0, 1.0)


class ExtraTreesReliabilityRegressor:
    """Predicts continuous expected match RPS loss using extremely randomized trees."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 6):
        self.imputer = SimpleImputer(strategy="median")
        self.reg = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y_continuous_rps: np.ndarray) -> ExtraTreesReliabilityRegressor:
        X_proc = self.imputer.fit_transform(X)
        self.reg.fit(X_proc, y_continuous_rps)
        return self

    def predict_expected_rps(self, X: np.ndarray) -> np.ndarray:
        X_proc = self.imputer.transform(X)
        return self.reg.predict(X_proc)

    def predict_reliability_score(self, X: np.ndarray) -> np.ndarray:
        exp_rps = self.predict_expected_rps(X)
        return np.clip(1.0 - (exp_rps / 0.50), 0.0, 1.0)


class AnalyticalConfidenceEngine:
    """Closed-form pre-match geometric confidence calculator."""

    def compute_confidence(self, norm_entropy: np.ndarray, max_prob: np.ndarray, margin: np.ndarray) -> np.ndarray:
        # Confidence is high when entropy is low and margin/max_prob are high
        raw_conf = (1.0 - norm_entropy) + 0.5 * margin + 0.5 * (max_prob - 0.333)
        return np.clip(raw_conf / 1.5, 0.0, 1.0)


class IsotonicCalibratedReliabilityEngine:
    """Applies non-parametric isotonic regression to calibrate raw reliability scores."""

    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, raw_scores: np.ndarray, realized_reliability: np.ndarray) -> IsotonicCalibratedReliabilityEngine:
        self.iso.fit(raw_scores, realized_reliability)
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        return self.iso.transform(raw_scores)
