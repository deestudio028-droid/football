"""Phase 40 — Experiment E15: Pre-Match Risk Models.

Implements:
1. LogisticRiskClassifier: Regularized Logistic Regression for binary high-error match prediction.
2. GradientBoostingRiskClassifier: Non-linear tree ensemble risk classifier.
3. ExpectedRPSRegressor: Continuous regression model predicting expected match RPS loss.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class LogisticRiskClassifier:
    """Regularized Logistic Regression model predicting probability of high prediction error."""

    def __init__(self, c_reg: float = 1.0):
        self.c_reg = c_reg
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.clf = LogisticRegression(C=c_reg, max_iter=1000)

    def fit(self, X: np.ndarray, y_binary_error: np.ndarray) -> LogisticRiskClassifier:
        X_proc = self.scaler.fit_transform(self.imputer.fit_transform(X))
        self.clf.fit(X_proc, y_binary_error)
        return self

    def predict_risk_scores(self, X: np.ndarray) -> np.ndarray:
        X_proc = self.scaler.transform(self.imputer.transform(X))
        probs = self.clf.predict_proba(X_proc)
        if probs.shape[1] == 1:
            return np.zeros(len(X), dtype=np.float64)
        return probs[:, 1]


class GradientBoostingRiskClassifier:
    """Non-linear gradient boosting classifier for high error risk prediction."""

    def __init__(self, max_iter: int = 100, max_depth: int = 4, learning_rate: float = 0.05):
        self.clf = HistGradientBoostingClassifier(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y_binary_error: np.ndarray) -> GradientBoostingRiskClassifier:
        self.clf.fit(X, y_binary_error)
        return self

    def predict_risk_scores(self, X: np.ndarray) -> np.ndarray:
        probs = self.clf.predict_proba(X)
        if probs.shape[1] == 1:
            return np.zeros(len(X), dtype=np.float64)
        return probs[:, 1]


class ExpectedRPSRegressor:
    """Continuous regression model predicting expected match RPS loss."""

    def __init__(self, max_iter: int = 100, max_depth: int = 4):
        self.reg = HistGradientBoostingRegressor(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=0.05,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y_continuous_rps: np.ndarray) -> ExpectedRPSRegressor:
        self.reg.fit(X, y_continuous_rps)
        return self

    def predict_expected_rps(self, X: np.ndarray) -> np.ndarray:
        return self.reg.predict(X)
