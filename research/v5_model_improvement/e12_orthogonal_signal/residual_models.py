"""Phase 37 — Experiment E12: Residual Models Module.

Fits regularized residual learners predicting bounded, zero-centered log-intensity adjustments
(delta_H, delta_A) over frozen V4 base lambdas.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class BoundedResidualModel:
    """Regularized residual model predicting log-intensity corrections delta_H and delta_A."""

    def __init__(self, model_type: str = "ridge", alpha: float = 200.0, l1_ratio: float = 0.5):
        self.model_type = model_type
        self.alpha = alpha
        self.l1_ratio = l1_ratio

        if model_type == "ridge":
            self.model_h = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=alpha, fit_intercept=False)),
            ])
            self.model_a = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=alpha, fit_intercept=False)),
            ])
        elif model_type == "elasticnet":
            self.model_h = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("reg", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, max_iter=2000)),
            ])
            self.model_a = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("reg", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, max_iter=2000)),
            ])
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(
        self,
        X_tr: np.ndarray,
        v4_lam_h: np.ndarray,
        v4_lam_a: np.ndarray,
        goals_h: np.ndarray,
        goals_a: np.ndarray,
    ) -> BoundedResidualModel:
        """Fits residual models on zero-centered log-intensity prediction errors."""
        smooth_eps = 0.20
        raw_res_h = np.log(goals_h + smooth_eps) - np.log(v4_lam_h + smooth_eps)
        raw_res_a = np.log(goals_a + smooth_eps) - np.log(v4_lam_a + smooth_eps)

        # Center residuals so adjustment represents relative tilt rather than base shift
        y_res_h = raw_res_h - np.mean(raw_res_h)
        y_res_a = raw_res_a - np.mean(raw_res_a)

        self.model_h.fit(X_tr, y_res_h)
        self.model_a.fit(X_tr, y_res_a)
        return self

    def predict_deltas(self, X_te: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts continuous zero-centered log-intensity adjustments delta_H and delta_A."""
        delta_h = self.model_h.predict(X_te)
        delta_a = self.model_a.predict(X_te)
        return delta_h, delta_a
