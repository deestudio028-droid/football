"""Phase 39 — Experiment E14: Gating Models & Dynamic Weight Computation.

Implements:
1. HardEarlySeasonGate: Hard cutoff on early season matches (k <= 3).
2. FixedRegimeWeightGate: Per-regime fixed convex weights learned via constrained optimization.
3. ContinuousSoftmaxGate: Parametric logistic/softmax gating network on pre-match regime features.
4. ModelDisagreementGate: Disagreement-aware switching based on Jensen-Shannon divergence.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def compute_jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Computes Jensen-Shannon divergence between two sets of probability distributions."""
    p = np.clip(p, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    q = np.clip(q, eps, 1.0)
    q = q / q.sum(axis=1, keepdims=True)

    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m), axis=1)
    kl_qm = np.sum(q * np.log(q / m), axis=1)
    js = 0.5 * (kl_pm + kl_qm)
    return np.clip(js, 0.0, 1.0)


class HardEarlySeasonGate:
    """Hard switching gate based on season match count."""

    def __init__(self, k_threshold: int = 3):
        self.k_threshold = k_threshold

    def compute_weights(self, season_match_nums: np.ndarray, n_experts: int = 2) -> np.ndarray:
        n = len(season_match_nums)
        weights = np.zeros((n, n_experts), dtype=np.float64)
        is_early = (season_match_nums <= self.k_threshold)
        # Expert 0 is E10, Expert 1 is E13
        weights[is_early, 1] = 1.0
        weights[~is_early, 0] = 1.0
        return weights


class FixedRegimeWeightGate:
    """Learns fixed convex combination weights per discrete regime on training fold."""

    def __init__(self):
        self.weights_by_regime: Dict[str, np.ndarray] = {}

    def fit(
        self,
        regime_labels: np.ndarray,
        expert_probs_list: List[np.ndarray],
        y_true_indices: np.ndarray,
    ) -> FixedRegimeWeightGate:
        n_experts = len(expert_probs_list)
        unique_regimes = np.unique(regime_labels)

        for reg in unique_regimes:
            mask = (regime_labels == reg)
            y_reg = y_true_indices[mask]
            if len(y_reg) < 10:
                # Default to 100% Expert 0 (E10)
                w_default = np.zeros(n_experts)
                w_default[0] = 1.0
                self.weights_by_regime[reg] = w_default
                continue

            P_experts_reg = [P[mask] for P in expert_probs_list]

            # Optimize weights to minimize RPS
            def loss_fn(w):
                w_norm = np.clip(w, 0.0, 1.0)
                if np.sum(w_norm) > 0:
                    w_norm = w_norm / np.sum(w_norm)
                else:
                    w_norm = np.zeros(n_experts)
                    w_norm[0] = 1.0

                P_blend = np.zeros_like(P_experts_reg[0])
                for j in range(n_experts):
                    P_blend += w_norm[j] * P_experts_reg[j]

                # Ranked Probability Score
                e1 = (P_blend[:, 0] - (y_reg == 0).astype(float)) ** 2
                e2 = ((P_blend[:, 0] + P_blend[:, 1]) - (y_reg <= 1).astype(float)) ** 2
                return float(0.5 * np.mean(e1 + e2))

            init_w = np.zeros(n_experts)
            init_w[0] = 1.0
            bounds = [(0.0, 1.0)] * n_experts
            res = minimize(loss_fn, init_w, method="SLSQP", bounds=bounds, constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
            if res.success:
                opt_w = np.clip(res.x, 0.0, 1.0)
                self.weights_by_regime[reg] = opt_w / np.sum(opt_w)
            else:
                w_default = np.zeros(n_experts)
                w_default[0] = 1.0
                self.weights_by_regime[reg] = w_default

        return self

    def compute_weights(self, regime_labels: np.ndarray, n_experts: int = 2) -> np.ndarray:
        n = len(regime_labels)
        weights = np.zeros((n, n_experts), dtype=np.float64)
        for i, reg in enumerate(regime_labels):
            if reg in self.weights_by_regime:
                weights[i] = self.weights_by_regime[reg]
            else:
                weights[i, 0] = 1.0  # Default anchor E10
        return weights


class ContinuousSoftmaxGate:
    """Continuous logistic/softmax gating network on pre-match regime features."""

    def __init__(self, c_reg: float = 1.0):
        self.c_reg = c_reg
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=c_reg, max_iter=1000, penalty="l2")

    def fit(
        self,
        X_regime: np.ndarray,
        expert_probs_list: List[np.ndarray],
        y_true_indices: np.ndarray,
    ) -> ContinuousSoftmaxGate:
        # Determine for each training match which expert had lower RPS
        P_e10 = expert_probs_list[0]
        P_e13 = expert_probs_list[1]

        rps_e10 = 0.5 * ((P_e10[:, 0] - (y_true_indices == 0)) ** 2 + ((P_e10[:, 0] + P_e10[:, 1]) - (y_true_indices <= 1)) ** 2)
        rps_e13 = 0.5 * ((P_e13[:, 0] - (y_true_indices == 0)) ** 2 + ((P_e13[:, 0] + P_e13[:, 1]) - (y_true_indices <= 1)) ** 2)

        # Target label: 1 if E13 was strictly better by margin >= 0.005, else 0 (prefer E10 anchor)
        target_labels = (rps_e13 < rps_e10 - 0.005).astype(int)

        X_scaled = self.scaler.fit_transform(X_regime)
        self.clf.fit(X_scaled, target_labels)
        return self

    def compute_weights(self, X_regime: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X_regime)
        probs = self.clf.predict_proba(X_scaled)
        if probs.shape[1] == 1:
            n = len(X_regime)
            weights = np.zeros((n, 2), dtype=np.float64)
            weights[:, 0] = 1.0
            return weights
        # Column 0 is P(E10 preferred), Column 1 is P(E13 preferred)
        return probs


class ModelDisagreementGate:
    """Gating based on Jensen-Shannon model divergence."""

    def __init__(self, js_threshold: float = 0.015, specialist_weight: float = 0.50):
        self.js_threshold = js_threshold
        self.specialist_weight = specialist_weight

    def compute_weights(self, P_e10: np.ndarray, P_e13: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        js_divs = compute_jensen_shannon_divergence(P_e10, P_e13)
        n = len(P_e10)
        weights = np.zeros((n, 2), dtype=np.float64)
        weights[:, 0] = 1.0  # Default 100% E10

        mask_disagree = (js_divs > self.js_threshold)
        weights[mask_disagree, 0] = 1.0 - self.specialist_weight
        weights[mask_disagree, 1] = self.specialist_weight
        return weights, js_divs
