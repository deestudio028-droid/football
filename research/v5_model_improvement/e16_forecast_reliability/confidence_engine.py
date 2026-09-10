"""Phase 41 — Experiment E16: Confidence Engine & Selective Coverage.

Implements:
1. Walk-Forward Confidence Band Classifier (HIGH, MODERATE, LOW CONFIDENCE).
2. Selective Prediction Coverage Evaluator (100% to 50% retained coverage).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


class WalkForwardConfidenceEngine:
    """Classifies fixtures into pre-match confidence bands using training fold thresholds."""

    def __init__(self, high_percentile: float = 70.0, low_percentile: float = 30.0):
        self.high_percentile = high_percentile
        self.low_percentile = low_percentile
        self.thresh_high = 0.65
        self.thresh_low = 0.45

    def fit_thresholds(self, train_reliability_scores: np.ndarray) -> WalkForwardConfidenceEngine:
        self.thresh_high = float(np.percentile(train_reliability_scores, self.high_percentile))
        self.thresh_low = float(np.percentile(train_reliability_scores, self.low_percentile))
        return self

    def assign_confidence_bands(self, test_reliability_scores: np.ndarray) -> np.ndarray:
        bands = np.full(len(test_reliability_scores), "MODERATE_CONFIDENCE", dtype=object)
        bands[test_reliability_scores >= self.thresh_high] = "HIGH_CONFIDENCE"
        bands[test_reliability_scores < self.thresh_low] = "LOW_CONFIDENCE"
        return bands


def compute_reliability_coverage_curve(
    reliability_scores: np.ndarray,
    probs_e10: np.ndarray,
    y_true: np.ndarray,
    coverage_levels: List[float] = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50],
) -> pd.DataFrame:
    """Evaluates retained forecast accuracy under selective prediction (retaining highest reliability fixtures)."""
    mapping = {"H": 0, "D": 1, "A": 2}
    y_idx = np.array([mapping[y] if isinstance(y, str) else int(y) for y in y_true])
    n_total = len(y_true)

    records = []
    for cov in coverage_levels:
        if cov == 1.0:
            mask = np.ones(n_total, dtype=bool)
        else:
            pct_discard = (1.0 - cov) * 100.0
            cutoff = float(np.percentile(reliability_scores, pct_discard))
            mask = (reliability_scores >= cutoff)

        n_retained = int(np.sum(mask))
        if n_retained == 0:
            continue

        y_ret = y_idx[mask]
        p_ret = probs_e10[mask]

        # RPS
        e1 = (p_ret[:, 0] - (y_ret == 0).astype(float)) ** 2
        e2 = ((p_ret[:, 0] + p_ret[:, 1]) - (y_ret <= 1).astype(float)) ** 2
        ret_rps = float(0.5 * np.mean(e1 + e2))

        # Log Loss
        clipped_p = np.clip(p_ret, 1e-15, 1.0 - 1e-15)
        clipped_p /= clipped_p.sum(axis=1, keepdims=True)
        ret_ll = float(-np.mean(np.log(clipped_p[np.arange(n_retained), y_ret])))

        # Multiclass Brier Score
        y_onehot = np.zeros((n_retained, 3), dtype=float)
        y_onehot[np.arange(n_retained), y_ret] = 1.0
        ret_brier = float(np.mean(np.sum((p_ret - y_onehot) ** 2, axis=1)))

        # Draw ECE
        bins = np.linspace(0.0, 1.0, 11)
        d_probs = p_ret[:, 1]
        y_d = (y_ret == 1).astype(float)
        bin_indices = np.digitize(d_probs, bins) - 1
        d_ece = 0.0
        for b in range(10):
            b_mask = (bin_indices == b)
            cnt = int(np.sum(b_mask))
            if cnt > 0:
                bin_acc = np.mean(y_d[b_mask])
                bin_conf = np.mean(d_probs[b_mask])
                d_ece += (cnt / n_retained) * abs(bin_acc - bin_conf)

        # Entropy & Max Prob
        entropy = -np.sum(clipped_p * np.log(clipped_p), axis=1)
        max_prob = np.max(p_ret, axis=1)

        records.append({
            "coverage_pct": int(round(cov * 100)),
            "retained_matches": n_retained,
            "abstain_count": n_total - n_retained,
            "retained_rps": round(ret_rps, 6),
            "retained_log_loss": round(ret_ll, 6),
            "retained_brier": round(ret_brier, 6),
            "draw_ece": round(d_ece, 4),
            "mean_entropy": round(float(np.mean(entropy)), 4),
            "mean_max_prob": round(float(np.mean(max_prob)), 4),
        })

    return pd.DataFrame(records)
