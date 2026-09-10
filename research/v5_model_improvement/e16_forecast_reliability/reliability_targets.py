"""Phase 41 — Experiment E16: Forecast Reliability Target Definitions.

Implements realized probabilistic evaluation targets:
- Continuous Ranked Probability Score (RPS)
- Continuous Multiclass Brier Score
- Continuous Multi-Class Log-Loss
- Correctness of Top-1 predicted outcome
- Continuous Calibration Residuals
- Ground-truth Confidence Indicator
"""
from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd


def compute_realized_error_targets(
    y_true: np.ndarray,
    probs_e10: np.ndarray,
    eps: float = 1e-15,
) -> pd.DataFrame:
    """Computes continuous and binary ground-truth loss metrics for each match."""
    mapping = {"H": 0, "D": 1, "A": 2}
    y_idx = np.array([mapping[y] if isinstance(y, str) else int(y) for y in y_true])
    n = len(y_idx)

    # 1. Realized RPS
    e1 = (probs_e10[:, 0] - (y_idx == 0).astype(np.float64)) ** 2
    e2 = ((probs_e10[:, 0] + probs_e10[:, 1]) - (y_idx <= 1).astype(np.float64)) ** 2
    match_rps = 0.5 * (e1 + e2)

    # 2. Realized Multi-Class Log-Loss
    clipped_probs = np.clip(probs_e10, eps, 1.0 - eps)
    clipped_probs /= clipped_probs.sum(axis=1, keepdims=True)
    match_ll = -np.log(clipped_probs[np.arange(n), y_idx])

    # 3. Realized Multiclass Brier Score: sum((p_k - y_k)^2)
    y_onehot = np.zeros((n, 3), dtype=np.float64)
    y_onehot[np.arange(n), y_idx] = 1.0
    match_brier = np.sum((probs_e10 - y_onehot) ** 2, axis=1)

    # 4. Top-1 Correctness (1 if highest probability class matched actual result, else 0)
    pred_top1 = np.argmax(probs_e10, axis=1)
    top1_correct = (pred_top1 == y_idx).astype(int)

    # 5. Continuous Reliability Ground-Truth: 1.0 - normalized RPS error in [0, 1]
    # Max possible match RPS is 1.0 (e.g. predicting P(H)=1.0 and A occurs: 0.5 * (1^2 + 1^2) = 1.0)
    realized_reliability = np.clip(1.0 - match_rps, 0.0, 1.0)

    return pd.DataFrame({
        "match_rps": match_rps,
        "match_log_loss": match_ll,
        "match_brier": match_brier,
        "top1_correct": top1_correct,
        "realized_reliability": realized_reliability,
    })
