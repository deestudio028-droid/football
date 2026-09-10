"""Phase 42 — Experiment E17: Forecast Status Target Definitions.

Computes ground-truth forecast error metrics and evaluation targets:
- Realized Ranked Probability Score (RPS)
- Realized Multi-Class Log-Loss
- Realized Multiclass Brier Score
- Top-1 Outcome Correctness
- Realized Reliability Target (1 - RPS)
"""
from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd


def compute_status_evaluation_targets(
    y_true: np.ndarray,
    probs_e10: np.ndarray,
    eps: float = 1e-15,
) -> pd.DataFrame:
    """Computes realized loss and accuracy targets for each fixture."""
    mapping = {"H": 0, "D": 1, "A": 2}
    y_idx = np.array([mapping[y] if isinstance(y, str) else int(y) for y in y_true])
    n = len(y_idx)

    # 1. Realized RPS
    e1 = (probs_e10[:, 0] - (y_idx == 0).astype(np.float64)) ** 2
    e2 = ((probs_e10[:, 0] + probs_e10[:, 1]) - (y_idx <= 1).astype(np.float64)) ** 2
    match_rps = 0.5 * (e1 + e2)

    # 2. Realized Log-Loss
    clipped_probs = np.clip(probs_e10, eps, 1.0 - eps)
    clipped_probs /= clipped_probs.sum(axis=1, keepdims=True)
    match_ll = -np.log(clipped_probs[np.arange(n), y_idx])

    # 3. Realized Multiclass Brier Score
    y_onehot = np.zeros((n, 3), dtype=np.float64)
    y_onehot[np.arange(n), y_idx] = 1.0
    match_brier = np.sum((probs_e10 - y_onehot) ** 2, axis=1)

    # 4. Top-1 Outcome Correctness
    pred_top1 = np.argmax(probs_e10, axis=1)
    top1_correct = (pred_top1 == y_idx).astype(int)

    # 5. Realized Reliability Target
    realized_reliability = np.clip(1.0 - match_rps, 0.0, 1.0)

    return pd.DataFrame({
        "match_rps": match_rps,
        "match_log_loss": match_ll,
        "match_brier": match_brier,
        "top1_correct": top1_correct,
        "realized_reliability": realized_reliability,
    })
