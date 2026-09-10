"""Phase 40 — Experiment E15: Forecast Error Target Generator.

Computes realized prediction error metrics for E10 predictions:
- Ranked Probability Score (RPS) per match
- Multi-class Log-Loss per match
- Top-decile and top-quintile binary high-error indicators
- Continuous error severity targets
"""
from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd


def compute_match_rps_and_log_loss(
    y_true: np.ndarray,
    probs_e10: np.ndarray,
    eps: float = 1e-15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes exact per-match RPS and Log-Loss values."""
    mapping = {"H": 0, "D": 1, "A": 2}
    y_idx = np.array([mapping[y] if isinstance(y, str) else int(y) for y in y_true])
    n = len(y_idx)

    # 1. Per-match RPS
    e1 = (probs_e10[:, 0] - (y_idx == 0).astype(np.float64)) ** 2
    e2 = ((probs_e10[:, 0] + probs_e10[:, 1]) - (y_idx <= 1).astype(np.float64)) ** 2
    match_rps = 0.5 * (e1 + e2)

    # 2. Per-match Log-Loss
    clipped_probs = np.clip(probs_e10, eps, 1.0 - eps)
    clipped_probs /= clipped_probs.sum(axis=1, keepdims=True)
    match_ll = -np.log(clipped_probs[np.arange(n), y_idx])

    return match_rps, match_ll


def create_error_targets(
    match_rps: np.ndarray,
    match_ll: np.ndarray,
    rps_thresh_10: float,
    rps_thresh_20: float,
    ll_thresh_10: float,
    ll_thresh_20: float,
) -> pd.DataFrame:
    """Generates binary and continuous error targets using pre-determined training thresholds."""
    return pd.DataFrame({
        "match_rps": match_rps,
        "match_log_loss": match_ll,
        "target_top_10pct_rps": (match_rps >= rps_thresh_10).astype(int),
        "target_top_20pct_rps": (match_rps >= rps_thresh_20).astype(int),
        "target_top_10pct_ll": (match_ll >= ll_thresh_10).astype(int),
        "target_top_20pct_ll": (match_ll >= ll_thresh_20).astype(int),
    })
