"""Phase 37 — Experiment E12: Bounded Lambda Adjustment Engine.

Applies bounded exponential multipliers to V4 baseline lambdas:
    lambda_home_adj = lambda_home_v4 * exp(clip(delta_home, -bound, +bound))
    lambda_away_adj = lambda_away_v4 * exp(clip(delta_away, -bound, +bound))

Then computes Dixon-Coles 1X2 probabilities (rho = -0.08).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "research/v5_model_improvement/e10_dixon_coles"))

from dixon_coles_engine import compute_1x2_from_score_matrix, compute_dixon_coles_matrix_fast


def apply_bounded_lambda_adjustment(
    v4_lam_h: np.ndarray,
    v4_lam_a: np.ndarray,
    delta_h: np.ndarray,
    delta_a: np.ndarray,
    bound: float = 0.05,
    rho: float = -0.08,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Applies bounded exponential multiplier to V4 base lambdas and returns adjusted lambdas and 1X2 probs.

    Returns:
        (probs_1x2, lam_h_adj, lam_a_adj)
    """
    clipped_dh = np.clip(delta_h, -bound, bound)
    clipped_da = np.clip(delta_a, -bound, bound)

    lam_h_adj = v4_lam_h * np.exp(clipped_dh)
    lam_a_adj = v4_lam_a * np.exp(clipped_da)

    probs = []
    for lh, la in zip(lam_h_adj, lam_a_adj):
        M = compute_dixon_coles_matrix_fast(lh, la, rho=rho)
        probs.append(list(compute_1x2_from_score_matrix(M)))

    return np.array(probs), lam_h_adj, lam_a_adj
