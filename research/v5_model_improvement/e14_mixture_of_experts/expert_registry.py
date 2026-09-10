"""Phase 39 — Experiment E14: Expert Models Registry.

Manages forecasting experts:
1. Expert 1: Pure E10 Champion Anchor (V4 + Dixon-Coles rho=-0.08)
2. Expert 2: E13 Dynamic Specialist (Adaptive-K Elo + State-Space AD + Hierarchical Shrinkage)
3. Expert 3: Conservative Empirical Baseline (League-wide historical baseline + empirical prior)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "research/v5_model_improvement/e10_dixon_coles"))

from dixon_coles_engine import compute_1x2_from_score_matrix, compute_dixon_coles_matrix_fast


def compute_expert_e10_probabilities(lam_h: np.ndarray, lam_a: np.ndarray, rho: float = -0.08) -> np.ndarray:
    """Generates Expert 1 (Pure E10) probabilities."""
    n = len(lam_h)
    probs = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        M = compute_dixon_coles_matrix_fast(lam_h[i], lam_a[i], rho=rho)
        probs[i] = compute_1x2_from_score_matrix(M)
    return probs


def compute_expert_conservative_probabilities(
    n_matches: int,
    base_mu_h: float = 1.5348,
    base_mu_a: float = 1.2740,
    rho: float = -0.08,
) -> np.ndarray:
    """Generates Expert 3 (Conservative Baseline) probabilities."""
    M = compute_dixon_coles_matrix_fast(base_mu_h, base_mu_a, rho=rho)
    base_prob = compute_1x2_from_score_matrix(M)
    return np.tile(base_prob, (n_matches, 1))
