"""Phase 38 — Experiment E13: Parameter Uncertainty Propagation Engine.

Integrates Dixon-Coles score distributions over latent parameter posterior uncertainties:
    P(x, y) = E_{lambda ~ Posterior} [ DixonColes(x, y | lambda_H, lambda_A, rho=-0.08) ]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "research/v5_model_improvement/e10_dixon_coles"))

from dixon_coles_engine import compute_1x2_from_score_matrix, compute_dixon_coles_matrix_fast


def propagate_parameter_uncertainty(
    mean_lam_h: np.ndarray,
    mean_lam_a: np.ndarray,
    var_lam_h: np.ndarray,
    var_lam_a: np.ndarray,
    rho: float = -0.08,
    n_quad_points: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrates Dixon-Coles probabilities over posterior log-normal parameter uncertainty.

    Uses Gauss-Hermite quadrature nodes for exact deterministic integration.
    """
    # 5-point Gauss-Hermite quadrature nodes and weights
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad_points)
    weights = weights / np.sqrt(np.pi)

    n_matches = len(mean_lam_h)
    all_probs = np.zeros((n_matches, 3), dtype=np.float64)
    all_entropies = np.zeros(n_matches, dtype=np.float64)

    # Scale parameter standard deviations (capped for safety)
    std_log_h = np.clip(np.sqrt(np.maximum(var_lam_h, 1e-4)), 0.01, 0.35)
    std_log_a = np.clip(np.sqrt(np.maximum(var_lam_a, 1e-4)), 0.01, 0.35)

    for i in range(n_matches):
        mu_h = mean_lam_h[i]
        mu_a = mean_lam_a[i]
        sh = std_log_h[i]
        sa = std_log_a[i]

        match_probs = np.zeros(3, dtype=np.float64)

        # 2D Gauss-Hermite integration
        for nh, wh in zip(nodes, weights):
            lh_sample = mu_h * np.exp(np.sqrt(2.0) * sh * nh - 0.5 * (sh ** 2))
            for na, wa in zip(nodes, weights):
                la_sample = mu_a * np.exp(np.sqrt(2.0) * sa * na - 0.5 * (sa ** 2))
                M = compute_dixon_coles_matrix_fast(lh_sample, la_sample, rho=rho)
                p1x2 = compute_1x2_from_score_matrix(M)
                match_probs += (wh * wa) * np.array(p1x2)

        match_probs /= np.sum(match_probs)
        all_probs[i] = match_probs
        all_entropies[i] = -np.sum(match_probs * np.log(np.clip(match_probs, 1e-12, 1.0)))

    return all_probs, all_entropies
