"""Dixon-Coles Low-Score Matrix Transformation & Continuous Rating Parity Engine for E10 (Vectorized).

Implements:
1. High-speed vectorized Poisson & Dixon-Coles bivariate score distribution generator.
2. Low-score dependency adjustment tau_lambda,mu(x, y) for (0,0), (1,0), (0,1), (1,1).
3. Continuous Rating Parity Index (RPI) and Combined Match Intensity metrics.
4. Exact numerical normalization guaranteeing sum(P) = 1.0 within 1e-15.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np

# Precomputed log factorials for k = 0..15
MAX_GOALS = 15
LOG_FACTORIALS = np.array([math.lgamma(k + 1) for k in range(MAX_GOALS + 1)], dtype=np.float64)
K_VEC = np.arange(MAX_GOALS + 1, dtype=np.float64)


@dataclass(frozen=True)
class DixonColesConfig:
    """Configuration for Dixon-Coles low-score probability transformation."""
    rho: float = -0.1000
    max_goals: int = 15
    use_parity: bool = False
    parity_sigma: float = 55.0
    parity_intensity_ref: float = 2.50
    parity_draw_scale: float = 0.1500


def compute_poisson_pmf_fast(lam: float, max_goals: int = 15) -> np.ndarray:
    """Vectorized Poisson PMF vector for 0..max_goals."""
    lam_safe = max(float(lam), 1e-9)
    log_pmf = -lam_safe + K_VEC[:max_goals + 1] * np.log(lam_safe) - LOG_FACTORIALS[:max_goals + 1]
    pmf = np.exp(log_pmf)
    pmf_sum = np.sum(pmf)
    if pmf_sum > 0:
        pmf /= pmf_sum
    return pmf


def compute_dixon_coles_matrix_fast(
    lambda_home: float,
    lambda_away: float,
    rho: float = -0.10,
    max_goals: int = 15,
) -> np.ndarray:
    """Generate (max_goals+1) x (max_goals+1) Dixon-Coles joint score probability matrix."""
    pmf_h = compute_poisson_pmf_fast(lambda_home, max_goals=max_goals)
    pmf_a = compute_poisson_pmf_fast(lambda_away, max_goals=max_goals)

    M = np.outer(pmf_h, pmf_a)

    lh = max(float(lambda_home), 1e-6)
    la = max(float(lambda_away), 1e-6)

    tau_00 = max(1.0 - lh * la * rho, 0.0)
    tau_10 = max(1.0 + la * rho, 0.0)
    tau_01 = max(1.0 + lh * rho, 0.0)
    tau_11 = max(1.0 - rho, 0.0)

    M[0, 0] *= tau_00
    M[1, 0] *= tau_10
    M[0, 1] *= tau_01
    M[1, 1] *= tau_11

    total_p = np.sum(M)
    if total_p > 0:
        M /= total_p

    return M


def compute_1x2_from_score_matrix(M: np.ndarray) -> Tuple[float, float, float]:
    """Aggregate joint score matrix M into P(Home), P(Draw), P(Away)."""
    p_draw = float(np.sum(np.diag(M)))
    p_home = float(np.sum(np.tril(M, -1)))
    p_away = float(np.sum(np.triu(M, 1)))

    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total

    return (p_home, p_draw, p_away)


def compute_continuous_rating_parity_index(
    abs_elo_diff: float,
    sigma: float = 55.0,
) -> float:
    """Compute Gaussian Continuous Rating Parity Index RPI in [0, 1]."""
    return float(np.exp(- (abs_elo_diff ** 2) / (2.0 * (sigma ** 2))))


def compute_match_intensity_factor(
    lambda_home: float,
    lambda_away: float,
    ref_intensity: float = 2.50,
) -> float:
    """Compute low-intensity dampener in [0, 1]. Higher for lower total expected goals."""
    tot_lambda = lambda_home + lambda_away
    return float(np.exp(- tot_lambda / ref_intensity))


def apply_dixon_coles_parity_transform(
    lambda_home: float,
    lambda_away: float,
    abs_elo_diff: float,
    config: DixonColesConfig,
) -> Dict[str, Any]:
    """Execute isolated E10 Dixon-Coles + Continuous Parity probability layer."""
    M = compute_dixon_coles_matrix_fast(
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        rho=config.rho,
        max_goals=config.max_goals,
    )

    p_h, p_d, p_a = compute_1x2_from_score_matrix(M)

    rpi = compute_continuous_rating_parity_index(abs_elo_diff, sigma=config.parity_sigma)
    intensity_fac = compute_match_intensity_factor(lambda_home, lambda_away, ref_intensity=config.parity_intensity_ref)
    parity_multiplier = rpi * intensity_fac

    if config.use_parity and parity_multiplier > 0.01:
        draw_boost = config.parity_draw_scale * parity_multiplier * (1.0 - p_d)
        new_p_d = min(p_d + draw_boost, 0.450)
        remaining = 1.0 - new_p_d
        orig_win_sum = p_h + p_a
        if orig_win_sum > 0:
            new_p_h = p_h * (remaining / orig_win_sum)
            new_p_a = p_a * (remaining / orig_win_sum)
        else:
            new_p_h = remaining / 2.0
            new_p_a = remaining / 2.0

        p_h, p_d, p_a = new_p_h, new_p_d, new_p_a

    total = p_h + p_d + p_a
    assert abs(total - 1.0) < 1e-12, f"Probability normalization invariant violated: sum={total}"

    return {
        "p_home": float(p_h),
        "p_draw": float(p_d),
        "p_away": float(p_a),
        "probabilities": {"H": float(p_h), "D": float(p_d), "A": float(p_a)},
        "decision": "H" if (p_h >= p_d and p_h >= p_a) else ("D" if p_d >= p_a else "A"),
        "rho": config.rho,
        "rpi": rpi,
        "intensity_factor": intensity_fac,
        "parity_multiplier": parity_multiplier,
        "score_matrix_00": float(M[0, 0]),
        "score_matrix_11": float(M[1, 1]),
        "score_matrix_10": float(M[1, 0]),
        "score_matrix_01": float(M[0, 1]),
    }
