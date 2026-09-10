"""V4.2 Draw Resolution Research Candidate Model.

Combines Diagonal-Inflated Bivariate Poisson (DIBP), Multiclass Dirichlet Calibration (ODIR),
calibrated stacking, and an operational decision layer.

Classification: STRICTLY RESEARCH CANDIDATE — NOT PRODUCTION.
Production model (v4_draw_champion / v4.0) remains 100% frozen and untouched.

Usage:
    from models.v4_2_draw_resolution_candidate import DrawResolutionConfig, predict_draw_resolution
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# NUMERICAL UTILITIES
# ---------------------------------------------------------------------------

def stable_logit(p: float | np.ndarray, eps: float = 1e-7) -> float | np.ndarray:
    p_c = np.clip(p, eps, 1.0 - eps)
    return np.log(p_c / (1.0 - p_c))

def stable_sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

def stable_softmax(logits: np.ndarray) -> np.ndarray:
    l_max = np.max(logits, axis=-1, keepdims=True)
    exp_l = np.exp(logits - l_max)
    return exp_l / np.sum(exp_l, axis=-1, keepdims=True)

# ---------------------------------------------------------------------------
# DIAGONAL-INFLATED BIVARIATE POISSON (DIBP)
# ---------------------------------------------------------------------------

def compute_dibp_scoreline_matrix(
    lambda_h: float,
    lambda_a: float,
    rho: float = -0.0560,
    p_inf: float = 0.0500,
    max_goals: int = 10,
) -> np.ndarray:
    """Computes a (max_goals+1, max_goals+1) joint scoreline matrix with Dixon-Coles

    low-score adjustments and diagonal mixture inflation (Karlis & Ntzoufras 2005).
    """
    goals = np.arange(max_goals + 1)
    p_h = stats.poisson.pmf(goals, max(1e-4, lambda_h))
    p_a = stats.poisson.pmf(goals, max(1e-4, lambda_a))
    
    # 1. Base independent Poisson outer product
    grid = np.outer(p_h, p_a)
    
    # 2. Dixon-Coles tau low-score adjustment
    grid[0, 0] *= max(0.0, 1.0 - lambda_h * lambda_a * rho)
    grid[1, 0] *= max(0.0, 1.0 + lambda_a * rho)
    grid[0, 1] *= max(0.0, 1.0 + lambda_h * rho)
    grid[1, 1] *= max(0.0, 1.0 - rho)
    
    grid /= grid.sum()
    
    # 3. Diagonal Inflation Mixture: P(x, y) = (1 - p_inf) * P_DC(x, y) + p_inf * I(x=y) * P_diag(k)
    if p_inf > 0.0:
        diag_base = np.diag(grid).copy()
        diag_sum = diag_base.sum()
        if diag_sum > 0:
            diag_prob = diag_base / diag_sum
        else:
            diag_prob = np.ones(max_goals + 1) / (max_goals + 1)
            
        grid = (1.0 - p_inf) * grid
        for k in range(max_goals + 1):
            grid[k, k] += p_inf * diag_prob[k]
            
        grid /= grid.sum()
        
    return grid

def extract_1x2_from_grid(grid: np.ndarray) -> Tuple[float, float, float]:
    """Extracts P(Home), P(Draw), P(Away) from a joint scoreline grid."""
    p_home = float(np.sum(np.tril(grid, -1)))
    p_draw = float(np.sum(np.diag(grid)))
    p_away = float(np.sum(np.triu(grid, 1)))
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total

# ---------------------------------------------------------------------------
# CONFIGURATION & DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DrawResolutionConfig:
    """Configuration for V4.2 Draw Resolution Candidate."""
    model_name: str = "v4_2_draw_resolution_candidate"
    model_version: str = "v4.2-draw-resolution-candidate"
    
    # Probability Layer Parameters
    use_dibp: bool = True
    dibp_inflation_p: float = 0.0500
    stacking_intercept: float = 0.2450
    stacking_weight_dc: float = 0.6037
    stacking_weight_elo: float = 0.4812
    
    # Dirichlet Calibration Layer Parameters (Optional 3x3 Log-linear)
    use_dirichlet: bool = False
    dirichlet_weights_diag: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    dirichlet_biases: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Decision Layer Parameters
    decision_policy: str = "threshold" # 'argmax', 'threshold', 'margin_aware', 'utility'
    decision_threshold: float = 0.2800
    decision_margin: float = 0.0800
    
    # League Rhos (Dixon-Coles)
    league_rhos: Dict[str, float] = field(default_factory=lambda: {
        "Bundesliga": -0.0560,
        "La Liga": -0.0560,
        "Ligue 1": -0.0560,
        "Premier League": -0.0560,
        "Serie A": -0.0560,
    })
    global_fallback_rho: float = -0.0560

@dataclass
class DrawResolutionPrediction:
    """Prediction outcome from V4.2 Draw Resolution Candidate."""
    fixture_id: int | str
    probabilities: Dict[str, float]
    predicted_class: str
    decision_policy_applied: str
    p_draw_raw_v4: float
    p_draw_dibp: float
    p_draw_elo: float
    expected_goals_home: float
    expected_goals_away: float
    confidence_tier: str
    metadata: Dict[str, Any] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# PREDICTION INFERENCE PIPELINE
# ---------------------------------------------------------------------------

def predict_draw_resolution(
    lambda_h: float,
    lambda_a: float,
    p_v4: np.ndarray | List[float],
    abs_elo_diff: float,
    league: str,
    config: Optional[DrawResolutionConfig] = None,
    fixture_id: int | str = 0,
) -> DrawResolutionPrediction:
    """Generates a leak-free, calibrated probabilistic and decision-aware forecast

    for a single fixture using the V4.2 Draw Resolution architecture.
    """
    if config is None:
        config = DrawResolutionConfig()
        
    p_v4_arr = np.array(p_v4, dtype=float)
    rho = config.league_rhos.get(league, config.global_fallback_rho)
    
    # 1. Compute DIBP Joint Scoreline Grid
    grid = compute_dibp_scoreline_matrix(
        lambda_h=lambda_h,
        lambda_a=lambda_a,
        rho=rho,
        p_inf=config.dibp_inflation_p if config.use_dibp else 0.0,
    )
    p_h_dibp, p_d_dibp, p_a_dibp = extract_1x2_from_grid(grid)
    
    # 2. Compute Elo-based Draw Probability (Davidson Model Layer)
    p_d_v4 = float(p_v4_arr[1])
    # Baseline Elo Draw link: p_d_elo = p_d_v4 * exp(-0.0006 * abs_elo_diff)
    p_d_elo = float(p_d_v4 * np.exp(-0.0006 * abs_elo_diff))
    
    # 3. Calibrated Logit Stacking
    z_dc = stable_logit(p_d_dibp)
    z_elo = stable_logit(p_d_elo)
    z_champ = config.stacking_intercept + config.stacking_weight_dc * z_dc + config.stacking_weight_elo * z_elo
    p_d_calibrated = float(stable_sigmoid(z_champ))
    
    # 4. Proportional Redistribution on Simplex
    # P(H) = P_V4(H) * (1 - P(D)_new) / (1 - P_V4(D))
    # P(A) = P_V4(A) * (1 - P(D)_new) / (1 - P_V4(D))
    denom = max(1e-12, 1.0 - p_d_v4)
    scale = (1.0 - p_d_calibrated) / denom
    p_h_final = float(p_v4_arr[0] * scale)
    p_a_final = float(p_v4_arr[2] * scale)
    
    raw_probs = np.array([p_h_final, p_d_calibrated, p_a_final], dtype=float)
    raw_probs /= raw_probs.sum()
    
    # 5. Optional Multiclass Dirichlet Calibration Post-Processing
    if config.use_dirichlet:
        W = np.diag(config.dirichlet_weights_diag)
        b = np.array(config.dirichlet_biases)
        log_p = np.log(np.clip(raw_probs, 1e-12, 1.0 - 1e-12))
        logits = log_p @ W + b
        raw_probs = stable_softmax(logits)
        
    p_h, p_d, p_a = float(raw_probs[0]), float(raw_probs[1]), float(raw_probs[2])
    probs_dict = {"H": p_h, "D": p_d, "A": p_a}
    
    # 6. Operational Decision Policy
    pred_class = "H"
    if config.decision_policy == "argmax":
        classes = ["H", "D", "A"]
        pred_class = classes[int(np.argmax(raw_probs))]
    elif config.decision_policy == "threshold":
        if p_d >= config.decision_threshold:
            pred_class = "D"
        else:
            pred_class = "H" if p_h >= p_a else "A"
    elif config.decision_policy == "margin_aware":
        max_ha = max(p_h, p_a)
        if p_d >= config.decision_threshold and (max_ha - p_d) <= config.decision_margin:
            pred_class = "D"
        else:
            pred_class = "H" if p_h >= p_a else "A"
    elif config.decision_policy == "utility":
        # Expected value betting policy under fair odds: U = E[1/p_fair * 1_{y=c}] - 1
        classes = ["H", "D", "A"]
        pred_class = classes[int(np.argmax(raw_probs))]
    else:
        pred_class = "H" if p_h >= max(p_d, p_a) else ("D" if p_d >= p_a else "A")
        
    # 7. Confidence Tier
    max_p = max(p_h, p_d, p_a)
    if max_p >= 0.50:
        conf = "HIGH"
    elif max_p >= 0.35:
        conf = "MEDIUM"
    else:
        conf = "LOW"
        
    # Scoreline Top-3 extract
    top_scores = []
    flat_indices = np.argsort(grid.ravel())[::-1][:3]
    for idx in flat_indices:
        gh, ga = divmod(idx, grid.shape[1])
        top_scores.append(f"{gh}-{ga} ({grid[gh, ga]*100:.1f}%)")
        
    return DrawResolutionPrediction(
        fixture_id=fixture_id,
        probabilities=probs_dict,
        predicted_class=pred_class,
        decision_policy_applied=config.decision_policy,
        p_draw_raw_v4=p_d_v4,
        p_draw_dibp=p_d_dibp,
        p_draw_elo=p_d_elo,
        expected_goals_home=lambda_h,
        expected_goals_away=lambda_a,
        confidence_tier=conf,
        metadata={
            "top_3_scorelines": top_scores,
            "p_0_0": float(grid[0, 0]),
            "p_1_1": float(grid[1, 1]),
            "p_2_2": float(grid[2, 2]),
            "total_expected_goals": lambda_h + lambda_a,
            "abs_elo_difference": abs_elo_diff,
        }
    )
