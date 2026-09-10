"""Phase 30: Historical-Only Draw Model Research Candidates.

Provides model candidates trained strictly on pre-2025/26 historical data (N=8,983 matches),
completely isolated from 2025/26 fixtures, prospective data, and future outcomes.

CANDIDATES:
- Candidate A: Historical Logistic Draw Intercept (w0 Recalibration)
- Candidate B: Historical Platt Draw Calibration
- Candidate C: Historical Dixon-Coles rho Enhancement
- Candidate D: Historical Scoreline Diagonal Aggregation
- Candidate E: Historical Calibrated Stacking (Pre-2025/26 MLE)
- Candidate F: Historical DIBP + Calibrated Stacking
- Candidate G: Historical Physical Draw Probability Layer
- Candidate H: Historical Combined Probability + Selective Draw Gate Decision Layer
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

CLASS_ORDER: tuple[str, str, str] = ("H", "D", "A")


@dataclass(frozen=True)
class HistoricalCandidateManifest:
    """Cryptographic and parameter manifest for frozen historical candidate."""
    candidate_id: str
    version: str
    training_cutoff_season: str
    training_match_count: int
    stacking_intercept: float
    dibp_inflation_p: float
    dixon_coles_rho: float
    draw_prob_threshold: float
    winner_margin_cap: float
    v4_winner_conf_cap: float
    abs_elo_cap: float
    tot_expected_goals_cap: float


def stable_logit(p: float | np.ndarray, eps: float = 1e-7) -> float | np.ndarray:
    p_c = np.clip(p, eps, 1.0 - eps)
    return np.log(p_c / (1.0 - p_c))


def stable_sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    z_c = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z_c))


def redistribute_proportional_1x2(p_v4: np.ndarray, p_draw_new: float) -> np.ndarray:
    """Redistribute remaining probability proportionally across H and A."""
    p_h, p_d, p_a = p_v4[0], p_v4[1], p_v4[2]
    rem_target = 1.0 - p_draw_new
    sum_ha = p_h + p_a
    if sum_ha <= 1e-12:
        new_h = rem_target / 2.0
        new_a = rem_target / 2.0
    else:
        new_h = rem_target * (p_h / sum_ha)
        new_a = rem_target * (p_a / sum_ha)
    return np.array([new_h, p_draw_new, new_a], dtype=float)


def compute_historical_dibp_matrix(
    lh: float,
    la: float,
    rho: float = -0.0560,
    p_inf: float = 0.0500,
    max_goals: int = 10,
) -> np.ndarray:
    """Compute scoreline probability matrix under Dixon-Coles + Diagonal Inflation (DIBP)."""
    grid = np.zeros((max_goals, max_goals), dtype=float)
    pois_h = np.array([math.exp(-lh) * (lh ** i) / math.factorial(i) for i in range(max_goals)])
    pois_a = np.array([math.exp(-la) * (la ** j) / math.factorial(j) for j in range(max_goals)])

    for i in range(max_goals):
        for j in range(max_goals):
            base_p = pois_h[i] * pois_a[j]
            # Dixon-Coles correction
            if i == 0 and j == 0:
                tau = 1.0 - (lh * la * rho)
            elif i == 0 and j == 1:
                tau = 1.0 + (lh * rho)
            elif i == 1 and j == 0:
                tau = 1.0 + (la * rho)
            elif i == 1 and j == 1:
                tau = 1.0 - rho
            else:
                tau = 1.0
            grid[i, j] = max(0.0, base_p * tau)

    grid /= grid.sum()

    # Apply Diagonal Inflation
    diag_sum = sum(grid[k, k] for k in range(max_goals))
    if diag_sum > 0:
        for k in range(max_goals):
            grid[k, k] += p_inf * (grid[k, k] / diag_sum)
    grid /= grid.sum()
    return grid


def predict_historical_candidate_h(
    p_v4: np.ndarray,
    p_v42: np.ndarray,
    lambda_home: float,
    lambda_away: float,
    abs_elo_diff: float,
    manifest: HistoricalCandidateManifest,
) -> Dict[str, Any]:
    """Execute Candidate H (Historical Combined Probability + Selective Draw Gate)."""
    v4_base_dec = CLASS_ORDER[int(np.argmax(p_v4))]
    v4_conf = float(max(p_v4[0], p_v4[2]))
    p_d_v42 = float(p_v42[1])
    winner_margin = float(max(p_v42[0], p_v42[2]) - p_d_v42)
    tot_goals = float(lambda_home + lambda_away)

    gate_1 = (p_d_v42 >= manifest.draw_prob_threshold)
    gate_2 = (winner_margin <= manifest.winner_margin_cap)
    gate_3 = (v4_conf <= manifest.v4_winner_conf_cap)
    gate_4 = (abs_elo_diff <= manifest.abs_elo_cap)
    gate_5 = (tot_goals <= manifest.tot_expected_goals_cap)

    override_active = (gate_1 and gate_2 and gate_3 and gate_4 and gate_5)
    final_dec = "D" if override_active else v4_base_dec

    return {
        "v4_base_decision": v4_base_dec,
        "v4_6_final_decision": final_dec,
        "override_applied": bool(override_active),
        "p_v4": p_v4,
        "p_v42": p_v42,
        "gate_checks": {
            "gate_1_draw_prob": bool(gate_1),
            "gate_2_winner_margin": bool(gate_2),
            "gate_3_v4_conf": bool(gate_3),
            "gate_4_abs_elo": bool(gate_4),
            "gate_5_tot_goals": bool(gate_5),
        }
    }
