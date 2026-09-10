"""Candidate Module for Draw-Calibrated Model Layer (v4.1 Candidate).

Model Identifier: v4_1_draw_calibrated_candidate
Version: v4.1-draw-calibrated-candidate
Classification: RESEARCH / SHADOW CANDIDATE ONLY — NOT FOR UNGATED PRODUCTION PROMOTION

Candidate Architecture:
    V4 Poisson Baseline (Frozen)
        ↓
    V4 Lambdas (lambda_home, lambda_away) & Baseline Probabilities [P(H), P(D), P(A)]
        ↓
    +------------------------------------+   +------------------------------------+
    |     Frozen Dixon-Coles Layer       |   |       Frozen Elo Draw Layer        |
    |  - League-specific rho (Frozen)    |   |  - Pre-match |dElo| / 100 (Frozen) |
    |  - Low-score tau matrix correction |   |  - logit(P(D)_V4) (Frozen)         |
    |  -> P(D)_DC                        |   |  -> P(D)_Elo                       |
    +------------------------------------+   +------------------------------------+
        ↓                                       ↓
    +-----------------------------------------------------------------------------+
    |                    CANDIDATE Logistic Stacking Calibration                  |
    |  logit(P(D)_new) = w_0 + 0.6037 * logit(P_DC) + 0.4812 * logit(P_Elo)       |
    |  where w_0 is the candidate stacking intercept under investigation          |
    +-----------------------------------------------------------------------------+
        ↓
    +-----------------------------------------------------------------------------+
    |                    Proportional Odds Simplex Redistribution                 |
    |  P(H)_new = P(H)_V4 * (1 - P(D)_new) / (1 - P(D)_V4)                        |
    |  P(A)_new = P(A)_V4 * (1 - P(D)_new) / (1 - P(D)_V4)                        |
    +-----------------------------------------------------------------------------+
        ↓
    Candidate Probabilities [P(H), P(D), P(A)] + Original V4 Shadow Retained

INVARIANTS & INTEGRITY:
- Fully deterministic pure functions.
- Preserves exact conditional relative odds: P(H)_new / P(A)_new == P(H)_V4 / P(A)_V4.
- Sums strictly to 1.0 (simplex normalization) and strictly non-negative.
- Zero market odds used as model input.
- Causal pre-match information only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import CLASS_ORDER
from .draw_champion import (
    DrawChampionConfig,
    DrawChampionError,
    compute_dc_draw_probability,
    compute_elo_draw_probability,
    redistribute_proportional_odds,
    stable_logit,
    stable_sigmoid,
)
from .poisson import TAIL_TOL, _grid_size, modal_scoreline

CANDIDATE_MODEL_ID = "v4_1_draw_calibrated_candidate"
CANDIDATE_MODEL_VERSION = "v4.1-draw-calibrated-candidate"
FROZEN_BASELINE_INTERCEPT = 0.1130


class DrawChampionV41Error(RuntimeError):
    """Raised when a candidate validation or mathematical invariant gate fails."""


@dataclass(frozen=True)
class DrawChampionV41Config:
    """Configuration for the Candidate Draw Champion Layer."""
    stacking_intercept: float = 0.2000  # Candidate parameter under investigation
    stacking_weight_dc: float = 0.6037  # Frozen research weight
    stacking_weight_elo: float = 0.4812  # Frozen research weight

    elo_intercept: float = 0.2227  # Frozen research weight
    elo_slope_logit_v4: float = 1.1278  # Frozen research weight
    elo_slope_abs_elo: float = -0.1652  # Frozen research weight

    league_rhos: dict[str, float] = None
    global_fallback_rho: float = -0.0560

    def __post_init__(self):
        if self.league_rhos is None:
            object.__setattr__(self, "league_rhos", {
                "Bundesliga": -0.0768,
                "Ligue 1": -0.0583,
                "Serie A": -0.0468,
                "La Liga": -0.0387,
                "Premier League": -0.0163,
            })

    def with_intercept(self, new_intercept: float) -> DrawChampionV41Config:
        """Return a new config instance with a modified stacking intercept."""
        return DrawChampionV41Config(
            stacking_intercept=float(new_intercept),
            stacking_weight_dc=self.stacking_weight_dc,
            stacking_weight_elo=self.stacking_weight_elo,
            elo_intercept=self.elo_intercept,
            elo_slope_logit_v4=self.elo_slope_logit_v4,
            elo_slope_abs_elo=self.elo_slope_abs_elo,
            league_rhos=self.league_rhos,
            global_fallback_rho=self.global_fallback_rho,
        )

    def to_prod_config(self) -> DrawChampionConfig:
        """Convert to production DrawChampionConfig object."""
        return DrawChampionConfig(
            stacking_intercept=self.stacking_intercept,
            stacking_weight_dc=self.stacking_weight_dc,
            stacking_weight_elo=self.stacking_weight_elo,
            elo_intercept=self.elo_intercept,
            elo_slope_logit_v4=self.elo_slope_logit_v4,
            elo_slope_abs_elo=self.elo_slope_abs_elo,
            league_rhos=self.league_rhos,
            global_fallback_rho=self.global_fallback_rho,
        )


@dataclass(frozen=True)
class CandidateDrawPrediction:
    """Individual match candidate prediction carrying full provenance."""
    prediction: str
    probabilities: dict[str, float]
    v4_probabilities: dict[str, float]
    p_draw_dc: float
    p_draw_elo: float
    p_draw_champion: float
    expected_goals_home: float
    expected_goals_away: float
    modal_scoreline: str
    modal_scoreline_probability: float
    lambda_home: float
    lambda_away: float
    fixture_id: int | None = None
    model_name: str = CANDIDATE_MODEL_ID
    model_id: str = CANDIDATE_MODEL_ID
    model_version: str = CANDIDATE_MODEL_VERSION


def stack_draw_probabilities_v41(
    p_draw_dc: float | np.ndarray,
    p_draw_elo: float | np.ndarray,
    config: DrawChampionV41Config,
) -> float | np.ndarray:
    """Stack DC and Elo draw signals into candidate draw probability P(D)_new."""
    z_dc = stable_logit(p_draw_dc)
    z_elo = stable_logit(p_draw_elo)
    z_champ = config.stacking_intercept + config.stacking_weight_dc * z_dc + config.stacking_weight_elo * z_elo
    return stable_sigmoid(z_champ)


def predict_draw_champion_v41(
    lam_h: float | np.ndarray,
    lam_a: float | np.ndarray,
    p_v4: np.ndarray,
    abs_elo_diff: float | np.ndarray,
    league_name: str | list[str] | np.ndarray,
    config: DrawChampionV41Config | None = None,
    class_order: list[str] = list(CLASS_ORDER),
) -> list[CandidateDrawPrediction]:
    """Generate candidate draw predictions using configured stacking intercept."""
    if config is None:
        config = DrawChampionV41Config()

    lam_h_arr = np.asarray(lam_h, dtype=float)
    lam_a_arr = np.asarray(lam_a, dtype=float)
    if lam_h_arr.ndim == 0:
        lam_h_arr = np.array([float(lam_h_arr)])
        lam_a_arr = np.array([float(lam_a_arr)])

    n = len(lam_h_arr)
    p_v4_arr = np.asarray(p_v4, dtype=float)
    if p_v4_arr.ndim == 1:
        p_v4_arr = p_v4_arr[None, :]

    if len(p_v4_arr) != n or p_v4_arr.shape[1] != 3:
        raise DrawChampionV41Error(f"p_v4 must have shape ({n}, 3), got {p_v4_arr.shape}")

    if np.isscalar(abs_elo_diff):
        elo_diff_arr = np.full(n, float(abs_elo_diff))
    else:
        elo_diff_arr = np.asarray(abs_elo_diff, dtype=float)

    if isinstance(league_name, str):
        leagues = [league_name] * n
    else:
        leagues = list(league_name)

    # 1. Resolve Dixon-Coles rho per fixture
    rhos = np.array([config.league_rhos.get(lg, config.global_fallback_rho) for lg in leagues])

    # 2. Compute DC Draw Probabilities
    p_dc_draw = compute_dc_draw_probability(lam_h_arr, lam_a_arr, rhos)

    # 3. Compute Elo Draw Probabilities
    prod_cfg = config.to_prod_config()
    p_elo_draw = compute_elo_draw_probability(p_v4_arr[:, 1], elo_diff_arr, prod_cfg)

    # 4. Stacking Integration Layer
    p_champ_draw = stack_draw_probabilities_v41(p_dc_draw, p_elo_draw, config)

    # 5. Proportional Odds Redistribution
    p_champ_all = redistribute_proportional_odds(p_v4_arr, p_champ_draw)

    # 6. Modal Scorelines
    lam_max = float(max(np.max(lam_h_arr), np.max(lam_a_arr)))
    K = max(_grid_size(lam_max, TAIL_TOL), 1)
    sh, sa, sp = modal_scoreline(lam_h_arr, lam_a_arr, K)

    results: list[CandidateDrawPrediction] = []
    for i in range(n):
        pred_idx = int(p_champ_all[i].argmax())
        pred_class = class_order[pred_idx]

        champ_probs = {cls: float(p_champ_all[i, j]) for j, cls in enumerate(class_order)}
        v4_probs = {cls: float(p_v4_arr[i, j]) for j, cls in enumerate(class_order)}

        results.append(CandidateDrawPrediction(
            prediction=pred_class,
            probabilities=champ_probs,
            v4_probabilities=v4_probs,
            p_draw_dc=float(p_dc_draw[i]),
            p_draw_elo=float(p_elo_draw[i]),
            p_draw_champion=float(p_champ_draw[i]),
            expected_goals_home=float(lam_h_arr[i]),
            expected_goals_away=float(lam_a_arr[i]),
            modal_scoreline=f"{int(sh[i])}-{int(sa[i])}",
            modal_scoreline_probability=float(sp[i]),
            lambda_home=float(lam_h_arr[i]),
            lambda_away=float(lam_a_arr[i]),
            model_name=CANDIDATE_MODEL_ID,
            model_version=CANDIDATE_MODEL_VERSION,
        ))

    return results


__all__ = [
    "CANDIDATE_MODEL_ID",
    "CANDIDATE_MODEL_VERSION",
    "FROZEN_BASELINE_INTERCEPT",
    "DrawChampionV41Config",
    "CandidateDrawPrediction",
    "DrawChampionV41Error",
    "stack_draw_probabilities_v41",
    "predict_draw_champion_v41",
]
