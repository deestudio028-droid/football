"""Phase 41 — Experiment E16: Causal Pre-Match Reliability & Confidence Features.

Extracts strictly causal pre-match features prior to kickoff:
- E10 probability geometry (entropy, normalized entropy, concentration, margins, spread)
- Scoring environment and Poisson intensity ratios
- Elo strength differences and parity indicators
- Early season and sample maturity counters
- Epistemic model agreement signals
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def extract_reliability_features(
    meta_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    probs_e10: np.ndarray,
    probs_e13: np.ndarray,
    v4_lam_h: np.ndarray,
    v4_lam_a: np.ndarray,
    home_elo: np.ndarray,
    away_elo: np.ndarray,
    team_samples_h: np.ndarray,
    team_samples_a: np.ndarray,
) -> pd.DataFrame:
    """Extracts strictly causal pre-match features for reliability and confidence prediction."""
    # 1. Probability Geometry
    p = np.clip(probs_e10, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    p_home = p[:, 0]
    p_draw = p[:, 1]
    p_away = p[:, 2]

    entropy = -np.sum(p * np.log(p), axis=1)
    norm_entropy = entropy / math.log(3.0)  # [0, 1] range

    sorted_p = np.sort(p, axis=1)
    min_prob = sorted_p[:, 0]
    second_prob = sorted_p[:, 1]
    max_prob = sorted_p[:, 2]

    prob_spread = max_prob - min_prob
    margin_p1_p2 = max_prob - second_prob
    prob_concentration = np.sum(p ** 2, axis=1)  # Herfindahl index in [1/3, 1]

    # 2. Intensity & Scoring Environment
    total_lam = v4_lam_h + v4_lam_a
    min_lam = np.maximum(np.minimum(v4_lam_h, v4_lam_a), 0.10)
    max_lam = np.maximum(v4_lam_h, v4_lam_a)
    lambda_ratio = max_lam / min_lam

    # 3. Elo Metrics
    signed_elo_diff = (home_elo + 100.0) - away_elo
    abs_elo_diff = np.abs(signed_elo_diff)
    elo_parity_tight = (abs_elo_diff <= 25.0).astype(int)
    elo_parity_wide = (abs_elo_diff > 100.0).astype(int)

    # 4. Season & Sample Maturity
    fx = fixtures_df.copy()
    fx["season_match_num"] = fx.groupby(["competition_id", "season_id"])["unix"].rank(method="dense").astype(int)
    fx_match_num_map = fx.set_index("fixture_id")["season_match_num"]
    season_match_num = meta_df["fixture_id"].map(fx_match_num_map).fillna(10).astype(int).values

    min_samples = np.minimum(team_samples_h, team_samples_a)
    is_promoted_sparse = (min_samples < 5).astype(int)
    is_early_3 = (season_match_num <= 3).astype(int)
    is_early_5 = ((season_match_num > 3) & (season_match_num <= 5)).astype(int)

    # 5. Model Disagreement Signals
    q = np.clip(probs_e13, 1e-12, 1.0)
    q = q / q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)
    js_div = 0.5 * (np.sum(p * np.log(p / m), axis=1) + np.sum(q * np.log(q / m), axis=1))
    mean_abs_diff = np.mean(np.abs(p - q), axis=1)

    return pd.DataFrame({
        "fixture_id": meta_df["fixture_id"].values,
        "max_prob_e10": max_prob,
        "min_prob_e10": min_prob,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "prob_spread": prob_spread,
        "entropy": entropy,
        "norm_entropy": norm_entropy,
        "margin_p1_p2": margin_p1_p2,
        "prob_concentration": prob_concentration,
        "total_lam": total_lam,
        "lambda_ratio": lambda_ratio,
        "signed_elo_diff": signed_elo_diff,
        "abs_elo_diff": abs_elo_diff,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_parity_tight": elo_parity_tight,
        "elo_parity_wide": elo_parity_wide,
        "season_match_num": season_match_num,
        "min_team_samples": min_samples,
        "is_promoted_sparse": is_promoted_sparse,
        "is_early_3": is_early_3,
        "is_early_5": is_early_5,
        "js_divergence_e10_e13": js_div,
        "mean_abs_prob_diff": mean_abs_diff,
    })
