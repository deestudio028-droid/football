"""Phase 40 — Experiment E15: Pre-Match Risk & Difficulty Feature Extractor.

Extracts strictly causal pre-match features prior to kickoff:
- E10 prediction geometry (entropy, max probability, probability margin, draw probability)
- Intensity characteristics (total lambda, lambda ratio)
- Elo parity metrics (absolute Elo difference, team ratings)
- Sample maturity & early-season indicators
- Model disagreement signals (Jensen-Shannon divergence between E10 and E13)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def compute_entropy_and_geometry(probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes Shannon entropy, max probability, and probability margin."""
    probs = np.clip(probs, 1e-12, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    
    sorted_probs = np.sort(probs, axis=1)
    max_prob = sorted_probs[:, -1]
    second_prob = sorted_probs[:, -2]
    margin = max_prob - second_prob
    
    return entropy, max_prob, margin


def build_pre_match_risk_features(
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
    """Builds tabular pre-match difficulty and error-risk feature matrix."""
    # 1. Prediction Geometry
    entropy, max_prob, margin = compute_entropy_and_geometry(probs_e10)
    p_draw = probs_e10[:, 1]
    p_home = probs_e10[:, 0]
    p_away = probs_e10[:, 2]

    # 2. Intensity & Scoring Environment
    total_lam = v4_lam_h + v4_lam_a
    min_lam = np.maximum(np.minimum(v4_lam_h, v4_lam_a), 0.10)
    max_lam = np.maximum(v4_lam_h, v4_lam_a)
    lambda_ratio = max_lam / min_lam

    # 3. Elo Metrics
    abs_elo_diff = np.abs((home_elo + 100.0) - away_elo)

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
    # Jensen-Shannon Divergence
    p = np.clip(probs_e10, 1e-12, 1.0)
    q = np.clip(probs_e13, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)
    js_div = 0.5 * (np.sum(p * np.log(p / m), axis=1) + np.sum(q * np.log(q / m), axis=1))
    mean_abs_diff = np.mean(np.abs(p - q), axis=1)

    risk_df = pd.DataFrame({
        "fixture_id": meta_df["fixture_id"].values,
        "entropy_e10": entropy,
        "max_prob_e10": max_prob,
        "margin_p1_p2": margin,
        "p_draw_e10": p_draw,
        "p_home_e10": p_home,
        "p_away_e10": p_away,
        "total_lam": total_lam,
        "lambda_ratio": lambda_ratio,
        "abs_elo_diff": abs_elo_diff,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "season_match_num": season_match_num,
        "min_team_samples": min_samples,
        "is_promoted_sparse": is_promoted_sparse,
        "is_early_3": is_early_3,
        "is_early_5": is_early_5,
        "js_divergence_e10_e13": js_div,
        "mean_abs_prob_diff": mean_abs_diff,
    })

    return risk_df
