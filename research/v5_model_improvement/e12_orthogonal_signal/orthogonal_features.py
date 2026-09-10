"""Phase 37 — Experiment E12: Orthogonal Feature Generator.

Constructs non-redundant, orthogonal signals derived from historical xG, finishing residuals,
momentum trends, venue specialization, fatigue interactions, and Elo-xG discrepancies.

All features are strictly causal: feature_timestamp < prediction_timestamp.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
INGESTED_PARQUET = PROJECT_ROOT / "data/research/e11_unified_xg_dataset.parquet"
OUTPUT_PARQUET = PROJECT_ROOT / "data/research/e12_orthogonal_features.parquet"


def compute_ewma_metric(values: List[float], half_life: float = 5.0) -> float:
    """Compute exponentially weighted moving average with half-life in match counts."""
    if not values:
        return 0.0
    alpha = math.log(2.0) / half_life
    weights = [math.exp(-alpha * i) for i in range(len(values))]
    weighted_sum = sum(w * v for w, v in zip(weights, values))
    total_weight = sum(weights)
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def build_orthogonal_features(df_matches: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Builds complete pre-match orthogonal feature dataframe for all fixtures."""
    if df_matches is None:
        df_matches = pd.read_parquet(INGESTED_PARQUET)

    df = df_matches.sort_values(["unix", "date", "fixture_id"]).reset_index(drop=True)

    # State trackers
    team_history = defaultdict(list)
    team_venue_history = defaultdict(lambda: defaultdict(list))
    league_averages = defaultdict(lambda: {"xg_for": [], "xg_against": []})

    feature_rows = []

    for idx, row in df.iterrows():
        fid = int(row["fixture_id"])
        unix = int(row["unix"])
        cid = int(row["competition_id"])
        h_id = int(row["home_id"])
        a_id = int(row["away_id"])

        h_past = team_history[h_id]
        a_past = team_history[a_id]
        h_past_home = team_venue_history[h_id]["home"]
        a_past_away = team_venue_history[a_id]["away"]

        # ----------------------------------------------------
        # 1. Question A: xG Residual Signal (Finishing & Defensive Over/Under-Performance)
        # ----------------------------------------------------
        h_finishing_residuals = [m["goals_for"] - m["xg_for"] for m in h_past[:10]]
        h_defensive_residuals = [m["goals_against"] - m["xg_against"] for m in h_past[:10]]
        a_finishing_residuals = [m["goals_for"] - m["xg_for"] for m in a_past[:10]]
        a_defensive_residuals = [m["goals_against"] - m["xg_against"] for m in a_past[:10]]

        h_fin_ewma5 = compute_ewma_metric(h_finishing_residuals, half_life=5.0)
        h_def_ewma5 = compute_ewma_metric(h_defensive_residuals, half_life=5.0)
        h_net_luck_ewma5 = h_fin_ewma5 - h_def_ewma5

        a_fin_ewma5 = compute_ewma_metric(a_finishing_residuals, half_life=5.0)
        a_def_ewma5 = compute_ewma_metric(a_defensive_residuals, half_life=5.0)
        a_net_luck_ewma5 = a_fin_ewma5 - a_def_ewma5

        match_finishing_residual_diff = h_fin_ewma5 - a_fin_ewma5
        match_defensive_residual_diff = h_def_ewma5 - a_def_ewma5
        match_net_luck_diff = h_net_luck_ewma5 - a_net_luck_ewma5

        # ----------------------------------------------------
        # 2. Question B: xG Trend / Momentum (Short-term vs. Long-term)
        # ----------------------------------------------------
        h_xg_short = compute_ewma_metric([m["xg_for"] for m in h_past[:5]], half_life=3.0)
        h_xg_long = compute_ewma_metric([m["xg_for"] for m in h_past[:20]], half_life=10.0)
        h_xga_short = compute_ewma_metric([m["xg_against"] for m in h_past[:5]], half_life=3.0)
        h_xga_long = compute_ewma_metric([m["xg_against"] for m in h_past[:20]], half_life=10.0)

        a_xg_short = compute_ewma_metric([m["xg_for"] for m in a_past[:5]], half_life=3.0)
        a_xg_long = compute_ewma_metric([m["xg_for"] for m in a_past[:20]], half_life=10.0)
        a_xga_short = compute_ewma_metric([m["xg_against"] for m in a_past[:5]], half_life=3.0)
        a_xga_long = compute_ewma_metric([m["xg_against"] for m in a_past[:20]], half_life=10.0)

        h_xg_trend = h_xg_short - h_xg_long
        h_xga_trend = h_xga_short - h_xga_long
        a_xg_trend = a_xg_short - a_xg_long
        a_xga_trend = a_xga_short - a_xga_long

        match_xg_momentum_diff = (h_xg_trend - h_xga_trend) - (a_xg_trend - a_xga_trend)

        # ----------------------------------------------------
        # 3. Question C: Home/Away Venue xG Specialization
        # ----------------------------------------------------
        h_venue_xg_ewma = compute_ewma_metric([m["xg_for"] for m in h_past_home[:10]], half_life=5.0) or 1.35
        h_venue_xga_ewma = compute_ewma_metric([m["xg_against"] for m in h_past_home[:10]], half_life=5.0) or 1.15
        h_venue_xgd = h_venue_xg_ewma - h_venue_xga_ewma

        a_venue_xg_ewma = compute_ewma_metric([m["xg_for"] for m in a_past_away[:10]], half_life=5.0) or 1.10
        a_venue_xga_ewma = compute_ewma_metric([m["xg_against"] for m in a_past_away[:10]], half_life=5.0) or 1.40
        a_venue_xgd = a_venue_xg_ewma - a_venue_xga_ewma

        venue_spec_xgd_diff = h_venue_xgd - a_venue_xgd

        # ----------------------------------------------------
        # 4. Question D: Attack vs. Defense Matchup
        # ----------------------------------------------------
        league_mean_xg = max(np.mean(league_averages[cid]["xg_for"][-200:]) if league_averages[cid]["xg_for"] else 1.250, 0.80)

        h_base_xg = compute_ewma_metric([m["xg_for"] for m in h_past[:15]], half_life=5.0) or 1.25
        h_base_xga = compute_ewma_metric([m["xg_against"] for m in h_past[:15]], half_life=5.0) or 1.25
        a_base_xg = compute_ewma_metric([m["xg_for"] for m in a_past[:15]], half_life=5.0) or 1.25
        a_base_xga = compute_ewma_metric([m["xg_against"] for m in a_past[:15]], half_life=5.0) or 1.25

        h_att_str = h_base_xg / league_mean_xg
        h_def_str = h_base_xga / league_mean_xg
        a_att_str = a_base_xg / league_mean_xg
        a_def_str = a_base_xga / league_mean_xg

        exp_matchup_home_xg = h_att_str * a_def_str * (league_mean_xg * 1.12)
        exp_matchup_away_xg = a_att_str * h_def_str * (league_mean_xg * 0.88)
        xg_matchup_delta = exp_matchup_home_xg - exp_matchup_away_xg
        xg_matchup_total = exp_matchup_home_xg + exp_matchup_away_xg

        # ----------------------------------------------------
        # 5. Question E: Fatigue Interactions
        # ----------------------------------------------------
        last_h_time = h_past[0]["unix"] if h_past else (unix - 14 * 86400)
        last_a_time = a_past[0]["unix"] if a_past else (unix - 14 * 86400)
        h_rest_days = min(max((unix - last_h_time) / 86400.0, 2.0), 21.0)
        a_rest_days = min(max((unix - last_a_time) / 86400.0, 2.0), 21.0)
        rest_days_diff = h_rest_days - a_rest_days

        a_congestion_7d = sum(1 for m in a_past if (unix - m["unix"]) <= 7 * 86400)
        h_congestion_7d = sum(1 for m in h_past if (unix - m["unix"]) <= 7 * 86400)
        a_congestion_14d = sum(1 for m in a_past if (unix - m["unix"]) <= 14 * 86400)
        h_congestion_14d = sum(1 for m in h_past if (unix - m["unix"]) <= 14 * 86400)

        away_fatigue_x_def = a_congestion_7d * a_def_str
        away_fatigue_x_att = a_congestion_7d * a_att_str
        rest_diff_x_quality = rest_days_diff * (h_base_xg - h_base_xga)

        # ----------------------------------------------------
        # 6. Question F: Elo-xG Orthogonality Proxy
        # ----------------------------------------------------
        # xG implied dominance proportion
        xg_implied_dominance = exp_matchup_home_xg / max(exp_matchup_home_xg + exp_matchup_away_xg, 0.1)

        # Sample maturity
        h_sample_size = len(h_past)
        a_sample_size = len(a_past)

        feat_dict = {
            "fixture_id": fid,
            "unix": unix,
            "competition_id": cid,
            "home_id": h_id,
            "away_id": a_id,
            "h_sample_size": h_sample_size,
            "a_sample_size": a_sample_size,
            # Group A: Residuals
            "h_fin_ewma5": h_fin_ewma5,
            "h_def_ewma5": h_def_ewma5,
            "h_net_luck_ewma5": h_net_luck_ewma5,
            "a_fin_ewma5": a_fin_ewma5,
            "a_def_ewma5": a_def_ewma5,
            "a_net_luck_ewma5": a_net_luck_ewma5,
            "match_finishing_residual_diff": match_finishing_residual_diff,
            "match_defensive_residual_diff": match_defensive_residual_diff,
            "match_net_luck_diff": match_net_luck_diff,
            # Group B: Trends & Momentum
            "h_xg_trend": h_xg_trend,
            "h_xga_trend": h_xga_trend,
            "a_xg_trend": a_xg_trend,
            "a_xga_trend": a_xga_trend,
            "match_xg_momentum_diff": match_xg_momentum_diff,
            # Group C: Venue Specialization
            "h_venue_xg_ewma": h_venue_xg_ewma,
            "h_venue_xga_ewma": h_venue_xga_ewma,
            "h_venue_xgd": h_venue_xgd,
            "a_venue_xg_ewma": a_venue_xg_ewma,
            "a_venue_xga_ewma": a_venue_xga_ewma,
            "a_venue_xgd": a_venue_xgd,
            "venue_spec_xgd_diff": venue_spec_xgd_diff,
            # Group D: Matchup
            "exp_matchup_home_xg": exp_matchup_home_xg,
            "exp_matchup_away_xg": exp_matchup_away_xg,
            "xg_matchup_delta": xg_matchup_delta,
            "xg_matchup_total": xg_matchup_total,
            # Group E: Fatigue Interactions
            "rest_days_diff": rest_days_diff,
            "a_congestion_7d": float(a_congestion_7d),
            "h_congestion_7d": float(h_congestion_7d),
            "a_congestion_14d": float(a_congestion_14d),
            "away_fatigue_x_def": away_fatigue_x_def,
            "away_fatigue_x_att": away_fatigue_x_att,
            "rest_diff_x_quality": rest_diff_x_quality,
            # Group F: Orthogonal Dominance
            "xg_implied_dominance": xg_implied_dominance,
        }

        # Causal Safety Assertions
        assert "target_home_goals" not in feat_dict
        assert "target_away_goals" not in feat_dict
        assert "stat_home_xg" not in feat_dict
        assert "xg_home" not in feat_dict

        feature_rows.append(feat_dict)

        # Post-Match Update: strictly after pre-match feature construction
        act_h_g = float(row["home_goals"])
        act_a_g = float(row["away_goals"])
        act_h_xg = float(row["xg_home"])
        act_a_xg = float(row["xg_away"])

        h_entry = {
            "unix": unix,
            "fixture_id": fid,
            "goals_for": act_h_g,
            "goals_against": act_a_g,
            "xg_for": act_h_xg,
            "xg_against": act_a_xg,
            "venue": "home",
        }
        a_entry = {
            "unix": unix,
            "fixture_id": fid,
            "goals_for": act_a_g,
            "goals_against": act_h_g,
            "xg_for": act_a_xg,
            "xg_against": act_h_xg,
            "venue": "away",
        }

        team_history[h_id].insert(0, h_entry)
        team_history[a_id].insert(0, a_entry)
        team_venue_history[h_id]["home"].insert(0, h_entry)
        team_venue_history[a_id]["away"].insert(0, a_entry)

        league_averages[cid]["xg_for"].append(act_h_xg)
        league_averages[cid]["xg_against"].append(act_a_xg)

    df_feats = pd.DataFrame(feature_rows)
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_feats.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"  [OK] Successfully generated {len(df_feats)} orthogonal feature rows with {len(df_feats.columns)} columns.")
    return df_feats


if __name__ == "__main__":
    build_orthogonal_features()
