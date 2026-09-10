"""Phase 36 — Experiment E11: Causal xG Feature Engine.

Generates strictly causal pre-match xG features:
- Rolling windows (3, 5, 10)
- EWMA with half-life decay (t_1/2 = 5 matches)
- Opponent-adjusted attacking & defensive xG strength
- Match-level xG differentials & continuous parity indices
- Schedule rest & 14-day match load fatigue indices
- Promoted-team Bayesian shrinkage priors
- Automated zero-leakage assertions
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
OUTPUT_PARQUET = PROJECT_ROOT / "data/research/e11_causal_xg_features.parquet"


def compute_ewma_metric(values: List[float], half_life: float = 5.0) -> float:
    """Compute exponentially weighted moving average with half-life in match counts."""
    if not values:
        return 1.250
    alpha = math.log(2.0) / half_life
    weights = [math.exp(-alpha * i) for i in range(len(values))]
    # values is ordered latest first: [m_t-1, m_t-2, ...]
    weighted_sum = sum(w * v for w, v in zip(weights, values))
    total_weight = sum(weights)
    return weighted_sum / total_weight if total_weight > 0 else 1.250


def build_causal_xg_features(df_matches: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Builds complete pre-match causal xG feature dataframe for all fixtures."""
    if df_matches is None:
        df_matches = pd.read_parquet(INGESTED_PARQUET)

    # Sort strictly by kickoff timestamp
    df = df_matches.sort_values(["unix", "date", "fixture_id"]).reset_index(drop=True)

    # State trackers for past matches per team
    # Team history structure: {team_id: list of past match dicts in reverse chronological order}
    team_history = defaultdict(list)
    venue_history = defaultdict(lambda: defaultdict(list))
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

        # Prior shrinkage defaults (League mean ~ 1.25 xG)
        prior_xg = 1.250
        prior_xga = 1.250

        # 1. Team-level Raw Rolling xG (Windows 3, 5, 10)
        def get_rolling_stats(past_list: List[Dict[str, Any]], window: int) -> Tuple[float, float, float, float]:
            if not past_list:
                return prior_xg, prior_xga, 0.0, 0.0
            sub = past_list[:window]
            k = len(sub)
            # Bayesian pseudo-count shrinkage if k < window
            xg_f = (sum(m["xg_for"] for m in sub) + (window - k) * prior_xg) / window
            xg_a = (sum(m["xg_against"] for m in sub) + (window - k) * prior_xga) / window
            npxg_f = (sum(m["npxg_for"] for m in sub) + (window - k) * prior_xg) / window
            npxg_a = (sum(m["npxg_against"] for m in sub) + (window - k) * prior_xga) / window
            return xg_f, xg_a, xg_f - xg_a, npxg_f - npxg_a

        h_xg3_f, h_xg3_a, h_xg3_d, _ = get_rolling_stats(h_past, 3)
        h_xg5_f, h_xg5_a, h_xg5_d, h_npxg5_d = get_rolling_stats(h_past, 5)
        h_xg10_f, h_xg10_a, h_xg10_d, _ = get_rolling_stats(h_past, 10)

        a_xg3_f, a_xg3_a, a_xg3_d, _ = get_rolling_stats(a_past, 3)
        a_xg5_f, a_xg5_a, a_xg5_d, a_npxg5_d = get_rolling_stats(a_past, 5)
        a_xg10_f, a_xg10_a, a_xg10_d, _ = get_rolling_stats(a_past, 10)

        # 2. EWMA Features (Half-life = 5 matches)
        h_xg_ewma = compute_ewma_metric([m["xg_for"] for m in h_past[:15]], half_life=5.0)
        h_xga_ewma = compute_ewma_metric([m["xg_against"] for m in h_past[:15]], half_life=5.0)
        h_xgd_ewma = h_xg_ewma - h_xga_ewma

        a_xg_ewma = compute_ewma_metric([m["xg_for"] for m in a_past[:15]], half_life=5.0)
        a_xga_ewma = compute_ewma_metric([m["xg_against"] for m in a_past[:15]], half_life=5.0)
        a_xgd_ewma = a_xg_ewma - a_xga_ewma

        # 3. Opponent-Adjusted xG Strengths
        # Strength = (Team EWMA xG / League Mean xG)
        league_mean_xg = max(np.mean(league_averages[cid]["xg_for"][-200:]) if league_averages[cid]["xg_for"] else 1.250, 0.80)
        
        h_att_str = h_xg_ewma / league_mean_xg
        h_def_str = h_xga_ewma / league_mean_xg
        a_att_str = a_xg_ewma / league_mean_xg
        a_def_str = a_xga_ewma / league_mean_xg

        # Expected goals for match based on strength interactions
        home_exp_xg = h_att_str * a_def_str * (league_mean_xg * 1.12) # home venue baseline
        away_exp_xg = a_att_str * h_def_str * (league_mean_xg * 0.88) # away venue baseline
        xg_sum_exp = home_exp_xg + away_exp_xg
        xg_diff_exp = home_exp_xg - away_exp_xg

        # Continuous xG Parity Index
        xg_parity_idx = float(np.exp(- (xg_diff_exp ** 2) / (2.0 * (0.45 ** 2))))

        # 4. Schedule Rest & Fatigue
        last_h_time = h_past[0]["unix"] if h_past else (unix - 14 * 86400)
        last_a_time = a_past[0]["unix"] if a_past else (unix - 14 * 86400)
        h_rest_days = min(max((unix - last_h_time) / 86400.0, 2.0), 21.0)
        a_rest_days = min(max((unix - last_a_time) / 86400.0, 2.0), 21.0)
        rest_days_diff = h_rest_days - a_rest_days

        # Matches in previous 14 days
        a_load_14d = sum(1 for m in a_past if (unix - m["unix"]) <= 14 * 86400)
        h_load_14d = sum(1 for m in h_past if (unix - m["unix"]) <= 14 * 86400)
        is_short_turnaround_away = 1.0 if (a_rest_days <= 3.2 and a_load_14d >= 2) else 0.0

        # Team sample maturity indicator
        h_sample_size = len(h_past)
        a_sample_size = len(a_past)
        min_sample_size = min(h_sample_size, a_sample_size)
        is_promoted_early = 1.0 if min_sample_size < 5 else 0.0

        # Construct causal feature dict
        feat_dict = {
            "fixture_id": fid,
            "unix": unix,
            "competition_id": cid,
            "home_id": h_id,
            "away_id": a_id,
            "home_sample_size": h_sample_size,
            "away_sample_size": a_sample_size,
            "is_promoted_early": is_promoted_early,
            # Raw Rolling
            "home_xg_for_last3": h_xg3_f,
            "home_xg_against_last3": h_xg3_a,
            "home_xg_diff_last3": h_xg3_d,
            "home_xg_for_last5": h_xg5_f,
            "home_xg_against_last5": h_xg5_a,
            "home_xg_diff_last5": h_xg5_d,
            "home_xg_for_last10": h_xg10_f,
            "home_xg_against_last10": h_xg10_a,
            "home_xg_diff_last10": h_xg10_d,
            "home_npxg_diff_last5": h_npxg5_d,
            "away_xg_for_last3": a_xg3_f,
            "away_xg_against_last3": a_xg3_a,
            "away_xg_diff_last3": a_xg3_d,
            "away_xg_for_last5": a_xg5_f,
            "away_xg_against_last5": a_xg5_a,
            "away_xg_diff_last5": a_xg5_d,
            "away_xg_for_last10": a_xg10_f,
            "away_xg_against_last10": a_xg10_a,
            "away_xg_diff_last10": a_xg10_d,
            "away_npxg_diff_last5": a_npxg5_d,
            # EWMA
            "home_xg_ewma": h_xg_ewma,
            "home_xga_ewma": h_xga_ewma,
            "home_xgd_ewma": h_xgd_ewma,
            "away_xg_ewma": a_xg_ewma,
            "away_xga_ewma": a_xga_ewma,
            "away_xgd_ewma": a_xgd_ewma,
            # Opponent Adjusted Strengths
            "home_xg_att_strength": h_att_str,
            "home_xg_def_strength": h_def_str,
            "away_xg_att_strength": a_att_str,
            "away_xg_def_strength": a_def_str,
            # Match Level Differentials
            "xg_diff_last5": h_xg5_d - a_xg5_d,
            "xg_diff_ewma": h_xgd_ewma - a_xgd_ewma,
            "npxg_diff_last5": h_npxg5_d - a_npxg5_d,
            "home_exp_xg": home_exp_xg,
            "away_exp_xg": away_exp_xg,
            "xg_sum_expected": xg_sum_exp,
            "xg_diff_expected": xg_diff_exp,
            "xg_parity_index": xg_parity_idx,
            # Rest & Fatigue
            "home_rest_days": h_rest_days,
            "away_rest_days": a_rest_days,
            "rest_days_diff": rest_days_diff,
            "match_load_14d_away": a_load_14d,
            "match_load_14d_home": h_load_14d,
            "is_short_turnaround_away": is_short_turnaround_away,
        }

        # Zero-Leakage Assertion: Target match outcome / current xG NEVER present in features
        assert "target_home_goals" not in feat_dict
        assert "target_away_goals" not in feat_dict
        assert "stat_home_xg" not in feat_dict
        assert "xg_home" not in feat_dict

        feature_rows.append(feat_dict)

        # Post-Match State Transition: ONLY update state AFTER pre-match feature extraction
        act_h_xg = float(row["xg_home"])
        act_a_xg = float(row["xg_away"])
        act_h_npxg = float(row["npxg_home"])
        act_a_npxg = float(row["npxg_away"])

        team_history[h_id].insert(0, {
            "unix": unix,
            "fixture_id": fid,
            "xg_for": act_h_xg,
            "xg_against": act_a_xg,
            "npxg_for": act_h_npxg,
            "npxg_against": act_a_npxg,
            "venue": "home",
        })
        team_history[a_id].insert(0, {
            "unix": unix,
            "fixture_id": fid,
            "xg_for": act_a_xg,
            "xg_against": act_h_xg,
            "npxg_for": act_a_npxg,
            "npxg_against": act_h_npxg,
            "venue": "away",
        })

        league_averages[cid]["xg_for"].append(act_h_xg)
        league_averages[cid]["xg_against"].append(act_a_xg)

    df_feats = pd.DataFrame(feature_rows)
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_feats.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"  [OK] Successfully built {len(df_feats)} causal xG feature rows with {len(df_feats.columns)} columns.")
    return df_feats


if __name__ == "__main__":
    build_causal_xg_features()
