"""Phase 38 — Experiment E13: Dynamic Causal Elo Module.

Components:
- Adaptive K-factor: K_i(t) = K_base * (1.0 + gamma / sqrt(n_i(t) + 1))
- Season Transition Regression: R_i(s+1, 0) = alpha * R_i(s, final) + (1 - alpha) * R_init
- Attack Elo & Defense Elo ratings
- Causal two-pass execution per timestamp
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

INIT_RATING: float = 1500.0
BASE_K: float = 20.0
HOME_ADVANTAGE: float = 100.0
SEASON_REGRESSION_ALPHA: float = 0.85


def compute_dynamic_elo_features(
    fixtures_df: pd.DataFrame,
    k_base: float = 20.0,
    gamma: float = 1.5,
    season_alpha: float = 0.85,
    track_attack_defense: bool = True,
) -> pd.DataFrame:
    """Computes causal dynamic Elo features with adaptive K and season transition shrinkage."""
    df = fixtures_df.sort_values(["unix", "date", "fixture_id"]).reset_index(drop=True)

    ratings: Dict[int, float] = defaultdict(lambda: INIT_RATING)
    match_counts: Dict[int, int] = defaultdict(int)
    current_season: Dict[int, str] = {}

    # Separate Attack / Defense Elo ratings
    att_ratings: Dict[int, float] = defaultdict(lambda: INIT_RATING)
    def_ratings: Dict[int, float] = defaultdict(lambda: INIT_RATING)

    records: List[Dict[str, Any]] = []

    for unix_ts, group in df.groupby("unix", sort=True):
        # Season boundary transition check
        for row in group.itertuples(index=False):
            s_name = str(row.season)
            for tid in (row.home_id, row.away_id):
                if tid in current_season and current_season[tid] != s_name:
                    # New season: apply regression toward mean
                    ratings[tid] = season_alpha * ratings[tid] + (1.0 - season_alpha) * INIT_RATING
                    att_ratings[tid] = season_alpha * att_ratings[tid] + (1.0 - season_alpha) * INIT_RATING
                    def_ratings[tid] = season_alpha * def_ratings[tid] + (1.0 - season_alpha) * INIT_RATING
                current_season[tid] = s_name

        # --- PASS 1: Extract Pre-Match Dynamic Elo Features ---
        for row in group.itertuples(index=False):
            h_id = row.home_id
            a_id = row.away_id
            r_h = ratings[h_id]
            r_a = ratings[a_id]
            n_h = match_counts[h_id]
            n_a = match_counts[a_id]

            k_h = k_base * (1.0 + gamma / math.sqrt(n_h + 1))
            k_a = k_base * (1.0 + gamma / math.sqrt(n_a + 1))

            records.append({
                "fixture_id": row.fixture_id,
                "dynamic_home_elo": r_h,
                "dynamic_away_elo": r_a,
                "dynamic_elo_diff": (r_h + HOME_ADVANTAGE) - r_a,
                "home_k_factor": k_h,
                "away_k_factor": k_a,
                "home_att_elo": att_ratings[h_id],
                "home_def_elo": def_ratings[h_id],
                "away_att_elo": att_ratings[a_id],
                "away_def_elo": def_ratings[a_id],
                "dyn_elo_att_diff": (att_ratings[h_id] + 50.0) - def_ratings[a_id],
                "dyn_elo_def_diff": (att_ratings[a_id]) - (def_ratings[h_id] + 50.0),
            })

        # --- PASS 2: Post-Match Rating Updates ---
        for row in group.itertuples(index=False):
            if row.status not in ("FT", "AWARDED") or pd.isna(row.home_goals) or pd.isna(row.away_goals):
                continue

            h_id = row.home_id
            a_id = row.away_id
            hg = int(row.home_goals)
            ag = int(row.away_goals)

            r_h = ratings[h_id]
            r_a = ratings[a_id]
            n_h = match_counts[h_id]
            n_a = match_counts[a_id]

            # Standard Elo expected score
            exp_h = 1.0 / (1.0 + 10.0 ** (-((r_h + HOME_ADVANTAGE) - r_a) / 400.0))
            exp_a = 1.0 - exp_h

            if hg > ag:
                s_h, s_a = 1.0, 0.0
            elif hg == ag:
                s_h, s_a = 0.5, 0.5
            else:
                s_h, s_a = 0.0, 1.0

            # Goal difference multiplier
            diff = abs(hg - ag)
            mult = 1.0 if diff <= 1 else (1.5 if diff == 2 else (11.0 + diff) / 8.0)

            k_h = k_base * (1.0 + gamma / math.sqrt(n_h + 1))
            k_a = k_base * (1.0 + gamma / math.sqrt(n_a + 1))

            ratings[h_id] = r_h + k_h * mult * (s_h - exp_h)
            ratings[a_id] = r_a + k_a * mult * (s_a - exp_a)
            match_counts[h_id] += 1
            match_counts[a_id] += 1

            # Attack / Defense Elo updates
            if track_attack_defense:
                exp_hg = math.exp(0.15 + (att_ratings[h_id] - def_ratings[a_id]) / 400.0)
                exp_ag = math.exp(-0.15 + (att_ratings[a_id] - def_ratings[h_id]) / 400.0)

                err_h = hg - exp_hg
                err_a = ag - exp_ag

                att_ratings[h_id] += 5.0 * err_h
                def_ratings[a_id] -= 5.0 * err_h
                att_ratings[a_id] += 5.0 * err_a
                def_ratings[h_id] -= 5.0 * err_a

    return pd.DataFrame(records)
