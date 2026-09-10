"""Isolated Causal Elo Engine for Football Prediction Project (FPP).

Strict Causality Invariant:
For every fixture at timestamp T, pre-match ratings depend EXCLUSIVELY
on fixtures played strictly before T. No current fixture outcome or future
fixture information is ever accessed.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def goal_diff_multiplier(diff: int) -> float:
    """eloratings.net standard margin multiplier."""
    ad = abs(int(diff))
    if ad <= 1:
        return 1.0
    elif ad == 2:
        return 1.5
    else:
        return (11.0 + ad) / 8.0


@dataclass
class EloEngine:
    init_rating: float = 1500.0
    k_factor: float = 20.0
    home_advantage: float = 100.0
    use_goal_diff: bool = True
    mean_reversion: float = 0.0  # gamma for season transition mean reversion
    ratings: Dict[int, float] = field(default_factory=lambda: defaultdict(lambda: 1500.0))

    def expected_score(self, r_home: float, r_away: float, include_home_adv: bool = True) -> float:
        h_adv = self.home_advantage if include_home_adv else 0.0
        delta = (r_home + h_adv) - r_away
        return 1.0 / (1.0 + 10.0 ** (-delta / 400.0))

    def compute_features_for_matches(self, fixtures_df: pd.DataFrame) -> pd.DataFrame:
        """Process fixtures chronologically and extract pre-match Elo features.

        fixtures_df must contain:
        ['fixture_id', 'unix', 'season_id', 'competition_id', 'home_id', 'away_id',
         'home_goals', 'away_goals', 'status']
        """
        # Ensure chronological ordering
        df = fixtures_df.sort_values(['unix', 'fixture_id']).reset_index(drop=True)
        
        # We track ratings in this engine instance
        ratings = defaultdict(lambda: self.init_rating)
        
        records = []
        
        # Group by timestamp unix to prevent intra-timestamp leakage across simultaneous matches
        grouped = df.groupby('unix', sort=True)
        
        # Track active seasons for mean reversion
        seen_seasons = set()
        
        for unix_ts, group in grouped:
            # 1. First pass: extract pre-match features for ALL fixtures at this timestamp
            pre_match_data = []
            for row in group.itertuples(index=False):
                h_id = row.home_id
                a_id = row.away_id
                comp_id = row.competition_id
                season_id = row.season_id
                
                # Check for season transition mean reversion
                if self.mean_reversion > 0.0:
                    season_key = (comp_id, season_id)
                    if season_key not in seen_seasons and len(seen_seasons) > 0:
                        # New season boundary observed: regress all current ratings towards init_rating
                        for tid in list(ratings.keys()):
                            ratings[tid] = ratings[tid] - self.mean_reversion * (ratings[tid] - self.init_rating)
                        seen_seasons.add(season_key)
                    else:
                        seen_seasons.add(season_key)
                
                r_home = ratings[h_id]
                r_away = ratings[a_id]
                elo_diff = (r_home + self.home_advantage) - r_away
                elo_raw_diff = r_home - r_away
                
                pre_match_data.append({
                    'fixture_id': row.fixture_id,
                    'unix': unix_ts,
                    'season_id': season_id,
                    'competition_id': comp_id,
                    'home_id': h_id,
                    'away_id': a_id,
                    'home_elo': r_home,
                    'away_elo': r_away,
                    'elo_diff': elo_diff,
                    'elo_raw_diff': elo_raw_diff,
                    'abs_elo_diff': abs(elo_diff),
                })
            
            records.extend(pre_match_data)
            
            # 2. Second pass: update ratings with outcomes of completed matches at this timestamp
            for row in group.itertuples(index=False):
                if row.status in ('FT', 'AWARDED') and pd.notna(row.home_goals) and pd.notna(row.away_goals):
                    hg = int(row.home_goals)
                    ag = int(row.away_goals)
                    h_id = row.home_id
                    a_id = row.away_id
                    
                    r_home = ratings[h_id]
                    r_away = ratings[a_id]
                    
                    we = self.expected_score(r_home, r_away, include_home_adv=True)
                    
                    if hg > ag:
                        w = 1.0
                    elif hg == ag:
                        w = 0.5
                    else:
                        w = 0.0
                    
                    g = goal_diff_multiplier(hg - ag) if self.use_goal_diff else 1.0
                    delta_r = self.k_factor * g * (w - we)
                    
                    ratings[h_id] = r_home + delta_r
                    ratings[a_id] = r_away - delta_r
        
        out_df = pd.DataFrame(records)
        return out_df


def load_matches_and_build_elo(matches_db_path: Path, **elo_kwargs) -> pd.DataFrame:
    """Load fixtures from matches.db and compute pre-match Elo features."""
    uri = f"file:{Path(matches_db_path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        fixtures = pd.read_sql_query(
            "SELECT fixture_id, unix, season_id, competition_id, home_id, away_id, "
            "home_goals, away_goals, status FROM fixtures ORDER BY unix ASC, fixture_id ASC",
            conn
        )
    finally:
        conn.close()
    
    engine = EloEngine(**elo_kwargs)
    return engine.compute_features_for_matches(fixtures)