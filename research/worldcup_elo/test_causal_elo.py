"""Strict Causal & Integrity Test for FPP Elo Engine."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from elo_engine import EloEngine, load_matches_and_build_elo

MATCHES_DB = Path(__file__).resolve().parents[2] / "data" / "processed" / "matches.db"

def test_truncation_invariance():
    """Verify that future matches do not affect past match features."""
    import sqlite3
    uri = f"file:{MATCHES_DB.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        df = pd.read_sql_query(
            "SELECT fixture_id, unix, season_id, competition_id, home_id, away_id, "
            "home_goals, away_goals, status FROM fixtures ORDER BY unix ASC, fixture_id ASC",
            conn
        )
    finally:
        conn.close()

    engine_full = EloEngine()
    feat_full = engine_full.compute_features_for_matches(df)

    # Test with first 1000, 3000, 5000 matches
    for n in (1000, 3000, 5000):
        df_sub = df.iloc[:n].copy()
        engine_sub = EloEngine()
        feat_sub = engine_sub.compute_features_for_matches(df_sub)
        
        cols = ["home_elo", "away_elo", "elo_diff", "elo_raw_diff", "abs_elo_diff"]
        a = feat_full.iloc[:n][cols].to_numpy()
        b = feat_sub[cols].to_numpy()
        
        max_diff = np.max(np.abs(a - b))
        assert max_diff < 1e-12, f"Truncation leak at n={n}: max_diff={max_diff}"
        print(f"  PASS: Truncation invariance at n={n} (max_diff={max_diff:.2e})")

def test_same_fixture_outcome_insulation():
    """Verify that changing a match outcome does not alter its own pre-match Elo."""
    import sqlite3
    uri = f"file:{MATCHES_DB.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        df = pd.read_sql_query(
            "SELECT fixture_id, unix, season_id, competition_id, home_id, away_id, "
            "home_goals, away_goals, status FROM fixtures ORDER BY unix ASC, fixture_id ASC LIMIT 500",
            conn
        )
    finally:
        conn.close()

    engine1 = EloEngine()
    f1 = engine1.compute_features_for_matches(df)

    # Modify match 250's goals dramatically
    df2 = df.copy()
    df2.loc[250, 'home_goals'] = 10
    df2.loc[250, 'away_goals'] = 0

    engine2 = EloEngine()
    f2 = engine2.compute_features_for_matches(df2)

    cols = ["home_elo", "away_elo", "elo_diff"]
    # Pre-match features for match 250 must be IDENTICAL
    diff_250 = np.max(np.abs(f1.loc[250, cols].to_numpy() - f2.loc[250, cols].to_numpy()))
    assert diff_250 < 1e-12, f"Target outcome leaked into its own pre-match Elo: diff={diff_250}"

    # Pre-match features for matches < 250 must be IDENTICAL
    diff_pre = np.max(np.abs(f1.iloc[:250][cols].to_numpy() - f2.iloc[:250][cols].to_numpy()))
    assert diff_pre < 1e-12, f"Outcome leaked backward: diff={diff_pre}"

    print(f"  PASS: Target fixture outcome insulation (diff_250={diff_250:.2e}, diff_pre={diff_pre:.2e})")

if __name__ == "__main__":
    print("Running Causal Elo Tests...")
    test_truncation_invariance()
    test_same_fixture_outcome_insulation()
    print("ALL CAUSAL TESTS PASSED! ✅")